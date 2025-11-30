"""Telescoping density ratio estimation following Choi et al.

This module implements telescoping DRE where a probability path bridges
numerator and denominator distributions through intermediate distributions.
"""

from dataclasses import dataclass
from typing import Literal, Optional
import copy
import math

import torch
import torch.nn as nn
from torch.func import stack_module_state, functional_call
from torch.utils.data import TensorDataset, DataLoader, random_split

from .base import BaseDRE, BaseDREConfig
from ..nns.factory import MLPConfig, build_mlp
from ..nns.blocks import mlp_block
from ..nns.base import resolve_activation, resolve_norm


# ============================================================================
# Alpha Schedule Utilities
# ============================================================================

def get_alpha_schedule(
    num_steps: int,
    schedule_type: Literal['linear', 'cosine'],
    device: Optional[str] = None,
) -> torch.Tensor:
    """Generate alpha schedule for variance-preserving probability path.

    Alpha values control interpolation: x_t = sqrt(alpha_t) * x_num + sqrt(1 - alpha_t) * x_denom
    - At t=0: alpha=1, so x_0 = x_num (numerator)
    - At t=num_steps: alpha=0, so x_T = x_denom (denominator)

    Args:
        num_steps: Number of telescoping steps (m)
        schedule_type: Type of schedule ('linear' or 'cosine')
        device: Device to place tensor on

    Returns:
        Tensor of shape (num_steps + 1,) with alpha values from 1 to 0
    """
    t = torch.linspace(0, 1, num_steps + 1, device=device)

    if schedule_type == 'linear':
        # Linear: alpha_t = 1 - t
        alphas = 1.0 - t
    elif schedule_type == 'cosine':
        # Cosine: alpha_t = cos(pi * t / 2)^2
        alphas = torch.cos(math.pi * t / 2) ** 2
    else:
        raise ValueError(f"Unknown schedule_type: {schedule_type}. Expected 'linear' or 'cosine'.")

    return alphas


def interpolate_vp(
    x_num: torch.Tensor,
    x_denom: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Interpolate between numerator and denominator using VP path.

    VP-SDE style interpolation:
        x_t = sqrt(alpha) * x_num + sqrt(1 - alpha) * x_denom

    Args:
        x_num: Samples from numerator distribution, shape (batch_size, var_dim)
        x_denom: Samples from denominator distribution, shape (batch_size, var_dim)
        alpha: Interpolation coefficient(s), scalar or shape (batch_size,) or (batch_size, 1)

    Returns:
        Interpolated samples, shape (batch_size, var_dim)
    """
    # Ensure alpha has proper shape for broadcasting
    if alpha.dim() == 0:
        alpha = alpha.unsqueeze(0)
    if alpha.dim() == 1:
        alpha = alpha.unsqueeze(-1)  # (batch_size, 1)

    sqrt_alpha = torch.sqrt(alpha)
    sqrt_one_minus_alpha = torch.sqrt(1.0 - alpha)

    return sqrt_alpha * x_num + sqrt_one_minus_alpha * x_denom


# ============================================================================
# MultiheadTRE Configuration
# ============================================================================

@dataclass
class MultiheadTREConfig(BaseDREConfig):
    """Configuration for MultiheadTRE (telescoping density ratio estimation).

    Args:
        var_dim: Input variable dimension
        num_steps: Number of telescoping pairs (m). Creates m+1 intermediate
            distributions and m DirectDRE heads.
        loss: Loss function type for all heads ('nce', 'nwj', 'dv')
        path_type: Probability path type ('variance-preserving')
        schedule_type: Alpha schedule type ('linear', 'cosine')
        patience: Early stopping patience
    """
    var_dim: int
    num_steps: int = 4
    loss: Literal['nce', 'nwj', 'dv'] = 'nce'
    path_type: Literal['variance-preserving'] = 'variance-preserving'
    schedule_type: Literal['linear', 'cosine'] = 'linear'
    patience: int = 10


# ============================================================================
# MultiheadTRE Implementation
# ============================================================================

class MultiheadTRE(BaseDRE):
    """Telescoping density ratio estimator with shared projector.

    Architecture:
        Input (var_dim) -> Shared Projector (1 block) -> [Head_0, Head_1, ..., Head_{m-1}]
                          (var_dim -> hidden_dim)       (each has num_blocks-1 blocks)

    The telescoping decomposition:
        log(p/q) = sum_{t=0}^{m-1} log(p_t / p_{t+1})

    where p_0 = p (numerator) and p_m = q (denominator), with intermediate
    distributions defined by the variance-preserving probability path.

    Attributes:
        projector: Shared first block (var_dim -> hidden_dim)
        heads: ModuleList of m head networks (hidden_dim -> 1)
        alphas: Tensor of alpha schedule values
    """

    def __init__(self, config: MultiheadTREConfig):
        self._cached_log_partitions: Optional[list[torch.Tensor]] = None
        # Must set these before super().__init__ since _get_nn_config needs them
        self._num_steps = config.num_steps

        # Call nn.Module.__init__ directly, then set up components manually
        # This avoids the issue with BaseDRE creating optimizer before heads exist
        nn.Module.__init__(self)

        self.config = config
        self.nn_config = self._get_nn_config()
        self.nn = build_mlp(self.nn_config)
        self.loss = self._get_loss()
        self.nn.to(self.config.device)

        # Build alpha schedule
        self.register_buffer(
            'alphas',
            get_alpha_schedule(config.num_steps, config.schedule_type, config.device)
        )

        # Build head networks (num_blocks - 1 blocks each, hidden_dim -> 1)
        # This also creates stacked_params, stacked_buffers, and head_meta
        self.heads = self._build_heads()

        # Move heads to device
        self.heads.to(config.device)

        # Move stacked params/buffers to device
        self.stacked_params = {k: v.to(config.device) for k, v in self.stacked_params.items()}
        self.stacked_buffers = {k: v.to(config.device) for k, v in self.stacked_buffers.items()}

        # Create vmapped head forward function
        # Wrapper calls functional_call on the meta model with given params/buffers
        def _head_forward(params, buffers, x):
            return functional_call(self.head_meta, (params, buffers), (x,))

        # vmap over: params (dim 0), buffers (dim 0), input (dim 1 = timestep dimension)
        # Input x has shape (batch, num_timesteps+1, hidden_dim)
        # Output has shape (num_timesteps+1, batch, 1)
        self._vmapped_heads = torch.vmap(
            _head_forward,
            in_dims=(0, 0, 1),
            out_dims=0,
        )

        # Create vmapped loss function for parallel loss computation across heads
        # Input: logits_num and logits_denom both (batch, num_steps)
        # Output: (num_steps,) tensor of per-head losses
        self._vmapped_loss = torch.vmap(self.loss, in_dims=(1, 1))

        # Now create optimizer with all parameters
        self.optim = self._create_optimizer()

        # Freeze all params after init
        for p in self.nn.parameters():
            p.requires_grad = False
        for p in self.heads.parameters():
            p.requires_grad = False

        # Bind forward method based on loss type
        if self.config.loss == 'nce':
            self.forward = self._forward_nce
        else:
            self.forward = self._forward_dv

    def _get_nn_config(self) -> MLPConfig:
        """Return MLPConfig for the shared projector (first block only).

        The projector transforms from var_dim to hidden_dim.
        """
        return MLPConfig(
            in_dim=self.config.var_dim,
            out_dim=self.config.nn_hidden_dim,
            hidden_dim=self.config.nn_hidden_dim,
            num_blocks=1,
            depth=self.config.nn_block_depth,
            activation=self.config.nn_hidden_activation,
            output_activation='identity',
            norm=self.config.nn_norm,
            dropout=self.config.nn_dropout_p,
            residual_blocks=False,  # First block can't be residual (dimension change)
        )

    def _build_heads(self) -> nn.ModuleList:
        """Build m head networks, each with num_blocks - 1 blocks.

        Each head takes hidden_dim input (from projector) and outputs scalar.
        Also stacks all head parameters/buffers for vmapped execution and
        creates a meta model for functional_call.
        """
        heads = nn.ModuleList()

        # Resolve activation and norm
        activation = resolve_activation(self.config.nn_hidden_activation)
        norm = resolve_norm(self.config.nn_norm)

        num_head_blocks = max(1, self.config.nn_num_blocks - 1)

        for _ in range(self.config.num_steps):
            if num_head_blocks == 1:
                # Single block: hidden_dim -> 1
                head = mlp_block(
                    in_dim=self.config.nn_hidden_dim,
                    out_dim=1,
                    hidden_dim=self.config.nn_hidden_dim,
                    depth=self.config.nn_block_depth,
                    activation=activation,
                    output_activation=nn.Identity,
                    norm=norm,
                    dropout=self.config.nn_dropout_p,
                    residual=False,
                )
            else:
                # Multiple blocks: use MLPConfig
                head_config = MLPConfig(
                    in_dim=self.config.nn_hidden_dim,
                    out_dim=1,
                    hidden_dim=self.config.nn_hidden_dim,
                    num_blocks=num_head_blocks,
                    depth=self.config.nn_block_depth,
                    activation=self.config.nn_hidden_activation,
                    output_activation='identity',
                    norm=self.config.nn_norm,
                    dropout=self.config.nn_dropout_p,
                    residual_blocks=self.config.nn_is_residual,
                )
                head = build_mlp(head_config)

            heads.append(head)

        # Stack all head parameters and buffers for vmapped execution
        # stacked_params: dict of {name: tensor of shape (num_heads, ...)}
        # stacked_buffers: dict of {name: tensor of shape (num_heads, ...)}
        self.stacked_params, self.stacked_buffers = stack_module_state(list(heads))

        # Create a meta model (single head) for functional_call
        # Deep copy one head and move to 'meta' device
        self.head_meta = copy.deepcopy(heads[0]).to('meta')

        return heads

    def _get_loss(self):
        """Return loss function based on config.loss setting."""
        if self.config.loss == 'nce':
            from .losses import nce_loss
            return nce_loss
        elif self.config.loss == 'dv':
            from .losses import dv_loss
            return dv_loss
        elif self.config.loss == 'nwj':
            from .losses import nwj_loss
            return nwj_loss
        else:
            raise ValueError(
                f"Invalid loss: {self.config.loss}. "
                f"Expected 'nce', 'dv', or 'nwj'."
            )

    def _cast_dataset(self, n: torch.Tensor, d: torch.Tensor):
        """Create train/validation datasets for telescoping pairs.

        For each pair (n_i, d_i), generates num_steps + 1 interpolated samples
        along the VP path, stacked along the timestep dimension.

        Args:
            n: Numerator samples, shape (N, var_dim)
            d: Denominator samples, shape (N, var_dim)

        Returns:
            Tuple of (train_dataset, val_dataset)
            Each sample in dataset has shape (num_steps + 1, var_dim)
        """
        n = n.to(self.config.device)
        d = d.to(self.config.device)

        # Generate all interpolated samples for all timesteps
        # alphas has shape (num_steps + 1,)
        # For each sample pair, generate num_steps + 1 interpolated points
        batch_size = n.shape[0]

        # Stack all interpolated samples: shape (num_steps + 1, batch_size, var_dim)
        interpolated = []
        for t in range(self.config.num_steps + 1):
            alpha_t = self.alphas[t]
            x_t = interpolate_vp(n, d, alpha_t)
            interpolated.append(x_t)

        # Shape: (num_steps + 1, batch_size, var_dim) -> (batch_size, num_steps + 1, var_dim)
        interpolated = torch.stack(interpolated, dim=0).permute(1, 0, 2)

        # Create TensorDataset with shape (N, num_steps + 1, var_dim)
        dataset = TensorDataset(interpolated)

        # Split into train/validation
        if self.config.val_ratio is not None and self.config.val_ratio > 0:
            n_total = len(dataset)
            n_val = int(n_total * self.config.val_ratio)
            n_train = n_total - n_val
            ds_tr, ds_val = random_split(dataset, [n_train, n_val])
        else:
            ds_tr = dataset
            ds_val = None

        return ds_tr, ds_val

    def _train(self, ds_tr, ds_val):
        """Execute joint training loop for projector and all heads.

        Args:
            ds_tr: Training dataset with samples of shape (num_steps+1, var_dim)
            ds_val: Validation dataset (can be None)
        """
        # Setup data loaders
        train_loader = DataLoader(
            ds_tr,
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        if ds_val is not None:
            val_loader = DataLoader(
                ds_val,
                batch_size=self.config.batch_size,
                shuffle=False,
            )

        # Training state
        best_val_loss = float('inf')
        patience_counter = 0
        patience = self.config.patience

        self.nn.train()
        self.heads.train()

        for epoch in range(self.config.epochs):
            # Training phase
            epoch_train_loss = 0.0
            for (batch_x,) in train_loader:
                # batch_x has shape (batch_size, num_steps+1, var_dim)
                self.optim.zero_grad()

                # Extract numerator samples (timesteps 0..num_steps-1) and
                # denominator samples (timesteps 1..num_steps)
                x_num = batch_x[:, :-1, :]   # (batch, num_steps, var_dim)
                x_denom = batch_x[:, 1:, :]  # (batch, num_steps, var_dim)

                # Compute logits for numerator and denominator samples
                # Each head t processes its corresponding samples
                logits_num = self.logits(x_num).squeeze(-1)      # (batch, num_steps)
                logits_denom = self.logits(x_denom).squeeze(-1)  # (batch, num_steps)

                # Compute per-head losses using vmapped loss and take mean
                total_loss = self._vmapped_loss(logits_num, logits_denom).mean()
                total_loss.backward()

                # Gradient clipping
                if self.config.grad_clip > 0:
                    all_params = list(self.nn.parameters()) + list(self.heads.parameters())
                    torch.nn.utils.clip_grad_norm_(all_params, max_norm=self.config.grad_clip)

                self.optim.step()

                epoch_train_loss += total_loss.item()

            # Validation phase
            if ds_val is not None:
                self.nn.eval()
                self.heads.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for (batch_x,) in val_loader:
                        x_num = batch_x[:, :-1, :]
                        x_denom = batch_x[:, 1:, :]
                        logits_num = self.logits(x_num).squeeze(-1)
                        logits_denom = self.logits(x_denom).squeeze(-1)

                        val_loss += self._vmapped_loss(logits_num, logits_denom).sum().item()

                val_loss /= len(val_loader) * self.config.num_steps

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

                self.nn.train()
                self.heads.train()

        self.nn.eval()
        self.heads.eval()

        # Cache partition functions for DV/NWJ bounds
        if self.config.loss in ('dv', 'nwj'):
            self._cache_partitions(ds_tr, ds_val)

    def _cache_partitions(self, ds_tr, ds_val):
        """Cache partition functions for each head (DV/NWJ bounds).

        For each head t, caches log E_{p_{t+1}}[exp(f_t(x))].
        """
        # Collect all samples
        all_loaders = [DataLoader(ds_tr, batch_size=len(ds_tr), shuffle=False)]
        if ds_val is not None:
            all_loaders.append(DataLoader(ds_val, batch_size=len(ds_val), shuffle=False))

        # Collect all interpolated samples
        all_samples = []
        for loader in all_loaders:
            for (batch_x,) in loader:
                # batch_x has shape (batch_size, num_steps+1, var_dim)
                all_samples.append(batch_x)

        all_samples = torch.cat(all_samples, dim=0)  # (N, num_steps+1, var_dim)

        # For partition of head t, we need head t applied to samples from timestep t+1
        # Extract denominator samples: timesteps 1..num_steps
        x_denom = all_samples[:, 1:, :]  # (N, num_steps, var_dim)

        # Compute partition for each head using vmapped logits
        self._cached_log_partitions = []
        with torch.no_grad():
            # Get logits: head t applied to samples from timestep t+1
            # all_logits: (N, num_steps, 1)
            all_logits = self.logits(x_denom)

            for t in range(self.config.num_steps):
                # critic_values: (N,)
                critic_values = all_logits[:, t, 0]
                n_samples = critic_values.shape[0]
                log_partition = (
                    torch.logsumexp(critic_values, dim=0) -
                    torch.log(torch.tensor(n_samples, dtype=critic_values.dtype, device=critic_values.device))
                ).detach()

                self._cached_log_partitions.append(log_partition)

    def _create_optimizer(self):
        """Create optimizer for projector and all heads."""
        # Collect all parameters
        all_params = list(self.nn.parameters()) + list(self.heads.parameters())

        if self.config.optimizer == 'sgd':
            return torch.optim.SGD(
                all_params,
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == 'adam':
            return torch.optim.Adam(
                all_params,
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == 'adamw':
            return torch.optim.AdamW(
                all_params,
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Compute raw network output for all heads using vmapped execution.

        Args:
            x: Input tensor, shape (batch_size, num_steps, var_dim)
               Each x[:, t, :] will be processed by head t.

        Returns:
            Logits tensor, shape (batch_size, num_steps, 1)
            Each logits[:, t, :] corresponds to head t's output on input x[:, t, :].
        """
        x = x.to(self.config.device)
        batch_size, num_steps, var_dim = x.shape

        # Apply shared projector to all timesteps
        # Reshape: (batch, num_steps, var_dim) -> (batch * num_steps, var_dim)
        x_flat = x.reshape(-1, var_dim)
        proj_flat = self.nn(x_flat)
        # Reshape back: (batch * num_steps, hidden_dim) -> (batch, num_steps, hidden_dim)
        proj = proj_flat.reshape(batch_size, num_steps, -1)

        # Apply vmapped heads
        # Input: proj has shape (batch, num_steps, hidden_dim)
        # vmap over: stacked_params (dim 0), stacked_buffers (dim 0), proj (dim 1)
        # This maps head_t to proj[:, t, :]
        # Output: (num_steps, batch, 1)
        out = self._vmapped_heads(self.stacked_params, self.stacked_buffers, proj)

        # Permute to (batch, num_steps, 1)
        out = out.permute(1, 0, 2)

        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log density ratio. Bound to _forward_nce or _forward_dv in __init__."""
        raise NotImplementedError("forward should be bound in __init__")

    def _forward_nce(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log density ratio for NCE using telescoping sum.

        At optimum: log(p/q) = sum_t [-logit_t]

        Args:
            x: Input tensor, shape (batch_size, var_dim)

        Returns:
            Log density ratio, shape (batch_size,)
        """
        x = x.to(self.config.device)
        batch_size = x.shape[0]

        # Broadcast x across all heads for vmapped forward
        # Shape: (batch, var_dim) -> (batch, num_steps, var_dim)
        x_expanded = x.unsqueeze(1).expand(-1, self.config.num_steps, -1)

        # Get all logits: (batch, num_steps, 1)
        all_logits = self.logits(x_expanded)

        # Sum negative logits across all heads
        log_ratio = -all_logits[:, :, 0].sum(dim=1)

        return log_ratio

    def _forward_dv(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log density ratio for DV/NWJ using telescoping sum.

        Each head's contribution: logit_t - cached_log_partition_t

        Args:
            x: Input tensor, shape (batch_size, var_dim)

        Returns:
            Log density ratio, shape (batch_size,)
        """
        x = x.to(self.config.device)
        batch_size = x.shape[0]

        # Broadcast x across all heads for vmapped forward
        # Shape: (batch, var_dim) -> (batch, num_steps, var_dim)
        x_expanded = x.unsqueeze(1).expand(-1, self.config.num_steps, -1)

        # Get all logits: (batch, num_steps, 1)
        all_logits = self.logits(x_expanded)

        # Sum logits across all heads, subtracting partitions
        head_logits = all_logits[:, :, 0]  # (batch, num_steps)

        if self._cached_log_partitions is not None:
            # Stack partitions: (num_steps,) and broadcast
            partitions = torch.stack(self._cached_log_partitions)  # (num_steps,)
            log_ratio = (head_logits - partitions.unsqueeze(0)).sum(dim=1)
        else:
            log_ratio = head_logits.sum(dim=1)

        return log_ratio

    @property
    def cached_log_partitions(self) -> Optional[list[torch.Tensor]]:
        """Return cached log partition functions if available."""
        return self._cached_log_partitions

    def fit(self, n: torch.Tensor, d: torch.Tensor):
        """Train the telescoping DRE on numerator and denominator samples.

        Override to handle both projector and heads parameter freezing.

        Args:
            n: Numerator samples, shape (N, var_dim)
            d: Denominator samples, shape (N, var_dim)
        """
        n, d = self._resample(n, d)
        ds_tr, ds_val = self._cast_dataset(n, d)

        # Unfreeze all parameters
        for p in self.nn.parameters():
            p.requires_grad = True
        for p in self.heads.parameters():
            p.requires_grad = True

        # Recreate optimizer with all parameters
        self.optim = self._create_optimizer()

        self._train(ds_tr, ds_val)

        # Freeze all parameters
        for p in self.nn.parameters():
            p.requires_grad = False
        for p in self.heads.parameters():
            p.requires_grad = False


# ============================================================================
# MultinomialTRE Configuration
# ============================================================================

@dataclass
class MultinomialTREConfig(BaseDREConfig):
    """Configuration for MultinomialTRE (multinomial classification DRE).

    Uses a single (m+1)-class classifier instead of m binary classifiers.
    The density ratio is computed via Bayes' rule:
        log p(x|0)/p(x|m) = log p(0|x) - log p(m|x)
    (since p(0) = p(m) by balanced class construction)

    Args:
        var_dim: Input variable dimension
        num_steps: Number of intermediate distributions (m). Creates m+1 classes.
        path_type: Probability path type ('variance-preserving')
        schedule_type: Alpha schedule type ('linear', 'cosine')
        patience: Early stopping patience
    """
    var_dim: int
    num_steps: int = 4
    path_type: Literal['variance-preserving'] = 'variance-preserving'
    schedule_type: Literal['linear', 'cosine'] = 'linear'
    patience: int = 10


# ============================================================================
# MultinomialTRE Implementation
# ============================================================================

class MultinomialTRE(BaseDRE):
    """Multinomial telescoping density ratio estimator.

    Uses a single (m+1)-class classifier to estimate density ratios between
    intermediate distributions along a probability path.

    Architecture:
        Input (var_dim) -> MLP -> Softmax (m+1 classes)

    The density ratio is computed as:
        log p(x|0)/p(x|m) = log p(0|x) - log p(m|x)

    where class 0 = numerator (alpha=1), class m = denominator (alpha=0).

    Attributes:
        alphas: Tensor of alpha schedule values (m+1 values)
    """

    def __init__(self, config: MultinomialTREConfig):
        self._num_steps = config.num_steps

        # Call nn.Module.__init__ directly
        nn.Module.__init__(self)

        self.config = config
        self.nn_config = self._get_nn_config()
        self.nn = build_mlp(self.nn_config)
        self.loss = self._get_loss()
        self.nn.to(self.config.device)

        # Build alpha schedule
        self.register_buffer(
            'alphas',
            get_alpha_schedule(config.num_steps, config.schedule_type, config.device)
        )

        # Create optimizer
        self.optim = self._get_optim(self.nn)

        # Freeze all params after init
        for p in self.nn.parameters():
            p.requires_grad = False

    def _get_nn_config(self) -> MLPConfig:
        """Return MLPConfig for the classifier network.

        Output dimension is num_steps + 1 (m+1 classes).
        """
        return MLPConfig(
            in_dim=self.config.var_dim,
            out_dim=self.config.num_steps + 1,  # m+1 classes
            hidden_dim=self.config.nn_hidden_dim,
            num_blocks=self.config.nn_num_blocks,
            depth=self.config.nn_block_depth,
            activation=self.config.nn_hidden_activation,
            output_activation='identity',  # Raw logits for cross-entropy
            norm=self.config.nn_norm,
            dropout=self.config.nn_dropout_p,
            residual_blocks=self.config.nn_is_residual,
        )

    def _get_loss(self):
        """Return cross-entropy loss for multinomial classification."""
        return nn.CrossEntropyLoss()

    def _cast_dataset(self, n: torch.Tensor, d: torch.Tensor):
        """Create train/validation datasets for multinomial classification.

        Generates m+1 interpolated distributions and stacks all samples
        with their class labels.

        Args:
            n: Numerator samples, shape (N, var_dim)
            d: Denominator samples, shape (N, var_dim)

        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        n = n.to(self.config.device)
        d = d.to(self.config.device)

        batch_size = n.shape[0]
        num_classes = self.config.num_steps + 1

        # Generate interpolated samples for each class
        all_samples = []
        all_labels = []

        for t in range(num_classes):
            alpha_t = self.alphas[t]
            x_t = interpolate_vp(n, d, alpha_t)
            all_samples.append(x_t)
            all_labels.append(torch.full((batch_size,), t, dtype=torch.long, device=self.config.device))

        # Stack: shape (N * (m+1), var_dim) and (N * (m+1),)
        all_samples = torch.cat(all_samples, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Create TensorDataset
        dataset = TensorDataset(all_samples, all_labels)

        # Split into train/validation
        if self.config.val_ratio is not None and self.config.val_ratio > 0:
            n_total = len(dataset)
            n_val = int(n_total * self.config.val_ratio)
            n_train = n_total - n_val
            ds_tr, ds_val = random_split(dataset, [n_train, n_val])
        else:
            ds_tr = dataset
            ds_val = None

        return ds_tr, ds_val

    def _train(self, ds_tr, ds_val):
        """Execute training loop with cross-entropy loss.

        Args:
            ds_tr: Training dataset with (samples, labels) tuples
            ds_val: Validation dataset (can be None)
        """
        # Setup data loaders
        train_loader = DataLoader(
            ds_tr,
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        if ds_val is not None:
            val_loader = DataLoader(
                ds_val,
                batch_size=self.config.batch_size,
                shuffle=False,
            )

        # Training state
        best_val_loss = float('inf')
        patience_counter = 0
        patience = self.config.patience

        self.nn.train()

        for epoch in range(self.config.epochs):
            # Training phase
            epoch_train_loss = 0.0
            for batch_x, batch_y in train_loader:
                self.optim.zero_grad()

                # Forward pass
                logits = self.nn(batch_x)

                # Compute cross-entropy loss
                loss = self.loss(logits, batch_y)

                # Backward pass
                loss.backward()

                # Gradient clipping
                if self.config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.nn.parameters(),
                        max_norm=self.config.grad_clip,
                    )

                self.optim.step()
                epoch_train_loss += loss.item()

            # Validation phase
            if ds_val is not None:
                self.nn.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        logits = self.nn(batch_x)
                        val_loss += self.loss(logits, batch_y).item()

                val_loss /= len(val_loader)

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

                self.nn.train()

        self.nn.eval()

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Compute raw network output (logits for all classes).

        Args:
            x: Input tensor, shape (batch_size, var_dim)

        Returns:
            Logits tensor, shape (batch_size, num_steps + 1)
        """
        x = x.to(self.config.device)
        return self.nn(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log density ratio log p(x|0)/p(x|m).

        Uses Bayes' rule:
            log p(x|0)/p(x|m) = log p(0|x) - log p(m|x) + log p(m) - log p(0)

        Since classes are balanced: p(0) = p(m), so:
            log p(x|0)/p(x|m) = log p(0|x) - log p(m|x)

        Args:
            x: Input tensor, shape (batch_size, var_dim)

        Returns:
            Log density ratio, shape (batch_size,)
        """
        x = x.to(self.config.device)
        logits = self.nn(x)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # log p(x|0)/p(x|m) = log p(0|x) - log p(m|x)
        # Class 0 = numerator, Class m (last) = denominator
        return log_probs[:, 0] - log_probs[:, -1]
