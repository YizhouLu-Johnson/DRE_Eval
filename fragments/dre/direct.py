from dataclasses import dataclass
from typing import Literal, Optional

import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

from .base import BaseDRE, BaseDREConfig
from ..nns.factory import MLPConfig


@dataclass
class DirectDREConfig(BaseDREConfig):
    var_dim: int
    loss: Literal['nce', 'nwj', 'dv'] = 'nce'
    patience: int = 10


class DirectDRE(BaseDRE):
    """Direct density ratio estimator using a single critic network.

    Supports NCE (Noise Contrastive Estimation) and DV (Donsker-Varadhan) losses.
    For DV/NWJ bounds, the partition function is cached after training for
    complete divergence estimation.
    """

    def __init__(self, config: DirectDREConfig):
        self._cached_log_partition: Optional[torch.Tensor] = None
        super().__init__(config)
        # Bind the correct forward method based on loss type
        if self.config.loss == 'nce':
            self.forward = self._forward_nce
        else:
            # DV/NWJ use the same forward
            self.forward = self._forward_dv

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

    def _get_nn_config(self):
        nn_config = MLPConfig(
            in_dim=self.config.var_dim,
            out_dim=1,
            hidden_dim=self.config.nn_hidden_dim,
            num_blocks=self.config.nn_num_blocks,
            depth=self.config.nn_block_depth,
            activation=self.config.nn_hidden_activation,
            norm=self.config.nn_norm,
            dropout=self.config.nn_dropout_p,
            residual_blocks=self.config.nn_is_residual,
        )
        return nn_config

    def _cast_dataset(self, n, d):
        """Create train/validation TensorDatasets from samples.

        Labels are not included as they are inferred within loss functions.
        The dataset returns (numerator_sample, denominator_sample) pairs.

        Args:
            n: Numerator samples tensor of shape (N, var_dim)
            d: Denominator samples tensor of shape (N, var_dim)

        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        # Move to device
        n = n.to(self.config.device)
        d = d.to(self.config.device)

        # Create combined dataset
        dataset = TensorDataset(n, d)

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
        """Execute training loop with validation-based early stopping.

        Args:
            ds_tr: Training dataset
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
            for batch_n, batch_d in train_loader:
                self.optim.zero_grad()

                # Forward pass
                logits_n = self.nn(batch_n).squeeze(-1)
                logits_d = self.nn(batch_d).squeeze(-1)

                # Compute loss
                loss = self.loss(logits_n, logits_d)

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

            # Validation phase (if validation set exists)
            if ds_val is not None:
                self.nn.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_n, batch_d in val_loader:
                        logits_n = self.nn(batch_n).squeeze(-1)
                        logits_d = self.nn(batch_d).squeeze(-1)
                        val_loss += self.loss(logits_n, logits_d).item()

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

        # Cache partition function for DV/NWJ bounds after training
        if self.config.loss in ('dv', 'nwj'):
            # Collect all denominator samples from training and validation sets
            all_d = []
            for _, batch_d in DataLoader(ds_tr, batch_size=len(ds_tr), shuffle=False):
                all_d.append(batch_d)
            if ds_val is not None:
                for _, batch_d in DataLoader(ds_val, batch_size=len(ds_val), shuffle=False):
                    all_d.append(batch_d)
            all_d = torch.cat(all_d, dim=0)
            self._cache_partition(all_d)

    def _cache_partition(self, denominator_samples: torch.Tensor):
        """Cache partition function: log E_q[exp(f(x))].

        Used for DV and NWJ bounds to compute complete divergence estimates.

        Args:
            denominator_samples: Samples from denominator distribution
        """
        with torch.no_grad():
            denominator_samples = denominator_samples.to(self.config.device)
            critic_values = self.logits(denominator_samples)
            n_samples = critic_values.shape[0]
            self._cached_log_partition = (
                torch.logsumexp(critic_values, dim=0) -
                torch.log(torch.tensor(
                    n_samples,
                    dtype=critic_values.dtype,
                    device=critic_values.device,
                ))
            ).detach()

    def logits(self, x):
        """Compute raw network output (logits).

        Args:
            x: Input tensor of shape (batch_size, var_dim)

        Returns:
            Logits tensor of shape (batch_size,)
        """
        x = x.to(self.config.device)
        return self.nn(x).squeeze(-1)

    def forward(self, x):
        """Compute log density ratio. Bound to _forward_nce or _forward_dv in __init__."""
        # This will be overwritten in __init__, but needs to exist for ABC
        raise NotImplementedError("forward should be bound in __init__")

    def _forward_nce(self, x):
        """Compute log density ratio for NCE.

        At optimum: log(p/q) = -logit

        Args:
            x: Input tensor of shape (batch_size, var_dim)

        Returns:
            Log density ratio of shape (batch_size,)
        """
        return -self.logits(x)

    def _forward_dv(self, x):
        """Compute log density ratio for DV/NWJ.

        Critic output estimates log(p/q) + C, where C = log E_q[exp(f)].
        We subtract the cached log partition to get the normalized log ratio.

        Args:
            x: Input tensor of shape (batch_size, var_dim)

        Returns:
            Log density ratio log(p(x)/q(x)) of shape (batch_size,)
        """
        return self.logits(x) - self._cached_log_partition

    @property
    def cached_log_partition(self) -> Optional[torch.Tensor]:
        """Return cached log partition function if available."""
        return self._cached_log_partition
