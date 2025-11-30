from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Literal

import torch
from torch import nn

from ..nns.factory import MLPConfig, build_mlp


@dataclass
class BaseDREConfig:
    var_dim: int
    nn_hidden_dim: Optional[int] = 64
    nn_num_blocks: Optional[int] = 2
    nn_norm: Optional[Literal['layer', 'batch1d', 'none']] = 'layer'
    nn_block_depth: Optional[int] = 3
    nn_hidden_activation: Optional[Literal['relu', 'silu', 'gelu', 'elu', 'leaky_relu', 'tanh']] = 'silu'
    nn_dropout_p: Optional[float] = 0.
    nn_is_residual: Optional[bool] = True

    sampling_strat: Literal['subsample', 'supersample'] = 'supersample'

    optimizer: Literal['sgd', 'adam', 'adamw'] = 'adam'
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    val_ratio: Optional[float] = 0.1
    batch_size: Optional[int] = 64
    epochs: Optional[int] = 1

    device: Optional[str] = None

    def __post_init__(self):
        if self.device is None:
            print(f'Device not set.')
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'


class BaseDRE(ABC, nn.Module):
    def __init__(self, config: BaseDREConfig):
        super().__init__()
        self.config = config
        self.nn_config = self._get_nn_config()
        self.nn = build_mlp(self.nn_config)
        self.optim = self._get_optim(self.nn)
        self.loss = self._get_loss()
        self.nn.to(self.config.device)
        # Freeze all params after init
        for p in self.nn.parameters():
            p.requires_grad = False

    def fit(self, n, d):
        n, d = self._resample(n, d)
        ds_tr, ds_val = self._cast_dataset(n, d)
        # Unfreeze all params before training
        for p in self.nn.parameters():
            p.requires_grad = True
        self._train(ds_tr, ds_val)
        # Freeze all params after training
        for p in self.nn.parameters():
            p.requires_grad = False

    @abstractmethod
    def logits(self, x):
        pass

    @abstractmethod
    def forward(self, x):
        pass

    def _resample(self, n, d):
        """Balance sample sizes between numerator and denominator distributions.

        Args:
            n: Numerator samples tensor
            d: Denominator samples tensor

        Returns:
            Tuple of (resampled_n, resampled_d) with equal sizes
        """
        n_samples_n = n.shape[0]
        n_samples_d = d.shape[0]

        if self.config.sampling_strat == 'subsample':
            # Subsample the larger set uniformly without replacement
            if n_samples_n > n_samples_d:
                indices = torch.randperm(n_samples_n)[:n_samples_d]
                n = n[indices]
            elif n_samples_d > n_samples_n:
                indices = torch.randperm(n_samples_d)[:n_samples_n]
                d = d[indices]
            return n, d

        elif self.config.sampling_strat == 'supersample':
            # Supersample the smaller set uniformly with replacement
            if n_samples_n < n_samples_d:
                indices = torch.randint(0, n_samples_n, (n_samples_d,))
                n = n[indices]
            elif n_samples_d < n_samples_n:
                indices = torch.randint(0, n_samples_d, (n_samples_n,))
                d = d[indices]
            return n, d

        else:
            raise ValueError(
                f"Invalid sampling_strat: {self.config.sampling_strat}. "
                f"Expected 'subsample' or 'supersample'."
            )

    def _get_optim(self, model: nn.Module):
        """Setup and return optimizer based on config.

        Args:
            model: The neural network model to optimize

        Returns:
            Configured optimizer
        """
        if self.config.optimizer == 'adam':
            return torch.optim.Adam(
                model.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == 'adamw':
            return torch.optim.AdamW(
                model.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == 'sgd':
            return torch.optim.SGD(
                model.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(
                f"Invalid optimizer: {self.config.optimizer}. "
                f"Expected 'adam', 'adamw', or 'sgd'."
            )

    @abstractmethod
    def _get_nn_config(self):
        pass

    @abstractmethod
    def _cast_dataset(self, n, d):
        """Create training/validation datasets from samples."""
        pass

    @abstractmethod
    def _train(self, ds_tr, ds_val):
        """Execute training loop."""
        pass

    @abstractmethod
    def _get_loss(self):
        """Return loss function."""
        pass
