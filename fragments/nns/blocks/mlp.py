"""MLP (Multi-Layer Perceptron) block implementations.

This module provides functional builders for MLP blocks using the shared
block template from base.py.
"""

import torch
import torch.nn as nn
from typing import Type
from dataclasses import dataclass

from ..base import BlockConfig, build_block_template, validate_residual_compatibility


# ============================================================================
# MLP Configuration
# ============================================================================

@dataclass
class MLPBlockConfig(BlockConfig):
    """Configuration for MLP blocks.

    MLP-specific fields only. Common fields (activation, norm, dropout, etc.)
    are inherited from BlockConfig.

    Args:
        in_dim: Input dimension
        out_dim: Output dimension
        hidden_dim: Hidden layer dimension
        depth: Number of hidden layers (from BlockConfig)
        activation: Activation class for hidden layers (from BlockConfig)
        output_activation: Activation class for output layer (from BlockConfig)
        norm: Normalization layer class (from BlockConfig)
        dropout: Dropout probability (from BlockConfig)
        residual: Whether to wrap with residual connection (from BlockConfig)
        pre_norm: Pre-normalization vs post-normalization (from BlockConfig)
    """
    in_dim: int
    out_dim: int
    hidden_dim: int


# ============================================================================
# Private Builder Functions
# ============================================================================

def _build_mlp_input(config: MLPBlockConfig) -> nn.Module:
    """Build core input layer for MLP.

    Template handles norm/activation/dropout assembly.

    Args:
        config: MLP block configuration

    Returns:
        Single Linear layer
    """
    return nn.Linear(config.in_dim, config.hidden_dim)


def _build_mlp_hidden(config: MLPBlockConfig) -> nn.Module:
    """Build core hidden layer for MLP.

    Template handles norm/activation/dropout assembly.

    Args:
        config: MLP block configuration

    Returns:
        Single Linear layer
    """
    return nn.Linear(config.hidden_dim, config.hidden_dim)


def _build_mlp_output(config: MLPBlockConfig) -> nn.Module:
    """Build core output layer for MLP.

    Template handles output_activation.

    Args:
        config: MLP block configuration

    Returns:
        Single Linear layer
    """
    return nn.Linear(config.hidden_dim, config.out_dim)


# ============================================================================
# Public API
# ============================================================================

def mlp_block(
    in_dim: int,
    out_dim: int,
    hidden_dim: int,
    depth: int,
    activation: Type[nn.Module] = nn.ReLU,
    output_activation: Type[nn.Module] = nn.Identity,
    norm: Type[nn.Module] | None = None,
    dropout: float = 0.0,
    residual: bool = False,
) -> nn.Sequential | nn.Module:
    """Build a simple feedforward MLP.

    Architecture: in -> hidden -> ... -> hidden -> out
    Each hidden layer: Linear -> [Norm] -> Activation -> [Dropout]
    Output layer: Linear -> OutputActivation

    Args:
        in_dim: Input dimension
        out_dim: Output dimension
        hidden_dim: Hidden layer dimension
        depth: Number of hidden layers (depth=1 means in->hidden->out)
        activation: Activation class for hidden layers
        output_activation: Activation class for output layer
        norm: Optional normalization layer class (e.g., nn.LayerNorm, nn.BatchNorm1d)
        dropout: Dropout probability (0 = no dropout)
        residual: If True, wraps the block with Residual (requires in_dim == out_dim)

    Returns:
        nn.Sequential module, or Residual-wrapped Sequential if residual=True

    Example:
        >>> # Standard MLP
        >>> encoder = mlp_block(10, 5, 64, depth=3, activation=nn.ReLU)
        >>> x = torch.randn(32, 10)
        >>> y = encoder(x)  # (32, 5)

        >>> # With residual connection
        >>> res_block = mlp_block(64, 64, 64, depth=2, residual=True)
        >>> x = torch.randn(32, 64)
        >>> y = res_block(x)  # (32, 64) with skip connection
    """
    # Validate residual compatibility
    if residual:
        validate_residual_compatibility(in_dim, out_dim, "mlp_block")

    # Create config
    config = MLPBlockConfig(
        in_dim=in_dim,
        out_dim=out_dim,
        hidden_dim=hidden_dim,
        depth=depth,
        activation=activation,
        output_activation=output_activation,
        norm=norm,
        dropout=dropout,
        residual=residual,
        pre_norm=True,
    )

    # Determine normalization dimensions based on pre_norm mode
    # Pre-norm: normalize INPUT to each layer
    # Post-norm: normalize OUTPUT of each layer
    norm_input = in_dim if config.pre_norm else hidden_dim
    norm_output = hidden_dim if config.pre_norm else out_dim

    # Build using template
    return build_block_template(
        build_input=_build_mlp_input,
        build_hidden=_build_mlp_hidden,
        build_output=_build_mlp_output,
        config=config,
        norm_features_input=norm_input,
        norm_features_hidden=hidden_dim,
        norm_features_output=norm_output,
        dropout_type="standard",
        skip_output_extras=True,  # MLP output just gets output_activation
    )


# ============================================================================
# Specialized Classes
# ============================================================================

class ResidualMLPBlock(nn.Module):
    """MLP block with residual connection and post-normalization.

    Can't be expressed as nn.Sequential due to custom residual+norm pattern.
    Architecture: x -> norm(x + block(x)) -> activation

    Args:
        dim: Input/output dimension (must match for residual)
        hidden_dim: Hidden layer dimension (defaults to dim)
        activation: Activation class
        norm: Normalization layer class
        dropout: Dropout probability

    Example:
        >>> block = ResidualMLPBlock(dim=64, hidden_dim=128)
        >>> x = torch.randn(32, 64)
        >>> y = block(x)  # (32, 64)
    """
    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        activation: Type[nn.Module] = nn.ReLU,
        norm: Type[nn.Module] = nn.LayerNorm,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_dim = hidden_dim or dim

        layers = [nn.Linear(dim, hidden_dim), activation()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, dim))

        self.block = nn.Sequential(*layers)
        self.norm = norm(dim)
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        return self.activation(self.norm(x + self.block(x)))
