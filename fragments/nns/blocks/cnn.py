"""CNN (Convolutional Neural Network) block implementations.

This module provides functional builders for CNN blocks using the shared
block template from base.py.
"""

import torch
import torch.nn as nn
from typing import Type
from dataclasses import dataclass

from ..base import (
    BlockConfig,
    build_block_template,
    validate_residual_compatibility,
    assemble_layer_with_extras,
)


# ============================================================================
# CNN-Specific Utilities
# ============================================================================

def compute_conv_output_dim(
    input_dim: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int = 1,
) -> int:
    """Compute output dimension for convolution operation.

    Formula: output = floor((input + 2*padding - dilation*(kernel_size - 1) - 1) / stride) + 1

    Args:
        input_dim: Input spatial dimension (height or width)
        kernel_size: Size of convolution kernel
        stride: Stride of convolution
        padding: Amount of padding added
        dilation: Spacing between kernel elements

    Returns:
        Output spatial dimension after convolution
    """
    return (input_dim + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


def compute_conv_same_padding(kernel_size: int, stride: int = 1, dilation: int = 1) -> int:
    """Compute padding needed for 'same' convolution (preserves spatial dimensions when stride=1).

    Args:
        kernel_size: Size of convolution kernel
        stride: Stride of convolution
        dilation: Spacing between kernel elements

    Returns:
        Padding value needed for 'same' convolution
    """
    return ((stride - 1) + dilation * (kernel_size - 1)) // 2


# ============================================================================
# CNN Configuration
# ============================================================================

@dataclass
class CNNBlockConfig(BlockConfig):
    """Configuration for CNN blocks.

    CNN-specific fields only. Common fields (activation, norm, dropout, etc.)
    are inherited from BlockConfig.

    Args:
        in_channels: Input channels
        out_channels: Output channels
        hidden_channels: Hidden layer channels
        depth: Number of hidden conv layers (from BlockConfig)
        kernel_size: Convolution kernel size (int or tuple)
        stride: Stride for convolutions
        padding: Padding mode or int ('same', 'valid', or int)
        pool: Pooling layer class (e.g., nn.MaxPool2d, nn.AvgPool2d)
        pool_kernel_size: Kernel size for pooling
        pool_stride: Stride for pooling
        pool_every: Apply pooling every N layers
        activation: Activation class for hidden layers (from BlockConfig)
        output_activation: Activation class for output layer (from BlockConfig)
        norm: Normalization layer class (from BlockConfig)
        dropout: Dropout probability (from BlockConfig)
        residual: Whether to wrap with residual connection (from BlockConfig)
        pre_norm: Pre-normalization vs post-normalization (from BlockConfig)
    """
    in_channels: int
    out_channels: int
    hidden_channels: int
    kernel_size: int | tuple[int, int] = 3
    stride: int = 1
    padding: int | str = "same"
    pool: Type[nn.Module] | None = None
    pool_kernel_size: int = 2
    pool_stride: int | None = None
    pool_every: int | None = None

    # Internal state (computed during initialization)
    _padding_int: int | None = None

    def __post_init__(self):
        """Process padding parameter."""
        if isinstance(self.padding, str):
            if self.padding == "same":
                if isinstance(self.kernel_size, tuple):
                    self._padding_int = compute_conv_same_padding(self.kernel_size[0], self.stride)
                else:
                    self._padding_int = compute_conv_same_padding(self.kernel_size, self.stride)
            elif self.padding == "valid":
                self._padding_int = 0
            else:
                raise ValueError(f"Unknown padding mode: {self.padding}. Expected 'same', 'valid', or int")
        else:
            self._padding_int = self.padding

        # Default pool stride to pool kernel size
        if self.pool_stride is None:
            object.__setattr__(self, 'pool_stride', self.pool_kernel_size)


# ============================================================================
# Private Builder Functions
# ============================================================================

def _build_cnn_input(config: CNNBlockConfig) -> nn.Module:
    """Build core input layer for CNN.

    Template handles norm/activation/dropout assembly.
    Note: Pooling is handled separately (TODO: needs template support).

    Args:
        config: CNN block configuration

    Returns:
        Single Conv2d layer
    """
    return nn.Conv2d(
        config.in_channels,
        config.hidden_channels,
        config.kernel_size,
        stride=config.stride,
        padding=config._padding_int
    )


def _build_cnn_hidden(config: CNNBlockConfig) -> nn.Module:
    """Build core hidden layer for CNN.

    Template handles norm/activation/dropout assembly.
    Note: Pooling is handled separately (TODO: needs template support).

    Args:
        config: CNN block configuration

    Returns:
        Single Conv2d layer
    """
    return nn.Conv2d(
        config.hidden_channels,
        config.hidden_channels,
        config.kernel_size,
        stride=1,
        padding=config._padding_int
    )


def _build_cnn_output(config: CNNBlockConfig) -> nn.Module:
    """Build core output layer for CNN.

    Template handles output_activation.

    Args:
        config: CNN block configuration

    Returns:
        Single Conv2d layer
    """
    return nn.Conv2d(
        config.hidden_channels,
        config.out_channels,
        config.kernel_size,
        stride=1,
        padding=config._padding_int
    )


# ============================================================================
# Public API
# ============================================================================

def cnn_block(
    in_channels: int,
    out_channels: int,
    hidden_channels: int,
    depth: int,
    kernel_size: int | tuple[int, int] = 3,
    stride: int = 1,
    padding: int | str = "same",
    activation: Type[nn.Module] = nn.ReLU,
    output_activation: Type[nn.Module] = nn.Identity,
    norm: Type[nn.Module] | None = None,
    dropout: float = 0.0,
    pool: Type[nn.Module] | None = None,
    pool_kernel_size: int = 2,
    pool_stride: int | None = None,
    pool_every: int | None = None,
    residual: bool = False,
) -> nn.Sequential | nn.Module:
    """Build a convolutional neural network block.

    Architecture: in -> hidden -> ... -> hidden -> out
    Each hidden layer: Conv2d -> [Norm] -> Activation -> [Pool] -> [Dropout]
    Output layer: Conv2d -> OutputActivation

    Args:
        in_channels: Input channels
        out_channels: Output channels
        hidden_channels: Hidden layer channels
        depth: Number of hidden conv layers (depth=1 means in->hidden->out)
        kernel_size: Convolution kernel size (int or (height, width))
        stride: Stride for convolutions
        padding: Padding mode - 'same' (preserve spatial dims when stride=1),
                 'valid' (no padding), or int for explicit padding
        activation: Activation class for hidden layers
        output_activation: Activation class for output layer
        norm: Normalization layer class (e.g., nn.BatchNorm2d, nn.GroupNorm)
        dropout: Dropout probability (applied as Dropout2d for spatial dropout)
        pool: Pooling layer class (e.g., nn.MaxPool2d, nn.AvgPool2d)
        pool_kernel_size: Kernel size for pooling operations
        pool_stride: Stride for pooling (defaults to pool_kernel_size)
        pool_every: Apply pooling every N layers (1=every layer, 2=every other layer, None=no pooling)
        residual: If True, wraps with Residual (requires in_channels == out_channels and no pooling)

    Returns:
        nn.Sequential module, or Residual-wrapped Sequential if residual=True

    Example:
        >>> # Standard conv block
        >>> block = cnn_block(3, 64, 32, depth=3, kernel_size=3)
        >>> x = torch.randn(8, 3, 32, 32)
        >>> y = block(x)  # (8, 64, 32, 32) with padding='same'

        >>> # ResNet-style block with residual
        >>> res_block = cnn_block(64, 64, 64, depth=2, norm=nn.BatchNorm2d, residual=True)
        >>> x = torch.randn(8, 64, 32, 32)
        >>> y = res_block(x)  # (8, 64, 32, 32) with skip connection

        >>> # Encoder with pooling
        >>> enc = cnn_block(3, 128, 64, depth=3, pool=nn.MaxPool2d, pool_every=1)
        >>> x = torch.randn(8, 3, 64, 64)
        >>> y = enc(x)  # (8, 128, 8, 8) - pooled 3 times
    """
    # Validate residual compatibility
    if residual:
        validate_residual_compatibility(in_channels, out_channels, "cnn_block")
        if pool is not None and pool_every is not None:
            raise ValueError(
                "Cannot use residual connection with pooling (pooling changes spatial dimensions). "
                "Set residual=False or pool_every=None."
            )

    # Create config
    config = CNNBlockConfig(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        depth=depth,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        activation=activation,
        output_activation=output_activation,
        norm=norm,
        dropout=dropout,
        pool=pool,
        pool_kernel_size=pool_kernel_size,
        pool_stride=pool_stride,
        pool_every=pool_every,
        residual=residual,
    )

    # Build using template
    # Note: Pooling not yet supported in template - will be added later
    if pool is not None and pool_every is not None:
        raise NotImplementedError(
            "Pooling is not yet supported with the refactored template. "
            "This will be added in a future update. For now, set pool=None or pool_every=None."
        )

    # Determine normalization dimensions based on pre_norm mode
    # Pre-norm: normalize INPUT to each layer
    # Post-norm: normalize OUTPUT of each layer
    norm_input = in_channels if config.pre_norm else hidden_channels
    norm_output = hidden_channels if config.pre_norm else out_channels

    return build_block_template(
        build_input=_build_cnn_input,
        build_hidden=_build_cnn_hidden,
        build_output=_build_cnn_output,
        config=config,
        norm_features_input=norm_input,
        norm_features_hidden=hidden_channels,
        norm_features_output=norm_output,
        dropout_type="dropout2d",
        skip_output_extras=True,  # CNN output just gets output_activation
    )


# ============================================================================
# Specialized Classes
# ============================================================================

class ResidualConvBlock(nn.Module):
    """CNN block with residual connection and post-normalization.

    Similar to ResidualMLPBlock but for convolutional layers.
    Architecture: x -> norm(x + conv_block(x)) -> activation

    Args:
        channels: Input/output channels (must match for residual)
        hidden_channels: Hidden layer channels (defaults to channels)
        kernel_size: Convolution kernel size
        activation: Activation class
        norm: Normalization layer class (e.g., nn.BatchNorm2d)
        dropout: Dropout probability (applied as Dropout2d)

    Example:
        >>> block = ResidualConvBlock(channels=64, hidden_channels=128, kernel_size=3)
        >>> x = torch.randn(8, 64, 32, 32)
        >>> y = block(x)  # (8, 64, 32, 32)
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int | None = None,
        kernel_size: int = 3,
        activation: Type[nn.Module] = nn.ReLU,
        norm: Type[nn.Module] = nn.BatchNorm2d,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_channels = hidden_channels or channels

        # Compute padding for 'same' convolution
        padding = compute_conv_same_padding(kernel_size)

        layers = [
            nn.Conv2d(channels, hidden_channels, kernel_size, padding=padding),
            activation(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers.append(nn.Conv2d(hidden_channels, channels, kernel_size, padding=padding))

        self.block = nn.Sequential(*layers)
        self.norm = norm(channels)
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.

        Args:
            x: Input tensor (batch, channels, height, width)

        Returns:
            Output tensor (batch, channels, height, width)
        """
        return self.activation(self.norm(x + self.block(x)))
