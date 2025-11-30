import torch.nn as nn
from typing import Type
from dataclasses import dataclass

from .blocks import mlp_block, cnn_block
from .base import resolve_activation, resolve_norm


# ============================================================================
# MLP Network Configuration
# ============================================================================

@dataclass
class MLPConfig:
    """Configuration for multi-layer perceptron networks.

    Creates a network with num_blocks where:
    - First block: in_dim → hidden_dim
    - Middle blocks: hidden_dim → hidden_dim
    - Last block: hidden_dim → out_dim

    Args:
        in_dim: Input dimension
        out_dim: Output dimension
        hidden_dim: Hidden dimension (fixed across all blocks)
        num_blocks: Total number of blocks (includes input and output transitions)
        depth: Number of hidden layers within each block
        activation: Activation function name (e.g., "relu", "gelu", "silu")
        output_activation: Output activation name (e.g., "identity", "sigmoid")
        norm: Normalization type (e.g., "layer", "batch1d", "none")
        dropout: Dropout probability
        residual_blocks: Add residual connections where dimensions match

    Example (simple 3-block network):
        >>> config = MLPConfig(
        ...     in_dim=10, out_dim=5, hidden_dim=64, num_blocks=3, depth=2
        ... )
        >>> model = build_mlp(config)
        # Creates: [10→64, 64→64, 64→5] with depth=2 hidden layers per block

    Example (deeper network with residuals):
        >>> config = MLPConfig(
        ...     in_dim=784, out_dim=10, hidden_dim=256, num_blocks=5,
        ...     depth=2, activation="relu", norm="layer",
        ...     residual_blocks=True
        ... )
        >>> model = build_mlp(config)
        # Creates: [784→256, 256→256 (res), 256→256 (res), 256→256 (res), 256→10]
    """
    in_dim: int
    out_dim: int
    hidden_dim: int
    num_blocks: int
    depth: int = 2
    activation: str = "relu"
    output_activation: str = "identity"
    norm: str = "none"
    dropout: float = 0.0
    residual_blocks: bool = False

    def __post_init__(self):
        """Validate configuration."""
        if self.num_blocks < 1:
            raise ValueError(f"num_blocks must be >= 1, got {self.num_blocks}")


def build_mlp(config: MLPConfig) -> nn.Sequential:
    """Build multi-layer perceptron network from configuration.

    Creates a network with num_blocks where first/last blocks handle
    dimension transitions and middle blocks maintain fixed hidden_dim.

    Args:
        config: MLPConfig specifying the network architecture

    Returns:
        nn.Sequential containing all MLP blocks

    Example:
        >>> config = MLPConfig(
        ...     in_dim=10, out_dim=5, hidden_dim=64, num_blocks=4,
        ...     activation="relu", residual_blocks=True
        ... )
        >>> model = build_mlp(config)
        >>> x = torch.randn(32, 10)
        >>> y = model(x)  # (32, 5)
    """
    # Resolve string parameters to classes
    activation = resolve_activation(config.activation)
    output_activation = resolve_activation(config.output_activation)
    norm = resolve_norm(config.norm)

    blocks = []

    for i in range(config.num_blocks):
        # Determine dimensions for this block
        if i == 0:
            # First block: input transition
            in_d, out_d = config.in_dim, config.hidden_dim
        elif i == config.num_blocks - 1:
            # Last block: output transition
            in_d, out_d = config.hidden_dim, config.out_dim
        else:
            # Middle blocks: maintain hidden_dim
            in_d, out_d = config.hidden_dim, config.hidden_dim

        # Determine if this is the last block
        is_last = (i == config.num_blocks - 1)

        # Determine if residual should be used
        use_residual = (
            config.residual_blocks
            and in_d == out_d
            and not is_last  # Don't force residual on output block
        )

        # Build block
        block = mlp_block(
            in_dim=in_d,
            out_dim=out_d,
            hidden_dim=config.hidden_dim,
            depth=config.depth,
            activation=activation,
            output_activation=output_activation if is_last else nn.Identity,
            norm=norm,
            dropout=config.dropout,
            residual=use_residual,
        )

        blocks.append(block)

    return nn.Sequential(*blocks)


# ============================================================================
# CNN Network Configuration
# ============================================================================

@dataclass
class CNNConfig:
    """Configuration for convolutional neural networks.

    Creates a homogeneous CNN with optional fully-connected layers.
    Architecture: [CNN blocks] → [optional flatten + FC blocks]

    CNN blocks use fixed hidden_channels throughout.
    Optional FC layers for classification/regression tasks.

    Args:
        in_channels: Input channels (e.g., 3 for RGB images)
        out_channels: Output channels (e.g., 10 for classification)
        hidden_channels: Fixed hidden channels for all CNN blocks
        num_cnn_blocks: Number of convolutional blocks
        kernel_size: Convolutional kernel size
        stride: Stride for convolutions
        padding: Padding mode ("same", "valid") or int
        activation: Activation function name
        output_activation: Output activation name
        norm: Normalization type (e.g., "batch2d", "instance2d", "none")
        dropout: Dropout probability
        residual_blocks: Add residual connections where dimensions match
        use_fc: Whether to add fully-connected layers after CNN
        fc_hidden_dim: Hidden dimension for FC layers (if use_fc=True)
        num_fc_blocks: Number of FC blocks (if use_fc=True)

    Example (simple CNN):
        >>> config = CNNConfig(
        ...     in_channels=3, out_channels=64, hidden_channels=32,
        ...     num_cnn_blocks=3, kernel_size=3
        ... )
        >>> model = build_cnn(config)

    Example (CNN classifier with FC layers):
        >>> config = CNNConfig(
        ...     in_channels=3, out_channels=10, hidden_channels=64,
        ...     num_cnn_blocks=4, kernel_size=3, norm="batch2d",
        ...     use_fc=True, fc_hidden_dim=128, num_fc_blocks=2
        ... )
        >>> model = build_cnn(config)
        # Creates: CNN blocks → Flatten → FC blocks → output
    """
    in_channels: int
    out_channels: int
    hidden_channels: int
    num_cnn_blocks: int
    kernel_size: int = 3
    stride: int = 1
    padding: str | int = "same"
    activation: str = "relu"
    output_activation: str = "identity"
    norm: str = "none"
    dropout: float = 0.0
    residual_blocks: bool = False
    # Fully-connected layers (optional)
    use_fc: bool = False
    fc_hidden_dim: int | None = None
    num_fc_blocks: int | None = None

    def __post_init__(self):
        """Validate configuration."""
        if self.num_cnn_blocks < 1:
            raise ValueError(f"num_cnn_blocks must be >= 1, got {self.num_cnn_blocks}")
        if self.use_fc:
            if self.fc_hidden_dim is None:
                raise ValueError("fc_hidden_dim must be specified when use_fc=True")
            if self.num_fc_blocks is None or self.num_fc_blocks < 1:
                raise ValueError("num_fc_blocks must be >= 1 when use_fc=True")


def build_cnn(config: CNNConfig) -> nn.Sequential:
    """Build convolutional neural network from configuration.

    Creates homogeneous CNN with fixed hidden channels, optionally
    followed by flatten and fully-connected layers.

    Args:
        config: CNNConfig specifying the network architecture

    Returns:
        nn.Sequential containing CNN blocks and optional FC layers

    Example (feature extractor):
        >>> config = CNNConfig(
        ...     in_channels=3, out_channels=64, hidden_channels=32,
        ...     num_cnn_blocks=3, norm="batch2d"
        ... )
        >>> model = build_cnn(config)
        >>> x = torch.randn(8, 3, 32, 32)
        >>> features = model(x)  # (8, 64, 32, 32)

    Example (classifier):
        >>> config = CNNConfig(
        ...     in_channels=3, out_channels=10, hidden_channels=64,
        ...     num_cnn_blocks=4, use_fc=True,
        ...     fc_hidden_dim=128, num_fc_blocks=2
        ... )
        >>> model = build_cnn(config)
        >>> x = torch.randn(8, 3, 32, 32)
        >>> logits = model(x)  # (8, 10) - flattened and FC applied
    """
    # Resolve string parameters to classes
    activation = resolve_activation(config.activation)
    output_activation = resolve_activation(config.output_activation)
    norm = resolve_norm(config.norm)

    layers = []

    # ========================================================================
    # Build CNN blocks
    # ========================================================================

    for i in range(config.num_cnn_blocks):
        # Determine dimensions for this block
        if i == 0:
            # First block: input transition
            in_c, out_c = config.in_channels, config.hidden_channels
        elif i == config.num_cnn_blocks - 1 and not config.use_fc:
            # Last block (no FC): output transition
            in_c, out_c = config.hidden_channels, config.out_channels
        else:
            # Middle blocks (or all blocks if FC is used): maintain hidden_channels
            in_c, out_c = config.hidden_channels, config.hidden_channels

        # Determine if residual should be used
        use_residual = config.residual_blocks and in_c == out_c

        # Build CNN block
        block = cnn_block(
            in_channels=in_c,
            out_channels=out_c,
            hidden_channels=config.hidden_channels,
            depth=2,  # Fixed depth for CNN blocks
            kernel_size=config.kernel_size,
            stride=config.stride,
            padding=config.padding,
            activation=activation,
            output_activation=nn.Identity,  # No output activation within CNN blocks
            norm=norm,
            dropout=config.dropout,
            residual=use_residual,
        )

        layers.append(block)

    # ========================================================================
    # Add FC layers if requested
    # ========================================================================

    if config.use_fc:
        # Add flatten layer
        layers.append(nn.Flatten())

        # FC blocks: Need to know flattened dimension
        # Note: This requires knowing spatial dimensions, which we don't have here
        # For now, use nn.LazyLinear for first FC layer
        # User needs to pass dummy input through model once to initialize

        for i in range(config.num_fc_blocks):
            if i == 0:
                # First FC block: use LazyLinear to infer from flattened CNN output
                layers.append(nn.LazyLinear(config.fc_hidden_dim))
            elif i == config.num_fc_blocks - 1:
                # Last FC block: output transition
                layers.append(nn.Linear(config.fc_hidden_dim, config.out_channels))
            else:
                # Middle FC blocks: maintain fc_hidden_dim
                layers.append(nn.Linear(config.fc_hidden_dim, config.fc_hidden_dim))

            # Add activation (except after last layer)
            if i < config.num_fc_blocks - 1:
                layers.append(activation())
                if config.dropout > 0.0:
                    layers.append(nn.Dropout(config.dropout))
            else:
                # Last layer: use output activation
                layers.append(output_activation())

    return nn.Sequential(*layers)
