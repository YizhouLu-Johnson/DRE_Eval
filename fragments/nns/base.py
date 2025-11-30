"""Base utilities and template for building neural network blocks.

This module contains:
- BlockConfig base class for block configurations
- build_block_template() function for constructing blocks
- Shared utility classes (Residual, MuxMLPBlock)
- Shared helper functions used across block types

Users should use the functional builders in blocks/ directory instead.
"""

from __future__ import annotations

from typing import Type, Callable, Union
from dataclasses import dataclass
import torch
import torch.nn as nn


# ============================================================================
# Base Configuration and Template
# ============================================================================

@dataclass
class BlockConfig:
    """Base configuration for all neural network blocks.

    All block-specific configs should inherit from this base class.
    Contains common parameters shared across all block types.

    Args:
        depth: Number of hidden layers in the block
        residual: Whether to wrap the block with a residual connection
        activation: Activation class for hidden layers
        output_activation: Activation class for output layer
        norm: Normalization layer class (e.g., nn.LayerNorm, nn.BatchNorm2d)
        dropout: Dropout probability
        pre_norm: If True, apply norm before layer (modern); if False, after (original)
    """
    depth: int
    residual: bool
    activation: Type[nn.Module]
    output_activation: Type[nn.Module]
    norm: Type[nn.Module] | None
    dropout: float
    pre_norm: bool


def build_block_template(
    build_input: Callable[[BlockConfig], nn.Module],
    build_hidden: Callable[[BlockConfig], nn.Module],
    build_output: Callable[[BlockConfig], nn.Module],
    config: BlockConfig,
    norm_features_input: int,
    norm_features_hidden: int,
    norm_features_output: int,
    dropout_type: str = "standard",
    skip_output_extras: bool = False,
) -> nn.Sequential | Residual:
    """Template function for building neural network blocks.

    This template captures the common pattern across block types:
    1. Build input projection layer
    2. Assemble with norm/activation/dropout (handles pre/post norm)
    3. Build depth-1 hidden layers with assembly
    4. Build output layer with output_activation only
    5. Wrap with Residual if config.residual is True

    Block-specific implementations provide functions that build their
    core layer only (Linear, Conv2d, MultiHeadAttention, etc.).
    The template handles adding norm/activation/dropout.

    Args:
        build_input: Function that builds core input layer (returns single nn.Module)
        build_hidden: Function that builds core hidden layer (returns single nn.Module)
        build_output: Function that builds core output layer (returns single nn.Module)
        config: Block configuration (BlockConfig or subclass)
        norm_features_input: Number of features for input layer normalization
        norm_features_hidden: Number of features for hidden layer normalization
        norm_features_output: Number of features for output layer normalization
        dropout_type: Type of dropout ("standard", "dropout2d", "dropout3d")
        skip_output_extras: If True, output layer gets no norm/activation/dropout assembly

    Returns:
        nn.Sequential module, or Residual-wrapped Sequential if config.residual=True

    Example:
        >>> def _build_mlp_input(cfg):
        ...     return nn.Linear(cfg.in_dim, cfg.hidden_dim)
        >>>
        >>> def _build_mlp_hidden(cfg):
        ...     return nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        >>>
        >>> def _build_mlp_output(cfg):
        ...     return nn.Linear(cfg.hidden_dim, cfg.out_dim)
        >>>
        >>> config = MLPBlockConfig(in_dim=10, out_dim=5, hidden_dim=64, depth=3)
        >>> block = build_block_template(_build_mlp_input, _build_mlp_hidden, _build_mlp_output,
        ...                              config, 64, 64, 5)
    """
    layers = []

    # Input projection with assembly
    input_layer = build_input(config)
    if config.pre_norm and config.norm is not None:
        layers.append(config.norm(norm_features_input))
        layers.append(input_layer)
    else:
        layers.append(input_layer)
        if config.norm is not None:
            layers.append(config.norm(norm_features_input))

    layers.append(config.activation())
    if config.dropout > 0:
        layers.append(build_dropout(config.dropout, dropout_type))

    # Hidden layers with assembly
    for _ in range(config.depth - 1):
        hidden_layer = build_hidden(config)
        if config.pre_norm and config.norm is not None:
            layers.append(config.norm(norm_features_hidden))
            layers.append(hidden_layer)
        else:
            layers.append(hidden_layer)
            if config.norm is not None:
                layers.append(config.norm(norm_features_hidden))

        layers.append(config.activation())
        if config.dropout > 0:
            layers.append(build_dropout(config.dropout, dropout_type))

    # Output layer
    output_layer = build_output(config)
    if not skip_output_extras:
        if config.pre_norm and config.norm is not None:
            layers.append(config.norm(norm_features_output))
            layers.append(output_layer)
        else:
            layers.append(output_layer)
            if config.norm is not None:
                layers.append(config.norm(norm_features_output))
        layers.append(config.output_activation())
    else:
        # Just output layer + output activation, no norm/dropout
        layers.append(output_layer)
        layers.append(config.output_activation())

    # Create sequential
    seq = nn.Sequential(*layers)

    # Wrap with residual if requested
    if config.residual:
        return Residual(seq)
    return seq


# ============================================================================
# Shared Utility Classes
# ============================================================================

class Residual(nn.Module):
    """Adds residual connection to any module: output = input + module(input).

    This wrapper enables residual connections within the Sequential regime,
    making it possible to use functional builders with skip connections.

    Args:
        module: The module to wrap with residual connection

    Example:
        >>> # Wrap a sequential block with residual
        >>> block = Residual(nn.Sequential(
        ...     nn.Linear(64, 64),
        ...     nn.ReLU(),
        ...     nn.Linear(64, 64)
        ... ))
        >>> x = torch.randn(32, 64)
        >>> y = block(x)  # y = x + block(x)

        >>> # Can be used in larger Sequential
        >>> net = nn.Sequential(
        ...     nn.Linear(10, 64),
        ...     Residual(nn.Sequential(nn.Linear(64, 64), nn.ReLU())),
        ...     Residual(nn.Sequential(nn.Linear(64, 64), nn.ReLU())),
        ...     nn.Linear(64, 5)
        ... )
    """
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        return x + self.module(x)


class Mux(nn.Module):
    """Multiplexer that routes inputs through multiple sub-blocks.

    A general-purpose wrapper for complex routing patterns. Works with arbitrary
    blocks (MLPs, CNNs, attention, or any nn.Module). The router function defines
    how inputs flow through the blocks.

    This pattern is inspired by the pseudoposterior router in cvhn.py:136-145,
    where a router function takes an nn.ModuleDict and returns a dict of tensors.

    The router is a callable that:
    1. Takes the ModuleDict of blocks and input tensors
    2. Routes inputs through appropriate blocks
    3. Returns a dictionary of computed tensors

    This enables complex routing patterns like:
    - Encode different inputs separately
    - Combine encodings with intermediate computations
    - Multi-path architectures with shared or separate processing

    Args:
        blocks: ModuleDict containing the sub-blocks (can be any nn.Module types)
        router: Callable that takes (blocks, *inputs) and returns dict[str, Tensor]

    Example (encoding multiple inputs):
        >>> blocks = nn.ModuleDict({
        ...     'd_enc': nn.Linear(d_dim, h_dim),
        ...     'y_enc': nn.Linear(y_dim, h_dim),
        ...     'x_enc': nn.Linear(x_dim, h_dim),
        ...     'joint_dec': nn.Linear(3*h_dim, out_dim)
        ... })
        >>>
        >>> def router(blocks, d, y, x):
        ...     return {
        ...         'd': (d_emb := blocks['d_enc'](d)),
        ...         'y': (y_emb := blocks['y_enc'](y)),
        ...         'x': (x_emb := blocks['x_enc'](x)),
        ...         'joint': blocks['joint_dec'](torch.cat([d_emb, y_emb, x_emb], dim=-1))
        ...     }
        >>>
        >>> mux = Mux(blocks, router)
        >>> outputs = mux(d, y, x)  # dict with keys: 'd', 'y', 'x', 'joint'
    """
    def __init__(
        self,
        blocks: nn.ModuleDict,
        router: Callable[[nn.ModuleDict, ...], dict[str, torch.Tensor]],
    ):
        super().__init__()
        self.blocks = blocks
        self.router = router

    def forward(self, *inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Route inputs through blocks using the router function.

        Args:
            *inputs: Variable number of input tensors to route

        Returns:
            Dictionary of output tensors computed by the router
        """
        return self.router(self.blocks, *inputs)


# ============================================================================
# Shared Helper Functions
# ============================================================================

def build_normalization(
    norm_type: Type[nn.Module] | None,
    num_features: int,
    num_groups: int | None = None,
    eps: float = 1e-5,
) -> nn.Module | None:
    """Construct normalization layer with appropriate configuration.

    Args:
        norm_type: Normalization layer class (LayerNorm, BatchNorm1d, BatchNorm2d,
                   GroupNorm, InstanceNorm2d, etc.) or None
        num_features: Number of features/channels to normalize
        num_groups: Number of groups for GroupNorm (required if norm_type is GroupNorm)
        eps: Small constant for numerical stability

    Returns:
        Initialized normalization module or None if norm_type is None

    Raises:
        ValueError: If GroupNorm is requested without num_groups
    """
    if norm_type is None:
        return None

    # GroupNorm requires num_groups parameter
    if norm_type == nn.GroupNorm:
        if num_groups is None:
            raise ValueError("GroupNorm requires num_groups parameter")
        return nn.GroupNorm(num_groups, num_features, eps=eps)

    # LayerNorm uses tuple for normalized_shape
    if norm_type == nn.LayerNorm:
        return nn.LayerNorm(num_features, eps=eps)

    # BatchNorm1d, BatchNorm2d, InstanceNorm2d, etc.
    return norm_type(num_features, eps=eps)


def build_activation(activation: Type[nn.Module]) -> nn.Module:
    """Construct activation layer.

    Args:
        activation: Activation layer class (ReLU, GELU, SiLU, Tanh, etc.)

    Returns:
        Initialized activation module
    """
    return activation()


def build_dropout(
    dropout: float,
    dropout_type: str = "standard",
) -> nn.Module | None:
    """Construct dropout layer if dropout probability is greater than 0.

    Args:
        dropout: Dropout probability (0.0 to 1.0)
        dropout_type: Type of dropout - "standard" (Dropout), "dropout2d" (Dropout2d),
                      "dropout3d" (Dropout3d)

    Returns:
        Initialized dropout module or None if dropout == 0

    Raises:
        ValueError: If dropout_type is not recognized
    """
    if dropout <= 0:
        return None

    if dropout_type == "standard":
        return nn.Dropout(dropout)
    elif dropout_type == "dropout2d":
        return nn.Dropout2d(dropout)
    elif dropout_type == "dropout3d":
        return nn.Dropout3d(dropout)
    else:
        raise ValueError(
            f"Unknown dropout_type: {dropout_type}. "
            f"Expected 'standard', 'dropout2d', or 'dropout3d'"
        )


def assemble_layer_with_extras(
    layer: nn.Module,
    norm: Type[nn.Module] | None,
    activation: Type[nn.Module],
    dropout: float,
    num_features: int,
    dropout_type: str = "standard",
    num_groups: int | None = None,
) -> list[nn.Module]:
    """Assemble layer with norm -> activation -> dropout pattern.

    This is a common pattern across different block types:
    layer -> [norm] -> activation -> [dropout]

    Args:
        layer: The main layer (Linear, Conv2d, etc.)
        norm: Normalization layer class or None
        activation: Activation layer class
        dropout: Dropout probability
        num_features: Number of features/channels for normalization
        dropout_type: Type of dropout to use
        num_groups: Number of groups for GroupNorm (if applicable)

    Returns:
        List of modules to add to Sequential
    """
    modules = [layer]

    # Add normalization if specified
    norm_layer = build_normalization(norm, num_features, num_groups)
    if norm_layer is not None:
        modules.append(norm_layer)

    # Add activation
    modules.append(build_activation(activation))

    # Add dropout if specified
    dropout_layer = build_dropout(dropout, dropout_type)
    if dropout_layer is not None:
        modules.append(dropout_layer)

    return modules


def validate_residual_compatibility(
    in_features: int,
    out_features: int,
    operation_name: str,
) -> None:
    """Raise ValueError if dimensions are incompatible with residual connection.

    Residual connections require input and output to have the same shape for
    element-wise addition: output = input + module(input)

    Args:
        in_features: Input feature dimension or number of channels
        out_features: Output feature dimension or number of channels
        operation_name: Name of the operation (for error message)

    Raises:
        ValueError: If in_features != out_features
    """
    if in_features != out_features:
        raise ValueError(
            f"Residual connection requires matching input/output dimensions. "
            f"Got in_features={in_features}, out_features={out_features} for {operation_name}. "
            f"Either set residual=False or ensure in_features == out_features."
        )


def assert_shape(
    tensor_ndim: int,
    expected_dims: int,
    tensor_name: str,
) -> None:
    """Assert tensor has expected number of dimensions.

    Args:
        tensor_ndim: Actual number of dimensions (tensor.ndim)
        expected_dims: Expected number of dimensions
        tensor_name: Name of tensor for error message

    Raises:
        ValueError: If tensor_ndim != expected_dims
    """
    if tensor_ndim != expected_dims:
        raise ValueError(
            f"{tensor_name} expected {expected_dims}D tensor, "
            f"got {tensor_ndim}D tensor"
        )


# ============================================================================
# Factory Utilities (String-to-Class Registries)
# ============================================================================

ACTIVATION_REGISTRY = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
    "identity": nn.Identity,
}

NORM_REGISTRY = {
    "layer": nn.LayerNorm,
    "batch1d": nn.BatchNorm1d,
    "batch2d": nn.BatchNorm2d,
    "batch3d": nn.BatchNorm3d,
    "instance1d": nn.InstanceNorm1d,
    "instance2d": nn.InstanceNorm2d,
    "instance3d": nn.InstanceNorm3d,
    "group": nn.GroupNorm,
    "none": None,
}


def resolve_activation(name: str) -> Type[nn.Module]:
    """Resolve activation string name to PyTorch class.

    Args:
        name: Activation name (e.g., "relu", "gelu", "silu")

    Returns:
        Activation class from torch.nn

    Raises:
        ValueError: If activation name is not recognized
    """
    act = ACTIVATION_REGISTRY.get(name.lower())
    if act is None:
        raise ValueError(
            f"Unknown activation: {name}. "
            f"Available: {list(ACTIVATION_REGISTRY.keys())}"
        )
    return act


def resolve_norm(name: str) -> Type[nn.Module] | None:
    """Resolve normalization string name to PyTorch class.

    Args:
        name: Normalization name (e.g., "layer", "batch2d", "none")

    Returns:
        Normalization class from torch.nn, or None if name is "none"

    Raises:
        ValueError: If normalization name is not recognized
    """
    if name.lower() not in NORM_REGISTRY:
        raise ValueError(
            f"Unknown norm: {name}. "
            f"Available: {list(NORM_REGISTRY.keys())}"
        )
    return NORM_REGISTRY[name.lower()]
