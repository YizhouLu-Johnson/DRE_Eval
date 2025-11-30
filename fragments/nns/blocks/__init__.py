"""Neural network block builders.

This package provides functional builders for common neural network patterns.
Each module contains block-specific implementations that use the shared template
from base.py.

Public API:
- mlp_block: Build MLP (Multi-Layer Perceptron) blocks
- cnn_block: Build CNN (Convolutional Neural Network) blocks
- attention_block: Build multi-head attention blocks
- ResidualMLPBlock: MLP block with post-normalization
- ResidualConvBlock: CNN block with post-normalization
- MultiHeadAttention: Core multi-head attention mechanism
- Residual: Residual connection wrapper (re-exported from base)
- Mux: Multiplexer for routing inputs through arbitrary blocks (re-exported from base)
"""

from .mlp import mlp_block, ResidualMLPBlock
from .cnn import cnn_block, ResidualConvBlock
from .attention import attention_block, MultiHeadAttention
from ..base import Residual, Mux

__all__ = [
    "mlp_block",
    "cnn_block",
    "attention_block",
    "ResidualMLPBlock",
    "ResidualConvBlock",
    "MultiHeadAttention",
    "Residual",
    "Mux",
]
