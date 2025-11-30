"""Multi-head attention block implementations.

This module provides functional builders for attention blocks using the shared
block template from base.py.
"""

import torch
import torch.nn as nn
from typing import Type
from dataclasses import dataclass

from ..base import BlockConfig, build_block_template


# ============================================================================
# Attention Configuration
# ============================================================================

@dataclass
class AttentionBlockConfig(BlockConfig):
    """Configuration for attention blocks.

    Attention-specific fields only. Common fields (activation, norm, dropout, pre_norm, etc.)
    are inherited from BlockConfig.

    Args:
        embed_dim: Embedding dimension (input/output dimension)
        num_heads: Number of attention heads
        depth: Number of stacked attention layers (from BlockConfig)
        head_dim: Dimension per head (defaults to embed_dim // num_heads)
        qkv_bias: Include bias in Q/K/V projections
        attn_dropout: Dropout applied to attention weights
        proj_dropout: Dropout applied to output projection
        is_causal: If True, apply causal masking for autoregressive models
        activation: Not used for attention (from BlockConfig, ignored)
        output_activation: Not used for attention (from BlockConfig, ignored)
        norm: Normalization layer class (from BlockConfig)
        dropout: Not used for attention, use attn_dropout/proj_dropout instead (from BlockConfig, ignored)
        residual: Whether to wrap with residual connection (from BlockConfig)
        pre_norm: Pre-normalization (True, modern) vs post-norm (False, original) (from BlockConfig)
    """
    embed_dim: int
    num_heads: int
    head_dim: int | None = None
    qkv_bias: bool = True
    attn_dropout: float = 0.0
    proj_dropout: float = 0.0
    is_causal: bool = False


# ============================================================================
# Multi-Head Attention Class
# ============================================================================

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism.

    Implements the core attention mechanism from "Attention Is All You Need".
    Supports self-attention, cross-attention, causal masking, and attention masks.

    This is a class (not a function) because:
    1. Forward signature: forward(x, mask=None, kv=None) for cross-attention
    2. Non-sequential: masking logic requires custom forward pass
    3. State: Can be extended to cache K/V for autoregressive generation

    Args:
        embed_dim: Embedding dimension (input/output dimension)
        num_heads: Number of attention heads
        head_dim: Dimension per head (defaults to embed_dim // num_heads)
        qkv_bias: Include bias in Q/K/V projections
        attn_dropout: Dropout applied to attention weights
        proj_dropout: Dropout applied to output projection
        is_causal: If True, apply causal masking for autoregressive models

    Example:
        >>> # Self-attention
        >>> attn = MultiHeadAttention(embed_dim=512, num_heads=8)
        >>> x = torch.randn(32, 100, 512)  # (batch, seq_len, embed_dim)
        >>> y = attn(x)  # (32, 100, 512)

        >>> # Cross-attention (e.g., decoder attending to encoder)
        >>> attn = MultiHeadAttention(embed_dim=512, num_heads=8)
        >>> query = torch.randn(32, 50, 512)  # decoder states
        >>> kv = torch.randn(32, 100, 512)    # encoder states
        >>> y = attn(query, kv=kv)  # (32, 50, 512)

        >>> # Causal attention (GPT-style)
        >>> attn = MultiHeadAttention(embed_dim=512, num_heads=8, is_causal=True)
        >>> x = torch.randn(32, 100, 512)
        >>> y = attn(x)  # (32, 100, 512) with causal masking
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        head_dim: int | None = None,
        qkv_bias: bool = True,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        is_causal: bool = False,
    ):
        super().__init__()

        # Compute head dimension
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim or (embed_dim // num_heads)
        self.is_causal = is_causal

        # Ensure dimensions are compatible
        if self.head_dim * num_heads != embed_dim and head_dim is None:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads}). "
                f"Got embed_dim // num_heads = {embed_dim / num_heads}. "
                f"Alternatively, specify head_dim explicitly."
            )

        # Total dimension for multi-head (may differ from embed_dim if head_dim is specified)
        self.inner_dim = self.head_dim * num_heads

        # Q, K, V projections
        self.q_proj = nn.Linear(embed_dim, self.inner_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(embed_dim, self.inner_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(embed_dim, self.inner_dim, bias=qkv_bias)

        # Output projection
        self.out_proj = nn.Linear(self.inner_dim, embed_dim, bias=True)

        # Dropout
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)

        # Scaling factor for attention scores
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        kv: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with multi-head attention.

        Args:
            x: Query tensor (batch, seq_len, embed_dim) or (seq_len, batch, embed_dim)
            mask: Attention mask (optional). Shape: (batch, seq_len, seq_len) or (seq_len, seq_len)
                  True/1 = attend, False/0 = mask out (will be converted to additive mask)
            kv: Key/value tensor for cross-attention (batch, kv_len, embed_dim)
                If None, performs self-attention (uses x for K and V)

        Returns:
            Output tensor with same shape as x
        """
        batch_size, seq_len, embed_dim = x.shape

        # Determine if self-attention or cross-attention
        if kv is None:
            # Self-attention: Q, K, V all from x
            kv = x
            kv_len = seq_len
        else:
            # Cross-attention: Q from x, K and V from kv
            kv_len = kv.shape[1]

        # Project to Q, K, V
        q = self.q_proj(x)  # (batch, seq_len, inner_dim)
        k = self.k_proj(kv)  # (batch, kv_len, inner_dim)
        v = self.v_proj(kv)  # (batch, kv_len, inner_dim)

        # Reshape for multi-head attention
        # (batch, seq_len, inner_dim) -> (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores: Q * K^T / sqrt(head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (batch, num_heads, seq_len, kv_len)

        # Apply mask if provided
        if mask is not None:
            # Convert boolean mask to additive mask: True -> 0.0, False -> -inf
            if mask.dtype == torch.bool:
                attn_scores = attn_scores.masked_fill(~mask, float("-inf"))
            else:
                # Assume mask is additive (0 = attend, -inf = mask out)
                attn_scores = attn_scores + mask

        # Apply causal mask if needed (for autoregressive models)
        if self.is_causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, kv_len, dtype=torch.bool, device=x.device),
                diagonal=1
            )
            attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

        # Compute attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (batch, num_heads, seq_len, kv_len)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # (batch, num_heads, seq_len, head_dim)

        # Reshape back to (batch, seq_len, inner_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.inner_dim)

        # Output projection
        output = self.out_proj(attn_output)  # (batch, seq_len, embed_dim)
        output = self.proj_dropout(output)

        return output


# ============================================================================
# Private Builder Functions
# ============================================================================

def _build_attention_input(config: AttentionBlockConfig) -> nn.Module:
    """Build first attention layer (same as hidden for attention).

    Template handles norm/activation/dropout assembly.

    Args:
        config: Attention block configuration

    Returns:
        Single MultiHeadAttention module
    """
    # For attention, input and hidden are the same
    return _build_attention_hidden(config)


def _build_attention_hidden(config: AttentionBlockConfig) -> nn.Module:
    """Build core attention layer.

    Template handles norm/activation/dropout assembly.

    Args:
        config: Attention block configuration

    Returns:
        Single MultiHeadAttention module
    """
    return MultiHeadAttention(
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        head_dim=config.head_dim,
        qkv_bias=config.qkv_bias,
        attn_dropout=config.attn_dropout,
        proj_dropout=config.proj_dropout,
        is_causal=config.is_causal,
    )


def _build_attention_output(config: AttentionBlockConfig) -> nn.Module:
    """Build output layer for attention (no separate output layer).

    Args:
        config: Attention block configuration

    Returns:
        Identity module (attention doesn't have a separate output layer)
    """
    # Attention doesn't have a separate output layer
    # Return Identity so template can still apply output_activation if needed
    return nn.Identity()


# ============================================================================
# Public API
# ============================================================================

def attention_block(
    embed_dim: int,
    num_heads: int,
    depth: int = 1,
    head_dim: int | None = None,
    qkv_bias: bool = True,
    attn_dropout: float = 0.0,
    proj_dropout: float = 0.0,
    residual: bool = False,
    pre_norm: bool = True,
    norm: Type[nn.Module] = nn.LayerNorm,
    is_causal: bool = False,
) -> nn.Sequential | nn.Module:
    """Build a multi-head self-attention block.

    Architecture (pre_norm=True, modern Transformer-style):
        x -> norm(x) -> attention(x) -> dropout -> [+ x if residual]

    Architecture (pre_norm=False, original Transformer):
        x -> attention(x) -> dropout -> [+ x if residual] -> norm

    Args:
        embed_dim: Embedding dimension (input/output dimension)
        num_heads: Number of attention heads
        depth: Number of stacked attention layers
        head_dim: Dimension per head (defaults to embed_dim // num_heads)
        qkv_bias: Include bias in Q/K/V projections
        attn_dropout: Dropout applied to attention weights
        proj_dropout: Dropout applied to output projection
        residual: Add residual connection
        pre_norm: Pre-normalization (True, modern) vs post-norm (False, original Transformer)
        norm: Normalization layer class (e.g., nn.LayerNorm)
        is_causal: If True, apply causal masking for autoregressive models

    Returns:
        nn.Sequential module, or Residual-wrapped Sequential if residual=True

    Example:
        >>> # Standard Transformer attention block
        >>> attn = attention_block(512, num_heads=8, residual=True, pre_norm=True)
        >>> x = torch.randn(32, 100, 512)  # (batch, seq_len, embed_dim)
        >>> y = attn(x)  # (32, 100, 512)

        >>> # Vision Transformer (ViT) style
        >>> vit_attn = attention_block(768, num_heads=12, residual=True)
        >>> patches = torch.randn(8, 196, 768)  # (batch, num_patches, embed_dim)
        >>> out = vit_attn(patches)

        >>> # Causal attention (GPT-style)
        >>> gpt_attn = attention_block(512, num_heads=8, residual=True, is_causal=True)
        >>> x = torch.randn(32, 100, 512)
        >>> y = gpt_attn(x)  # (32, 100, 512) with autoregressive masking
    """
    # Note: Attention always preserves shape (batch, seq_len, embed_dim)
    # so residual connections work without dimension checks

    # Create config
    # Note: Attention doesn't use standard activation/dropout (has internal attn_dropout/proj_dropout)
    # So we set activation=Identity and dropout=0.0 in BlockConfig
    config = AttentionBlockConfig(
        embed_dim=embed_dim,
        num_heads=num_heads,
        depth=depth,
        head_dim=head_dim,
        qkv_bias=qkv_bias,
        attn_dropout=attn_dropout,
        proj_dropout=proj_dropout,
        pre_norm=pre_norm,
        norm=norm,
        is_causal=is_causal,
        residual=residual,
        activation=nn.Identity,  # Attention has internal activations
        output_activation=nn.Identity,
        dropout=0.0,  # Attention uses attn_dropout/proj_dropout internally
    )

    # Build using template
    # Note: All dimensions are embed_dim, so pre_norm vs post_norm doesn't affect sizing
    return build_block_template(
        build_input=_build_attention_input,
        build_hidden=_build_attention_hidden,
        build_output=_build_attention_output,
        config=config,
        norm_features_input=embed_dim,
        norm_features_hidden=embed_dim,
        norm_features_output=embed_dim,
        dropout_type="standard",  # Not used since dropout=0.0
        skip_output_extras=True,  # Output is just Identity
    )
