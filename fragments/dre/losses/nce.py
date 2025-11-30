"""Noise Contrastive Estimation (NCE) loss for density ratio estimation."""

import torch
import torch.nn.functional as F


def nce_loss(
    numerator_logits: torch.Tensor,
    denominator_logits: torch.Tensor,
) -> torch.Tensor:
    """Compute NCE (Noise Contrastive Estimation) loss.

    NCE frames density ratio estimation as binary classification:
    - Numerator samples (from p) labeled as class 0
    - Denominator samples (from q) labeled as class 1

    At optimum: log(p/q) = -logit

    Args:
        numerator_logits: Critic outputs for samples from numerator distribution (p).
            Shape: (batch_size,) or (batch_size, 1)
        denominator_logits: Critic outputs for samples from denominator distribution (q).
            Shape: (batch_size,) or (batch_size, 1)

    Returns:
        Scalar loss value (binary cross-entropy sum)
    """
    # Flatten in case of shape (batch_size, 1)
    numerator_logits = numerator_logits.view(-1)
    denominator_logits = denominator_logits.view(-1)

    # Numerator samples (from p) should be classified as 0
    num_loss = F.binary_cross_entropy_with_logits(
        numerator_logits,
        torch.zeros_like(numerator_logits),
    )

    # Denominator samples (from q) should be classified as 1
    den_loss = F.binary_cross_entropy_with_logits(
        denominator_logits,
        torch.ones_like(denominator_logits),
    )

    return num_loss + den_loss
