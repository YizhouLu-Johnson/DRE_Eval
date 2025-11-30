"""Donsker-Varadhan variational loss for density ratio estimation."""

import torch


def dv_loss(
    numerator_critics: torch.Tensor,
    denominator_critics: torch.Tensor,
) -> torch.Tensor:
    """Compute Donsker-Varadhan variational loss.

    DV representation: KL(p || q) = sup_f E_p[f(x)] - log E_q[exp(f(x))]
    Loss (to minimize): L = -E_p[f] + log E_q[exp(f)]

    Uses log-sum-exp trick for numerical stability:
        log E[exp(f)] = logsumexp(f) - log(N)

    Args:
        numerator_critics: Critic outputs for samples from numerator distribution (p).
            Shape: (batch_size,) or (batch_size, 1)
        denominator_critics: Critic outputs for samples from denominator distribution (q).
            Shape: (batch_size,) or (batch_size, 1)

    Returns:
        Scalar loss value (negative lower bound on KL divergence)
    """
    # Flatten in case of shape (batch_size, 1)
    numerator_critics = numerator_critics.view(-1)
    denominator_critics = denominator_critics.view(-1)

    # First term: -E_p[f]
    first_term = -numerator_critics.mean()

    # Second term: log E_q[exp(f)] using log-sum-exp trick
    # log(mean(exp(f))) = logsumexp(f) - log(N)
    n_samples = denominator_critics.shape[0]
    second_term = torch.logsumexp(denominator_critics, dim=0) - torch.log(
        torch.tensor(n_samples, dtype=denominator_critics.dtype, device=denominator_critics.device)
    )

    return first_term + second_term
