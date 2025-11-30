"""Nguyen-Wainwright-Jordan (NWJ) variational loss for density ratio estimation."""

import torch


def nwj_loss(
    numerator_critics: torch.Tensor,
    denominator_critics: torch.Tensor,
) -> torch.Tensor:
    """Compute Nguyen-Wainwright-Jordan variational loss.

    NWJ representation: KL(p || q) >= E_p[f(x)] - e^{E_q[exp(f(x))] - 1}

    The NWJ bound differs from DV by using exp(log_mean_exp - 1) instead of
    log_mean_exp directly. This provides a tighter bound in some cases but
    can have higher variance.

    Loss (to minimize): L = -E_p[f] + exp(log E_q[exp(f)] - 1)

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

    # Second term: exp(log E_q[exp(f)] - 1) using log-sum-exp trick
    # log(mean(exp(f))) = logsumexp(f) - log(N)
    n_samples = denominator_critics.shape[0]
    log_mean_exp = torch.logsumexp(denominator_critics, dim=0) - torch.log(
        torch.tensor(n_samples, dtype=denominator_critics.dtype, device=denominator_critics.device)
    )
    second_term = torch.exp(log_mean_exp - 1.0)

    return first_term + second_term
