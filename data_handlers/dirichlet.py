import numpy as np
from scipy.special import gammaln

from utils.distribution_utils import dirichlet_kl


class DIRICHLET:
    """
    Synthetic Dirichlet dataset that mirrors the GAUSSIANS interface.
    """

    class Data:
        def __init__(self, data):
            self.x = data
            self.ldj = 0.0
            self.N = data.shape[0]

    def __init__(
        self,
        n_samples,
        numerator_concentration,
        denominator_concentration,
        n_dims=None,
        seed=None,
        **kwargs,
    ):
        self.num_alpha = np.asarray(numerator_concentration, dtype=np.float64)
        self.denom_alpha = np.asarray(denominator_concentration, dtype=np.float64)
        if self.num_alpha.shape != self.denom_alpha.shape:
            raise ValueError("Numerator and denominator concentration vectors must match.")
        self.n_dims = int(n_dims or self.num_alpha.shape[0])
        if self.num_alpha.shape[0] != self.n_dims:
            raise ValueError("n_dims mismatch for Dirichlet dataset.")
        if np.any(self.num_alpha <= 0) or np.any(self.denom_alpha <= 0):
            raise ValueError("Dirichlet concentration parameters must be > 0.")

        self.rng = np.random.RandomState(seed) if seed is not None else np.random
        trn = self.sample_data(n_samples)
        val = self.sample_data(n_samples)
        tst = self.sample_data(n_samples)

        self.trn = self.Data(trn)
        self.val = self.Data(val)
        self.tst = self.Data(tst)
        self.true_kl = dirichlet_kl(self.num_alpha, self.denom_alpha)

    def _sample_dirichlet(self, alpha, n_samples):
        return self.rng.dirichlet(alpha, size=n_samples)

    def sample_data(self, n_samples):
        return self._sample_dirichlet(self.num_alpha, n_samples)

    def sample_denominator(self, n_samples):
        return self._sample_dirichlet(self.denom_alpha, n_samples)

    @staticmethod
    def _log_norm(alpha):
        return gammaln(np.sum(alpha)) - np.sum(gammaln(alpha))

    def numerator_log_prob(self, samples):
        return self._log_prob(samples, self.num_alpha)

    def denominator_log_prob(self, samples):
        return self._log_prob(samples, self.denom_alpha)

    def _log_prob(self, samples, alpha):
        samples = np.asarray(samples, dtype=np.float64)
        if samples.shape[-1] != alpha.shape[0]:
            raise ValueError("Sample dimension mismatch for Dirichlet log-prob.")
        log_norm = self._log_norm(alpha)
        return log_norm + np.sum((alpha - 1.0) * np.log(samples + 1e-20), axis=-1)
