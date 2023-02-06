import numpy as np
from scipy.special import logsumexp
from prag_models.prag_comm import Speaker, PragCommModel


class RSASpeaker(Speaker):

    def __init__(self, comm_model, S0=None, **kwargs):
        super().__init__(comm_model, S0)

    def update(self, L):
        v = self.utility(L).T
        alpha_v = np.where(v == -np.inf, -np.inf, self.alpha * v)
        Z = logsumexp(alpha_v, axis=1)[:, None]
        self.S = np.exp(alpha_v - Z)
        self.S = np.where(np.isnan(self.S), 0, self.S)
        assert (np.abs(self.S.sum(axis=1) - 1) < 1e-8).all()
        return self.S


class RSA(PragCommModel):
    """
    implementation of the RSA model
    """

    def __init__(self, lex, pM=None, C=0.):
        super().__init__(lex, pM, C, speaker_class=RSASpeaker)
        self.complexity_str = '$H_S(U|M)$'
        self.objective_func_str = '$\mathcal{G}_{\\alpha}$'
        self.name = 'RSA'

    def complexity(self, S):
        return self.speaker_entropy(S)

    def objective_func(self, S, L, alpha, minimized=False):
        G_alpha = self.speaker_entropy(S) + alpha * self.expected_utility(S, L)
        return (1 - 2 * minimized) * G_alpha

