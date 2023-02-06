import warnings
import numpy as np
from scipy.special import logsumexp
from prag_models.prag_comm import Speaker, PragCommModel


class RDRSASpeaker(Speaker):

    def __init__(self, comm_model, S0=None, pY_M=None, **kwargs):
        super().__init__(comm_model, S0, **kwargs)
        self.pM = comm_model.pM

    def update(self, L):
        v = self.utility(L).T
        pU = self.S.T.dot(self.pM)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ok if log(0) = -inf
            alpha_v = np.where(v == -np.inf, -np.inf, self.alpha * v)
            logits = np.log(pU[None]) + alpha_v
        Z = logsumexp(logits, axis=1)[:, None]
        self.S = np.exp(logits - Z)
        assert (np.abs(self.S.sum(axis=1) - 1) < 1e-8).all()
        return self.S


class RDRSA(PragCommModel):
    """
    implementation of the RD-RSA model
    """

    def __init__(self, lex, pM=None, C=0., **kwargs):
        super().__init__(lex, pM, C, speaker_class=RDRSASpeaker)
        self.complexity_str = '$I_S(M;U)$'
        self.objective_func_str = '$\mathcal{F}_{\\alpha}$'
        self.name = 'RD-RSA'

    def complexity(self, S):
        return self.information_rate(S)

    def objective_func(self, S, L, alpha, minimized=True):
        F_alpha = self.information_rate(S) - alpha * self.expected_utility(S, L)
        return (1 - 2 * (not minimized)) * F_alpha

