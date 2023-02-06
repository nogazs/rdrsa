import numpy as np
import pandas as pd
from utils.project import get_logger
from IPython.display import display


_LOGGER = get_logger('lexicon')
pd.set_option('display.float_format', lambda x: '%.3f' % x)


def normalize_rows(A, p0=None):
    Z = A.sum(axis=1)[:, None]
    p0 = p0 if p0 is not None else 1/A.shape[1]
    return np.where(Z > 0, A / A.sum(axis=1)[:, None], p0)


class Lexicon(object):

    def __init__(self, lex_arr, M_labels=None, U_labels=None):
        self.nU, self.nM = lex_arr.shape
        self.M_labels = M_labels
        self.U_labels = U_labels
        self._lex_arr = lex_arr

    def __call__(self):
        return self._lex_arr.copy()

    def __add__(self, a):
        if type(a) == Lexicon:
            assert self._lex_arr.shape == a._lex_arr.shape
            return Lexicon(self._lex_arr + a._lex_arr, self.M_labels, self.U_labels)
        else:
            return Lexicon(self._lex_arr + a, self.M_labels, self.U_labels)

    def __mul__(self, a):
        return Lexicon(self._lex_arr * a, self.M_labels, self.U_labels)

    def __getitem__(self, i):
        return self._lex_arr[i]

    def __setitem__(self, i, value):
        self._lex_arr[i] = value

    def clip(self, min_val):
        return Lexicon(self._lex_arr.clip(min_val), self.M_labels, self.U_labels)

    def lit_speaker(self, pM=None):
        if pM is None:
            return normalize_rows(self().T)
        else:
            return normalize_rows(self().T * pM[:, None])

    def lit_listener(self, pM=None):
        if pM is None:
            return normalize_rows(self())
        else:
            return normalize_rows(self() * pM[None])

    def to_df(self, A=None):
        A = self._lex_arr if A is None else A
        return pd.DataFrame(A, index=self.U_labels, columns=self.M_labels)

    def display(self):
        display(self.to_df())

    def speaker_df(self, S):
        return pd.DataFrame(S, index=self.M_labels, columns=self.U_labels)

    # def listener_df(self, L):
    #     return self.to_df(L)

    def listener_df(self, L, rows=None):
        if rows is None:
            return self.to_df(L)
        else:
            return pd.DataFrame(L, index=[self.U_labels[r] for r in rows], columns=self.M_labels)
