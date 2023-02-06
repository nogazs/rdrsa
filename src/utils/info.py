import numpy as np
from math import log
from scipy.special import entr, rel_entr

EPSILON = 1e-16


def H(p, axis=None, base=2):
    return entr(p).sum(axis=axis) / np.log(base)


def MI(pXY, base=2):
    return H(pXY.sum(axis=0), base=base) + H(pXY.sum(axis=1), base=base) - H(pXY, base=base)
