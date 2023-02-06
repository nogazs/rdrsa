import logging
import warnings
import numpy as np
import pandas as pd
from IPython.display import display, display_html
from scipy.special import entr, xlogy
from prag_models.lexicon import normalize_rows
from utils.project import get_logger, running_in_notebook


_logger = get_logger('prag_comm')
pd.set_option('display.float_format', lambda x: '%.3f' % x)

if running_in_notebook():
    _logger.setLevel(logging.ERROR)


def set_logging_level(level_name):
    _logger.setLevel(logging._nameToLevel[level_name])


def set_rand_seed(seed):
    np.random.seed(seed)


class Agent(object):

    def __init__(self, comm_model, state_0=None):
        self.comm_model = comm_model
        self.state_0 = state_0 if state_0 is not None else self._init_state_0(comm_model)
        self.shape = self.state_0.shape
        self.formatter = None

    def __call__(self, *args):
        return self._state()

    def __str__(self):
        return self.to_df().__str__()

    def _state(self, copy=False):
        raise NotImplementedError

    def _set_state(self, state):
        raise NotImplementedError

    def _init_state_0(self, comm_model):
        raise NotImplementedError

    def reset(self, state=None):
        self._set_state(state if state is not None else self.state_0)

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def display(self):
        display(self.to_df())

    def to_df(self):
        if self.formatter is not None:
            return self.formatter(self._state())
        return pd.DataFrame(self._state())

    def copy(self, reset=False):
        new = self.__class__(self.comm_model, self.state_0.copy())
        if not reset:
            new.state = self._state(copy=True)
        return new


# ~~~~~~~~~~~
# SPEAKER
# ~~~~~~~~~~~
class Speaker(Agent):

    def __init__(self, comm_model, alpha=None, S0=None):
        super().__init__(comm_model, S0)
        self.S = self.state_0
        self.alpha = alpha
        self.utility = comm_model.utility
        self.formatter = comm_model.lex.speaker_df

    def _state(self, copy=False):
        if copy:
            return self.S.copy()
        return self.S

    def _set_state(self, state):
        self.S = state

    def _init_state_0(self, comm_model):
        return comm_model.lex.lit_speaker(comm_model.pM)

    def inject_noise(self, noise_level):
        if noise_level > 0:
            self.S = normalize_rows(self.S + noise_level * np.random.rand(*self.shape))

    def copy(self, reset=False):
        new = super(Speaker, self).copy()
        new.alpha = self.alpha
        return new

    def update(self, L):
        raise NotImplementedError


# ~~~~~~~~~~~
# LISTENER
# ~~~~~~~~~~~
class Listener(Agent):

    def __init__(self, comm_model, L0=None):
        super().__init__(comm_model, L0)
        self.L = self.state_0
        self.formatter = comm_model.lex.listener_df

    def _state(self, copy=False):
        if copy:
            return self.L.copy()
        return self.L

    def _set_state(self, state):
        self.L = state

    def _init_state_0(self, comm_model):
        return comm_model.lex.lit_listener(comm_model.pM)

    def update(self, S):
        raise NotImplementedError


class BayesianListener(Listener):

    def __init__(self, comm_model, L0=None):
        super().__init__(comm_model, L0)
        self.pM = comm_model.pM

    def update(self, S):
        pMU = S * self.pM[:, None]
        S_U = S.T.dot(self.pM)[:, None]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # np.where takes 0 div
            self.L = np.where(S_U > 0, pMU.T / S_U, self.pM)
        return self.L


# ~~~~~~~~~~~
# COMMUNICATION MODEL
# ~~~~~~~~~~~
class PragCommModel(object):

    def __init__(self, lex, pM=None, C=0.,
                 listener_class=BayesianListener,
                 speaker_class=Speaker, **kwargs):
        """

        :param lex:
        :param pM:
        :param C:
        :param listener_class:
        :param speaker_class:
        """
        self.lex = lex
        self.pM = pM if pM is not None else np.ones(lex.nM) / lex.nM
        _C = np.asarray(C)
        if len(_C.shape) == 1:
            self.C = np.repeat(_C[:, None], lex.nM, axis=1).T
        else:
            self.C = _C
        self.speaker = speaker_class(self)
        self.listener = listener_class(self)
        self.complexity_str = ''
        self.objective_func_str = ''
        self.name = ''

    def __call__(self, alpha=1., **kwargs):
        if len(np.asarray(alpha).shape) > 0:
            trajs = [None] * len(alpha)
            for i, a in enumerate(alpha):
                trajs[i] = self.solve(a, **kwargs)
            return trajs
        return self.solve(alpha, **kwargs)

    def utility(self, L):
        """
        default utility function in RSA
        :param L: L(m|u)
        :return: log(L(m|u)) - C(m, u)
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # it's ok to return log(0) = -inf
            return np.log(L) - self.C.T

    def expected_utility(self, S, L):
        SlogL = xlogy(S, L.T)
        return ((SlogL - S * self.C).T @ self.pM).sum()

    def speaker_entropy(self, S):
        return (entr(S).T @ self.pM).sum()

    def information_rate(self, S):
        H_U_M = self.speaker_entropy(S)
        H_U = entr(S.T @ self.pM).sum()
        return H_U - H_U_M

    def complexity(self, S):
        raise NotImplementedError

    def objective_func(self, S, L, alpha, minimized=True):
        raise NotImplementedError

    def compare(self, S1, S2, alpha):
        """
        Returns True if S2 a better solution than S1 for the given value of alpha
        """
        L1 = self.listener.copy(reset=True).update(S1)
        L2 = self.listener.copy(reset=True).update(S2)
        return self.objective_func(S2, L2, alpha, minimized=True) < self.objective_func(S1, L1, alpha, minimized=True)

    def display_cost(self):
        if len(self.C.shape) > 0:
            display(self.lex.speaker_df(self.C))
        else:
            print(self.C)

    def solve(self, alpha, init_mode='L0', keep_traj=True, T=int(1e6), **kwargs):
        traj = Trajectory(self, alpha, on=keep_traj, max_length=T + 2)
        self._init_solve(alpha, init_mode, traj=traj)
        return self._solve(traj=traj, keep_traj=keep_traj, max_itr=T, **kwargs)

    def _init_solve(self, alpha, init_mode='L0', S0=None, traj=None):
        # reset agents
        self.speaker.reset(S0)
        self.listener.reset()
        self.speaker.alpha = alpha
        # initialize
        if init_mode == 'S0':
            _logger.info('S0 initialization')
            self.listener.update(self.speaker())
        else:
            _logger.info('L0 initialization')
        if traj is not None:
            traj.add(self.speaker(), self.listener())  # (S0, L0)
        self.speaker.update(self.listener())  # S1

    def _solve(self, max_itr=int(1e6), report_time=100, tol=1e-5, keep_traj=True, traj=None, **kwargs):
        alpha = self.speaker.alpha
        S = self.speaker()
        if traj is None:
            traj = Trajectory(self, alpha, on=keep_traj, max_length=max_itr + 1)
            traj.add(S, self.listener())
        _logger.info('solving %s with alpha = %g' % (self.name, alpha))
        # solve model
        for t in range(max_itr):
            # update agents
            S_prev = S
            L = self.listener.update(S_prev)
            S = self.speaker.update(L)
            traj.add(S_prev, L)
            # check convergence
            dS = np.abs(S_prev - S).max()
            change = dS >= tol and t < max_itr
            self._log_info(t, S, L, alpha, dS, print_log=t % report_time == 0 or not change or t == max_itr - 1)
            if not change:
                break
        if keep_traj:
            traj.add(S, self.listener.update(S))
            traj.squeeze()
            return traj
        else:
            return S, L, self.expected_utility(S, L), self.complexity(S)

    def _log_info(self, t, S, L, alpha, dS, print_log=True):
        if print_log:
            EV = self.expected_utility(S, L)
            obj_func = self.objective_func(S, L, alpha)
            _logger.info('[t = %d] obj_func = %.5f | EV = %.5f | dS = %.5f' % (t, obj_func, EV, dS))


class Trajectory(object):

    def __init__(self, comm_model, alpha, on=True, annealing_mode=False, max_length=1e5):
        self.comm_model = comm_model
        self.alpha = alpha
        max_length = int(max_length * on)
        self.S = [None] * max_length
        self.L = [None] * max_length
        self.EV = np.zeros(max_length)
        self.complx = np.zeros(max_length)
        self.obj_func = np.zeros(max_length)
        self.length = 0
        self.on = on
        self.annealing_mode = annealing_mode
        self.E_projs = None
        self.I_projs = None

    def add(self, S, L, EV=None, complx=None):
        if self.on:
            i = self.length
            self.S[i] = S.astype('float32')
            self.L[i] = L.astype('float32')
            self.EV[i] = EV if EV is not None else self.comm_model.expected_utility(S, L)
            self.complx[i] = complx if complx is not None else self.comm_model.complexity(S)
            if not self.annealing_mode:
                self.obj_func[i] = self.comm_model.objective_func(S, L, self.alpha)
            self.length += 1

    def squeeze(self):
        if self.on:
            l = self.length
            self.S = self.S[:l]
            self.L = self.L[:l]
            self.complx = self.complx[:l]
            self.EV = self.EV[:l]
            self.obj_func = self.obj_func[:l]

    def reverse(self):
        self.S.reverse()
        self.L.reverse()
        self.complx = np.flip(self.complx)
        self.EV = np.flip(self.EV)
        self.obj_func = np.flip(self.obj_func)

    def alt_projs(self):
        if self.E_projs is None:
            self.E_projs = np.zeros(self.length - 1)
            self.I_projs = np.zeros(self.length - 1)
            for t in range(self.length - 1):
                self.E_projs[t] = self.comm_model.alt_min_obj(self.S[t], self.L[t], self.alpha)
                self.I_projs[t] = self.comm_model.alt_min_obj(self.S[t + 1], self.L[t], self.alpha)
        return self.E_projs, self.I_projs

    def get_state(self, t=-1):
        return self.S[t], self.L[t]

    def display_state(self, t=-1, precision=3):
        alpha = self.alpha[t] if self.annealing_mode else self.alpha
        S_df = self.comm_model.lex.speaker_df(self.S[t])
        L_df = self.comm_model.lex.listener_df(self.L[t])
        sub = '*' if t == -1 or self.annealing_mode else '%d' % t
        if running_in_notebook():  # nicer html format for notebooks
            cap_style = '<span style="color: #777777; text-align: left;">{}</span>'
            caption = '%s: {}%s \xa0\xa0 (𝛼 = %.4g)' % (self.comm_model.name, sub, alpha)
            caption = cap_style.format(caption)
            S_styler = S_df.style.set_table_attributes("style='display:inline' class='dataframe'"). \
                set_caption(caption.format('S')).format(precision=precision)
            L_styler = L_df.style.set_table_attributes("style='display:inline' class='dataframe'"). \
                set_caption(caption.format('L')).format(precision=precision)
            display_html(S_styler._repr_html_() + '\xa0' * 3 + L_styler._repr_html_(), raw=True)
        else:  # simple display
            caption = '\n%s: {}%s\t(𝛼 = %.4g)' % (self.comm_model.name, sub, alpha)
            print(caption.format('S'))
            display(S_df)
            print(caption.format('L'))
            display(L_df)

    def display_all(self, states=None):
        states = range(self.length) if states is None else states
        for t in states:
            self.display_state(t)

