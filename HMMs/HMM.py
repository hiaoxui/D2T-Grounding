import numpy as np
from utils import *
from HMMs._fb_helper import forward, backward
from HMMs.PR import PosteriorRegularization as PR
from framework import Sentence


class HMMs(object):
    def __init__(self, n_state, null_mode, semi=1, pr=None, use_pr_in_decode=True):
        """
        Constructor of HMMs.
        :param int n_state: # of states. I will use `m' in my code to denote this quantity.
        :param str null_mode: Should or not HMMs set a special NULL state.
        If null mode is set as 'without', NULL will not be added.
        If null mode is set as 'with', NULL will be treated as any other states.
        If null mode is set as 'skip', NULL will be added, but it will be skipped during calculating
        transition probability.
        :param semi: The maximum length for every word span. I will use `k' in my code
        to denote this quantity.
        :param PR pr: Posterior regularization.
        """
        self.use_pr_in_decode = use_pr_in_decode

        if n_state <= 0:
            raise Exception('# of state must be larger than 0.')

        if null_mode not in ['without', 'with', 'skip']:
            raise Exception('Unrecognizable null mode configuration.')

        if semi <= 0:
            raise Exception('Semi should be larger than 0.')

        self.m = n_state
        self.k = semi
        self.mode = null_mode

        self.transition_param = np.zeros(0)
        self._init_transition_param()

        # Placeholders
        # M is the dimension of transition matrix (# of all virtual states)
        self.M = 0
        self.transition_matrix = np.zeros(0)
        self._prepare_transition_matrix()

        self.pr = pr

    def _init_transition_param(self):
        m = self.m
        # [ground(:m) + null(m) + start(m+1) + end(m+2)] ^ 2
        mat = np.ones(shape=(m+3, m+3), dtype=np.float64)
        mat[m+1, m+1] = 0
        mat[m+2, :] = 0
        mat[:, m+1] = 0
        self.transition_param = normalize(mat)

    def _idx_map(self, idx):
        """

        :param int idx:
        :rtype: int
        """
        assert idx < self.M
        if idx >= self.m:
            return -1
        return idx

    def _prepare_transition_matrix(self, log_space=True):
        """
        Prepare the transition matrix from parameters.
        :param bool log_space: Whether to convert the results into
        log space
        :return:
        """
        m = self.m
        mat = np.zeros(0)
        if self.mode == 'without':
            self.M = m + 2
            # [ground(:m) + start(m) + end(m+1)] ^ 2
            mat = np.zeros(shape=(m+2, m+2))
            # from ground to ground
            mat[:m, :m] = self.transition_param[:m, :m]
            # from start to ground
            mat[m, :m] = self.transition_param[m+1, :m]
            # from start to end
            mat[m, m+2] = self.transition_param[m+1, m+2]
            # from ground to end
            mat[:m, m+2] = self.transition_param[:m, m+2]
        elif self.mode == 'with':
            self.M = m + 3
            # [ground(:m) + null(m) + start(m+1) + end(m+2)] ^ 2
            mat = self.transition_param.copy()
        elif self.mode == 'skip':
            self.M = m * 2 + 3
            # [ground(:m) + ground_null(m:2m) + start_null(2m) +
            #  start(2m+1) + end(2m + 2)] ^ 2
            mat = np.zeros(shape=(2*m+3, 2*m+3), dtype=np.float64)
            # from ground to ground
            mat[:m, :m] = self.transition_param[:m, :m]
            # from ground to null
            mat[np.arange(m), np.arange(m, 2*m)] = self.transition_param[:m, m]
            # from ground to end
            mat[np.arange(m), 2*m+2] = self.transition_param[:m, m+2]
            # from ground_null to ground and null
            mat[m:2*m, :2*m] = mat[:m, :2*m]
            # from ground_null to end
            mat[m:2*m, 2*m+2] = mat[:m, 2*m+2]
            # from start to ground
            mat[2*m+1, :m] = self.transition_param[m+1, :m]
            # from start to null
            mat[2*m+1, 2*m] = self.transition_param[m+1, m]
            # from start to end
            mat[2*m+1, 2*m+2] = self.transition_param[m+1, m+2]
            # from start_null to ground
            mat[2*m, :m] = mat[2*m+1, :m]
            # from start_null to null
            mat[2*m, 2*m] = mat[2*m+1, 2*m]
            # from start_null to end
            mat[2*m, 2*m+2] = mat[2*m+1, 2*m+2]

        if log_space:
            mat = safe_log(mat)

        self.transition_matrix = mat

        return mat

    def _phi(self, emit, log_space=True):
        """
        Calculate \phi variable, which could be treated as combination of emission
        scores and transition scores.
        :param np.ndarray emit: Shape [s, k, m+1(ground + null)]. Emission scores.
        :return np.ndarray: \phi. Shape [s+1, k, M], including the special END but
        excluding special START state.
        """
        M = self.M
        s, k, m = emit.shape
        m -= 1
        assert m == self.m
        assert k == self.k

        # emit_: [s+1, k, M]
        emit_ = np.zeros(shape=(s+1, k, M), dtype=np.float64)
        # for all ground emission
        emit_[:-1, :, :m] = emit[:, :, :-1]
        # emit_: [M, s+1, k]
        emit_ = emit_.transpose([2, 0, 1])
        # for null
        emit_[m:-2, :-1, :] = emit[:, :, -1]
        # emit_: [s+1, k, M]
        emit_ = emit_.transpose([1, 2, 0])
        # for end
        emit_[-1, 0, -1] = 1.0

        if log_space:
            emit_ = safe_log(emit_)

        # phi: [s+1, k, M(to), M(from)]
        phi = emit_.repeat(M).reshape(s+1, k, M, M)
        # phi: [s+1, k, M(from), M(to)]
        phi = phi.transpose([0, 1, 3, 2])
        phi = phi + self.transition_matrix

        return phi

    def _accumulate_transition_count(self, phi, alpha, beta, cnt):
        """
        Update transition count matrix.
        :param np.ndarray phi: Shape [s+1, k, M(from), M(to)]
        :param np.ndarray alpha: Shape [s+2, k(to), M(to)]
        :param np.ndarray beta: Shape [s+2, k(from), M(from)]
        :param no.ndarray cnt: Shape [M(from), M(to)]
        :return:
        """
        s, k, M = alpha.shape
        s -= 2
        assert k == self.k
        assert M == self.M

        offset = k
        z = alpha[-1, 0, -1]
        pad = np.zeros(shape=(offset, k, M), dtype=np.float64)
        pad = safe_log(pad)
        alpha = np.concatenate([pad, alpha], axis=0)
        for idx in range(s+1):
            # alpha_clip: [k(to), k(from), M(from)]
            alpha_clip = alpha[range(offset+idx, offset+idx-k, -1), :, :]
            # alpha_ext: [k(to), k(from), M(from), M(to)]
            alpha_ext = alpha_clip.repeat(M).reshape(k, k, M, M)
            # alpha_ext: [k(from), k(to), M(from), M(to)]
            alpha_ext = alpha_ext.transpose([1, 0, 2, 3])

            # phi_clip: [k(to), M(from), M(to)]
            phi_clip = phi[idx, :, :, :]
            # phi_ext: [k(to), M(from), M(to), k(from)]
            phi_ext = phi_clip.repeat(k).reshape(k, M, M, k)
            # phi_ext: [k(from), k(to), M(from), M(to)]
            phi_ext = phi_ext.transpose([3, 0, 1, 2])

            # beta_clip: [k(to), M(to)]
            beta_clip = beta[idx+1, :, :]
            # beta_ext: [k(to), M(to), k(from), M(from)]
            beta_ext = beta_clip.repeat(k * M).reshape(k, M, k, M)
            # beta_ext: [k(from), k(to), M(from), M(to)]
            beta_ext = beta_ext.transpose([2, 0, 3, 1])

            # to_sum: [k(from), k(to), M(from), M(to)]
            to_sum = alpha_ext + phi_ext + beta_ext
            # to_sum [k*k(to sum), M(from), M(to)]
            to_sum = to_sum.reshape(k*k, M, M)

            # counter: [M(from), M(to)]
            counter = log_sum_exp(to_sum)
            counter = counter - z
            counter = np.exp(counter)

            cnt += counter

    def forward_backward(self, emit, log_space=False, cnt=None, sentence=None):
        """
        Given emission, calculate the soft count of latent variables and possibly update
        transition count matrix.
        Note at this moment I will not update the parameters of transition matrix, you should
        call update_transition intently.
        :param np.ndarray emit: Shape [s, k, m+1]
        :param bool log_space: Whether to convert the soft count matrix into log space
        :param np.ndarray cnt: Counter
        :param Sentence sentence:
        :return:
        """
        s, k, m = emit.shape
        m -= 1
        assert k == self.k
        assert m == self.m

        # phi: [s+1, k, M]
        phi = self._phi(emit)
        if self.pr is None:
            # alpha: [s+2, k(to), M(to)]
            alpha = forward(phi)
            # beta: [s+2, k(from), M(from)]
            beta = backward(phi)
        else:
            phi, alpha, beta = self.pr.project(phi, sentence)

        # normalization
        z = alpha[-1, 0, -1]
        z_ = beta[0, 0, -2]
        epsilon = 1e-3
        assert 1-epsilon < z/z_ < 1+epsilon

        # mu: [s+2, k, M]
        mu = alpha + beta - z
        if not log_space:
            mu = np.exp(mu)

        if cnt is not None:
            if not log_space:
                self._accumulate_transition_count(phi, alpha, beta, cnt)
            else:
                self._accumulate_transition_count(np.exp(phi), np.exp(alpha), np.exp(beta), cnt)

        # soft_count: [s, k, m+1]
        soft_count = np.zeros(shape=(s, k, m+1))
        soft_count[:, :, :m] = mu[1:-1, :, :m]
        soft_count[:, :, -1] = mu[1:-1, :, m:-2].sum(axis=2)

        return soft_count

    def update_transition(self, cnt):
        """

        :param np.ndarray/list[np.ndarray] cnt:
        :rtype: None
        """
        if isinstance(cnt, list):
            # cnt: [M, M]
            cnt = np.sum(cnt, axis=0)
        # new_param: [m+3, m+3]
        new_param = np.zeros_like(self.transition_param)
        m, M = self.m, self.M

        if self.mode == 'without':
            # from ground to ground
            new_param[:m, :m] = cnt[:m, :m]
            # from ground to end
            new_param[:m, -1] = cnt[:m, -1]
            # from begin to ground
            new_param[-2, :m] = cnt[-2, :m]
            # from begin to end
            new_param[-2, -1] = cnt[-2, -1]
        elif self.mode == 'with':
            # from all to all
            new_param[:, :] = cnt[:, :]
        else:
            # from ground to ground
            new_param[:m, :m] = cnt[:m, :m] + cnt[m:2*m, :m]
            # from ground to null
            new_param[:m, m] = (cnt[:m, m:2*m] + cnt[m:2*m, m:2*m]).sum(axis=1)
            # from ground to end
            new_param[:m, m+2] = cnt[:m, 2*m+2] + cnt[m:2*m, 2*m+2]
            # from begin to ground
            new_param[m+1, :m] = cnt[2*m:2*m+2, :m].sum(axis=0)
            # from begin to null
            new_param[m+1, m] = cnt[2*m:2*m+2, 2*m:2*m+2].sum()
            # from begin to end
            new_param[m+1, m+2] = cnt[2*m:2*m+2, 2*m+2].sum(axis=0)

        # Apply updates
        self.transition_param = normalize(new_param)
        self._prepare_transition_matrix()

    def decode(self, emit, sentence):
        """

        :param np.ndarray emit: Shape [s, k, m+1(ground + null)]. Emission scores.
        :param Sentence sentence:
        :rtype: list
        """
        s, k, m = emit.shape
        m -= 1
        assert k == self.k
        assert m == self.m
        M = self.M

        # phi: [s+1, k, M]
        phi = self._phi(emit)
        if self.pr is not None and self.use_pr_in_decode:
            phi, _, _ = self.pr.project(phi, sentence)

        # score: [s+2, M]
        score = np.zeros(shape=(s+2, M), dtype=np.float64)
        score[0, M-2] = 1.0

        offset = k
        pad = np.zeros(shape=(offset, M), dtype=np.float64)
        score = np.concatenate([pad, score], axis=0)
        score = safe_log(score)

        # back_pointers: [s+1, M, [k_idx, M_idx]]
        back_pointers = np.zeros(shape=(s+1, M, 2), dtype=np.int32)

        for idx in range(s+1):
            # score_clip: [k(to), M(from)]
            score_clip = score[offset+idx:offset+idx-k:-1, :]
            # score_ext: [k(to), M(from), M(to)]
            score_ext = score_clip.repeat(M).reshape(k, M, M)

            # candidate: [k(to), M(from), M(to)]
            candidate = phi[idx] + score_ext

            score[offset+idx+1] = candidate.max(axis=(0, 1))

            # candidate_flatten: [k(to) * M(from), M(to)]
            candidate_flatten = candidate.reshape(k * M, M)

            # position_mixed: [M(to)]
            position_mixed = candidate_flatten.argmax(axis=0)

            back_pointers[idx, :, 0] = position_mixed // M
            back_pointers[idx, :, 1] = position_mixed % M

        max_score = score[-1, -1]

        trace = [M-1]
        idx = s
        while idx >= 0:
            curr_state = trace[0]
            k_idx, M_idx = back_pointers[idx, curr_state]
            for _ in range(k_idx):
                trace.insert(0, trace[0])
            trace.insert(0, M_idx)
            idx = idx - k_idx - 1

        trace = trace[1:-1]
        trace = [self._idx_map(idx) for idx in trace]

        return trace, max_score

    def blank_counter(self):
        """

        :rtype: np.ndarray
        """
        return np.zeros(shape=(self.M, self.M), dtype=np.float64)
