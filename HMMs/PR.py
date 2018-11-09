import numpy as np
from framework import Sentence
from ._fb_helper import forward, backward
from .PR_helper import null_b
from scipy.optimize import minimize


class PosteriorRegularization(object):
    def __init__(self, m, null_mode, k, cfg):
        self.cfg = cfg
        self.m = m
        self.M = 0
        self.mode = null_mode
        self.k = k
        self.features = list()
        self.boundary = list()
        self.b = list()
        if null_mode != 'without' and cfg.null_ratio > 0.:
            self._add_null_feature()
        self.f = len(self.features)
        self.mask = np.zeros(0)
        self._concat_feature()
        self.slack = cfg.slack

    def _null_mask(self):
        if self.mode == 'with':
            self.M = self.m + 3
            mask = np.zeros(shape=(self.k, self.M), dtype=np.bool)
            mask[:, self.m:self.m+1] = True
        elif self.mode == 'skip':
            self.M = self.m * 2 + 3
            mask = np.zeros(shape=(self.k, self.M), dtype=np.bool)
            mask[:, self.m:self.m*2+1] = True
        else:
            print('without is not allowed.')
            assert False
        return mask

    def _add_null_feature(self):

        null_mask = self._null_mask()

        # Shape: [k, M(to), 1]
        mask = np.zeros(shape=(self.k, self.M, 1), dtype=np.float64)
        mask[:, :, 0][null_mask] = -1.0

        self.features.append(mask)

        self.b.append(null_b)

        self.boundary.append((0, None))

    def _concat_feature(self):
        """
        Concatenate all feature matrix into a tensor.
        :rtype: None
        """
        # Shape: [k, M(to), f]
        self.mask = np.concatenate(self.features, axis=2)

    def project(self, phi, sentence):
        """
        Project onto Q.
        :param np.ndarray phi: Shape=[s+1, k, M(from), M(to)]
        :param Sentence sentence:
        :rtype: np.ndarray
        """
        b_list = [b_(sentence, self.cfg) for b_ in self.b]
        b_list = np.array(b_list)

        phi_ = [phi.copy()]
        alpha = [np.zeros(0)]
        beta = [np.zeros(0)]
        phi_lbd = [np.ones(shape=(self.f, ), dtype=np.float64)]
        alpha_lbd = [np.ones(shape=(self.f, ), dtype=np.float64)]
        beta_lbd = [np.ones(shape=(self.f, ), dtype=np.float64)]

        def func(lbd):
            """
            Value function.
            :param np.ndarray lbd: lambda
            :rtype: float
            """
            phi_app = self.mask @ lbd

            if any(lbd != phi_lbd[0]):
                # Shape: [s+1, M(from), k, M(to)]
                phi_[0] = phi.transpose([0, 2, 1, 3]) - phi_app
                # Shape: [s+1, k, M(from), M(to)]
                phi_[0] = phi_[0].transpose([0, 2, 1, 3])

                phi_lbd[0] = lbd.copy()

            if any(lbd != alpha_lbd[0]):
                alpha[0] = forward(phi_[0])
                alpha_lbd[0] = lbd.copy()

            rst = alpha[0][-1, 0, -1]
            rst += b_list @ lbd
            rst += self.cfg.slack * np.sqrt(lbd @ lbd)
            return rst

        def jac(lbd):
            """
            Jacobian Function
            :param np.ndarray lbd: lambda
            :rtype: np.ndarray
            """
            phi_app = self.mask @ lbd

            if any(lbd != phi_lbd[0]):
                # Shape: [s+1, M(from), k, M(to)]
                phi_[0] = phi.transpose([0, 2, 1, 3]) - phi_app
                # Shape: [s+1, k, M(from), M(to)]
                phi_[0] = phi_[0].transpose([0, 2, 1, 3])

                phi_lbd[0] = lbd.copy()

            if any(lbd != alpha_lbd[0]):
                alpha[0] = forward(phi_[0])
                alpha_lbd[0] = lbd.copy()

            if any(lbd != beta_lbd[0]):
                beta[0] = backward(phi_[0])
                beta_lbd[0] = lbd.copy()

            z = alpha[0][-1, 0, -1]
            z_ = beta[0][0, 0, -2]
            assert 1-1e-3 < z/z_ < 1+1e-3

            rst = b_list.copy()
            gamma = np.exp(alpha[0] + beta[0] - z)
            counter = gamma.sum(axis=0)
            counter = counter.reshape(self.M * self.k)
            rst -= counter @ self.mask.reshape(self.M * self.k)
            rst += self.cfg.slack * lbd / np.sqrt(lbd@lbd + 1e-100)

            return rst

        guess = np.zeros(shape=(self.f, ), dtype=np.float64)

        lbd_star = minimize(
            fun=func,
            jac=jac,
            x0=guess,
            # method='L-BFGS-B',
            method='SLSQP',
            bounds=self.boundary,
            options={
                'maxiter': self.cfg.optim_iter,
            },
        )

        return phi_[0], alpha[0], beta[0]
