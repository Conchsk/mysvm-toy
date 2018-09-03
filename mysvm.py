import numpy as np
import math


class MySVM:
    def fit(self, features: np.ndarray, labels: np.ndarray, C: float, tol: float, kernel: str, gamma: float = 1.0,
            max_iter: int = 1000):
        self._features = features
        self._labels = labels
        self._C = C
        self._tol = tol
        self._kernel = kernel
        self._gamma = gamma
        self._max_iter = max_iter

        self._smo()
        return self

    def predict(self, feature: np.ndarray) -> float:
        return self._calc_u2(feature)

    def _kernel_compute(self, x1: np.ndarray, x2: np.ndarray) -> float:
        if self._kernel == 'linear':
            return x1.dot(x2)
        elif self._kernel == 'poly':
            return pow(1 + x1.dot(x2), self._gamma)
        elif self._kernel == 'rbf':
            return math.exp(-self._gamma * np.linalg.norm(x1 - x2))

    def _calc_u1(self, i: int) -> float:
        return (self._labels * self._alphas).dot(self._kxx[i]) - self._b

    def _calc_u2(self, feature: np.ndarray) -> float:
        kxx = np.apply_along_axis(lambda it: self._kernel_compute(
            it, feature), axis=1, arr=self._features)
        return (self._labels * self._alphas).dot(kxx) - self._b

    def _take_step(self, i1: int, i2: int) -> int:
        if i1 == i2:
            return 0
        X1 = self._features[i1]
        X2 = self._features[i2]
        y1 = self._labels[i1]
        y2 = self._labels[i2]
        alph1 = self._alphas[i1]
        alph2 = self._alphas[i2]
        E1 = self._calc_u1(i1) - y1
        E2 = self._calc_u1(i2) - y2

        L = 0.0
        H = 0.0
        s = y1 * y2
        if s == -1.0:
            L = max(0, alph2 - alph1)
            H = min(self._C, self._C + alph2 - alph1)
        else:
            L = max(0, alph2 + alph1 - self._C)
            H = min(self._C, alph2 + alph1)
        if abs(L - H) < self._eps:
            return 0

        k11 = self._kxx[i1, i1]
        k12 = self._kxx[i1, i2]
        k22 = self._kxx[i2, i2]
        eta = k11 + k22 - 2 * k12

        a2 = 0.0
        if eta > 0:
            a2 = alph2 + y2 * (E1 - E2) / eta
            if a2 < L:
                a2 = L
            elif a2 > H:
                a2 = H
        else:
            f1 = y1 * (E1 + self._b) - alph1 * k11 - s * alph2 * k12
            f2 = y2 * (E2 + self._b) - s * alph1 * k12 - alph2 * k22
            L1 = alph1 + s * (alph2 - L)
            H1 = alph1 + s * (alph2 - H)
            Psi_L = L1 * f1 + L * f2 + 0.5 * L1 * L1 * \
                k11 + 0.5 * L * L * k22 + s * L * L1 * k12
            Psi_H = H1 * f1 + H * f2 + 0.5 * H1 * H1 * \
                k11 + 0.5 * H * H * k22 + s * H * H1 * k12
            if Psi_L < Psi_H:
                a2 = L
            else:
                a2 = H
        if abs(a2 - alph2) < self._eps:
            return 0

        a1 = alph1 + s * (alph2 - a2)

        # store alpha
        self._alphas[i1] = a1
        self._alphas[i2] = a2

        b1 = E1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12 + self._b
        b2 = E2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22 + self._b
        self._b = (b1 + b2) / 2
        return 1

    def _examine_example(self, i2: int):
        X2 = self._features[i2]
        y2 = self._labels[i2]
        alpha2 = self._alphas[i2]
        E2 = self._calc_u1(i2) - y2
        r2 = E2 * y2
        if (r2 < -self._tol and alpha2 < self._C) or (r2 > self._tol and alpha2 > 0.0):
            i1 = -1
            delta_E = 0.0
            for i in range(self._alphas.size):
                y_temp = self._alphas[i]
                alpha_temp = self._alphas[i]
                if 0.0 < alpha_temp < self._C:
                    E_temp = self._calc_u1(i2) - y_temp
                    delta_E_temp = abs(E_temp - E2)
                    if delta_E_temp > delta_E:
                        i1 = i
                        delta_E = delta_E_temp
            if i1 != -1:
                if self._take_step(i1, i2) == 1:
                    return 1
            for i in range(self._alphas.size):
                if self._take_step(i1, i2) == 1:
                    return 1
        return 0

    def _smo(self):
        # initialize
        self._eps = 1e-6
        self._b = 0.0
        self._alphas = np.zeros(self._labels.size, dtype=float)
        self._kxx = np.zeros((self._labels.size, self._labels.size), dtype=float)

        for i in range(self._labels.size):
            for j in range(i, self._labels.size):
                self._kxx[i, j] = self._kxx[j, i] = self._kernel_compute(
                    self._features[i], self._features[j])

        # main routine
        num_changed = 0
        examine_all = 1
        iter_time = 0
        while num_changed > 0 or examine_all == 1:
            num_changed = 0
            if examine_all == 1:
                for i in range(self._alphas.size):
                    num_changed += self._examine_example(i)
            else:
                for i in range(self._alphas.size):
                    alpha_i = self._alphas[i]
                    if 0 < alpha_i < self._C:
                        num_changed += self._examine_example(i)

            if examine_all == 1:
                examine_all = 0
            elif num_changed == 0:
                examine_all = 1
            iter_time += 1
            print(iter_time)
            if iter_time == self._max_iter:
                break
