import numpy as np
import math


class MySVM:
    def fit(self, features: np.ndarray, labels: np.ndarray, C: float, tol: float, kernel: str):
        self._features = features
        self._labels = labels
        self._C = C
        self._tol = tol
        self._kernel = kernel

        self._smo()
        return self

    def predict(self, feature: np.ndarray) -> float:
        return self._calcU(feature)

    def _kernelCompute(self, x1: np.ndarray, x2: np.ndarray) -> float:
        if self._kernel == 'linear':
            return x1.dot(x2)
        elif self._kernel == 'rbf':
            return math.exp(-np.linalg.norm(x1 - x2) / 200)

    def _calcU(self, feature: np.ndarray) -> float:
        kxx = np.apply_along_axis(lambda it: self._kernelCompute(it, feature), 1, self._features)
        return (self._labels * self._alphas).dot(kxx) - self._b

    def _takeStep(self, i1: int, i2: int) -> int:
        if i1 == i2:
            return 0
        X1 = self._features[i1]
        X2 = self._features[i2]
        y1 = self._labels[i1]
        y2 = self._labels[i2]
        alph1 = self._alphas[i1]
        alph2 = self._alphas[i2]
        E1 = self._calcU(X1)-y1
        E2 = self._calcU(X2)-y2

        L = 0.0
        H = 0.0
        s = y1*y2
        if s == -1.0:
            L = max(0, alph2-alph1)
            H = min(self._C, self._C+alph2-alph1)
        else:
            L = max(0, alph2+alph1-self._C)
            H = min(self._C, alph2+alph1)
        if abs(L-H) < self._eps:
            return 0

        k11 = self._kernelCompute(X1, X1)
        k12 = self._kernelCompute(X1, X2)
        k22 = self._kernelCompute(X2, X2)
        eta = k11+k22-2*k12

        a2 = 0.0
        if eta > 0:
            a2 = alph2+y2*(E1-E2)/eta
            if a2 < L:
                a2 = L
            elif a2 > H:
                a2 = H
        else:
            f1 = y1*(E1+self._b)-alph1*k11-s*alph2*k12
            f2 = y2*(E2+self._b)-s*alph1*k12-alph2*k22
            L1 = alph1 + s * (alph2-L)
            H1 = alph1 + s * (alph2-H)
            PsiL = L1 * f1 + L * f2 + 0.5 * L1 * L1 * k11 + 0.5 * L * L * k22 + s * L * L1 * k12
            PsiH = H1 * f1 + H * f2 + 0.5 * H1 * H1 * k11 + 0.5 * H * H * k22 + s * H * H1 * k12
            if PsiL < PsiH:
                a2 = L
            else:
                a2 = H
        if abs(a2-alph2) < self._eps:
            return 0

        a1 = alph1+s*(alph2-a2)

        # store alpha
        self._alphas[i1] = a1
        self._alphas[i2] = a2

        b1 = E1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12 + self._b
        b2 = E2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22 + self._b
        self._b = (b1+b2)/2
        return 1

    def _examineExample(self, i2: int):
        X2 = self._features[i2]
        y2 = self._labels[i2]
        alpha2 = self._alphas[i2]
        E2 = self._calcU(X2) - y2
        r2 = E2*y2
        if (r2 < -self._tol and alpha2 < self._C) or (r2 > self._tol and alpha2 > 0.0):
            i1 = -1
            deltaE = 0.0
            for i in range(self._alphas.size):
                yTemp = self._alphas[i]
                alphaTemp = self._alphas[i]
                if alphaTemp > 0.0 and alphaTemp < self._C:
                    ETemp = self._calcU(X2) - yTemp
                    deltaETemp = abs(ETemp - E2)
                    if deltaETemp > deltaE:
                        i1 = i
                        deltaE = deltaETemp
            if i1 != -1:
                if self._takeStep(i1, i2) == 1:
                    return 1
            for i in range(self._alphas.size):
                if self._takeStep(i1, i2) == 1:
                    return 1
        return 0

    def _smo(self):
        # initialize
        self._eps = 1e-6
        self._b = 0.0
        self._alphas = np.zeros(self._labels.size)

        # main routine
        numChanged = 0
        examineAll = 1
        while numChanged > 0 or examineAll == 1:
            numChanged = 0
            if examineAll == 1:
                for i in range(self._alphas.size):
                    numChanged += self._examineExample(i)
            else:
                for i in range(self._alphas.size):
                    alphaI = self._alphas[i]
                    if alphaI > 0 and alphaI < self._C:
                        numChanged += self._examineExample(i)

            if examineAll == 1:
                examineAll = 0
            elif numChanged == 0:
                examineAll = 1
