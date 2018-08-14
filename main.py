import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification, make_moons, make_circles

import mysvm

X, y = make_circles(noise=0.1, random_state=0, factor=0.5)
yt = y.copy()
yt[yt == 0] = -1
model = mysvm.MySVM().fit(X, yt, 10.0, 1e-3, 'rbf')
print(model._alphas)
xx, yy = np.meshgrid(np.arange(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 0.1),
                     np.arange(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 0.1))
Z = np.apply_along_axis(lambda it: model.predict(it), 1, np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()
