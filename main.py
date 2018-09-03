import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.svm import SVC

import mysvm

mysvm_plt = plt.subplot(1, 2, 1)
sksvm_plt = plt.subplot(1, 2, 2)
cm = plt.cm.RdBu
dist = 'circle'

if dist == 'linear':
    X, y = make_classification(n_features=2, n_redundant=0,
                               n_clusters_per_class=1, random_state=1)
    X += 2 * np.random.RandomState(2).uniform(X.size)
elif dist == 'moon':
    X, y = make_moons(noise=0.2, random_state=0)
else:
    X, y = make_circles(noise=0.2, factor=0.5, random_state=1)

yt = y.copy()
yt[yt == 0] = -1
xx, yy = np.meshgrid(np.arange(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 0.1),
                     np.arange(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 0.1))
# mysvm
mysvm_start_time = time.time()
mysvm_model = mysvm.MySVM().fit(X, yt, 1, 1e-3, 'rbf', gamma=2)
mysvm_Z = np.apply_along_axis(lambda it: mysvm_model.predict(it),
                              1, np.c_[xx.ravel(), yy.ravel()])
mysvm_stop_time = time.time()
print(mysvm_stop_time - mysvm_start_time)
mysvm_Z = mysvm_Z.reshape(xx.shape)
mysvm_plt.contourf(xx, yy, mysvm_Z, cmap=cm, alpha=0.8)
mysvm_plt.scatter(X[:, 0], X[:, 1], c=y)

# sksvm
sksvm_start_time = time.time()
sksvm_model = SVC(gamma=2, C=1).fit(X, y)
sksvm_Z = sksvm_model.decision_function(np.c_[xx.ravel(), yy.ravel()])
sksvm_Z = sksvm_Z.reshape(xx.shape)
sksvm_stop_time = time.time()
print(sksvm_stop_time - sksvm_start_time)
sksvm_plt.contourf(xx, yy, sksvm_Z, cmap=cm, alpha=0.8)
sksvm_plt.scatter(X[:, 0], X[:, 1], c=y)

plt.show()
