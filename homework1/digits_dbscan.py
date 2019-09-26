# -*- coding: utf-8 -*-
"""
===================================
Demo of DBSCAN clustering algorithm
===================================

Finds core samples of high density and expands clusters from them.

"""
from sklearn.decomposition import PCA

print(__doc__)

import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler


# #############################################################################
# Generate sample data
digits = load_digits()
centers = [[1, 1], [-1, -1], [1, -1]]
X = digits.data
labels_true = digits.target
X = StandardScaler().fit_transform(X)

# #############################################################################
# Compute DBSCAN
db = DBSCAN(eps=5, min_samples=15).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels,
                                           average_method='arithmetic'))

# #############################################################################
# Plot result
import matplotlib.pyplot as plt

plt.close('all')
plt.figure(1)
plt.clf()
x_compress = PCA(n_components=2).fit_transform(X)
plt.subplot(222).set_title('DBSCAN')
plt.scatter(x_compress[:, 0], x_compress[:, 1], c=labels)
plt.subplot(221).set_title('TrueLabel')
plt.scatter(x_compress[:, 0], x_compress[:, 1], c=labels_true)

plt.show()
