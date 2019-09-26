"""
=============================================
A demo of the mean-shift clustering algorithm
=============================================

Reference:

Dorin Comaniciu and Peter Meer, "Mean Shift: A robust approach toward
feature space analysis". IEEE Transactions on Pattern Analysis and
Machine Intelligence. 2002. pp. 603-619.

"""
from sklearn.decomposition import PCA

print(__doc__)

import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import load_digits

# #############################################################################
# Generate sample data
digits = load_digits()
centers = [[1, 1], [-1, -1], [1, -1]]
X = digits.data
labels_true = digits.target

# #############################################################################
# Compute clustering with MeanShift

# The following bandwidth can be automatically detected using
bandwidtha = estimate_bandwidth(X, quantile=0.5, n_samples=500)

ms = MeanShift(bandwidth=bandwidtha, bin_seeding=True)
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

ms_m = MeanShift(bandwidth=4.8, bin_seeding=True)
ms_m.fit(X)
mlabels = ms_m.labels_
mcluster_centers = ms_m.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)

# #############################################################################
# Plot result
import matplotlib.pyplot as plt

plt.close('all')
plt.figure(1)
plt.clf()
x_compress = PCA(n_components=2).fit_transform(X)
plt.subplot(222).set_title('MeanShiftAuto')
plt.scatter(x_compress[:, 0], x_compress[:, 1], c=labels)
plt.subplot(221).set_title('TrueLabel')
plt.scatter(x_compress[:, 0], x_compress[:, 1], c=labels_true)
plt.subplot(223).set_title('MeanShiftManuel')
plt.scatter(x_compress[:, 0], x_compress[:, 1], c=mlabels)


plt.show()
