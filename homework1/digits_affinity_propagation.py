from matplotlib import colors
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.datasets import load_digits
# #############################################################################
# Generate sample data
from sklearn.decomposition import PCA

digits = load_digits()
centers = [[1, 1], [-1, -1], [1, -1], [-1, 1]]
X = digits.data
labels_true = digits.target

# #############################################################################
# Compute Affinity Propagation
af = AffinityPropagation(preference=-5000).fit(X)
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_

n_clusters_ = len(cluster_centers_indices)

print('Estimated number of clusters: %d' % n_clusters_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels,
                                           average_method='arithmetic'))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels, metric='sqeuclidean'))

# #############################################################################
# Plot result
import matplotlib.pyplot as plt
from itertools import cycle

plt.close('all')
plt.figure(1)
plt.clf()
x_compress = PCA(n_components=2).fit_transform(X)
plt.subplot(222).set_title('AffinityPropagation')
plt.scatter(x_compress[:, 0], x_compress[:, 1], c=labels)
plt.subplot(221).set_title('TrueLabel')
plt.scatter(x_compress[:, 0], x_compress[:, 1], c=labels_true)

#plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

