y_pred = cluster.KMeans(n_clusters=true_k, random_state=9).fit_predict(X)
evaluate(y_true, y_pred, 'K-Means')
plt.subplot(331).set_title('K-Means')
plt.scatter(x_compress[:, 0], x_compress[:, 1], c=y_pred)

y_pred = cluster.AffinityPropagation(damping=0.88, preference=-3000).fit_predict(X)
evaluate(y_true, y_pred, 'AffinityPropagation')
plt.subplot(332).set_title('AffinityPropagation')
plt.scatter(x_compress[:, 0], x_compress[:, 1], c=y_pred)

y_pred = cluster.MeanShift(bandwidth=0.0001, bin_seeding=True).fit_predict(X)
evaluate(y_true, y_pred, 'MeanShift')
plt.subplot(333).set_title('MeanShift')
plt.scatter(x_compress[:, 0], x_compress[:, 1], c=y_pred)

y_pred = cluster.SpectralClustering(n_clusters=true_k).fit_predict(X)
evaluate(y_true, y_pred, 'SpectralClustering')
plt.subplot(334).set_title('SpectralClustering')
plt.scatter(x_compress[:, 0], x_compress[:, 1], c=y_pred)

y_pred = cluster.AgglomerativeClustering(n_clusters=true_k,
                                         linkage='ward').fit_predict(X)
evaluate(y_true, y_pred, 'Ward')
plt.subplot(335).set_title('Ward')
plt.scatter(x_compress[:, 0], x_compress[:, 1], c=y_pred)

y_pred = cluster.AgglomerativeClustering(n_clusters=true_k,
                                         linkage='average',
                                         affinity='manhattan').fit_predict(X)
evaluate(y_true, y_pred, 'AgglomerativeClustering')
plt.subplot(336).set_title('AgglomerativeClustering')
plt.scatter(x_compress[:, 0], x_compress[:, 1], c=y_pred)

y_pred = cluster.DBSCAN(eps=0.004, min_samples=1).fit_predict(X)
evaluate(y_true, y_pred, 'DBSCAN')
plt.subplot(337).set_title('DBSCAN')
plt.scatter(x_compress[:, 0], x_compress[:, 1], c=y_pred)

y_pred = mixture.GaussianMixture(n_components=true_k).fit_predict(X)
evaluate(y_true, y_pred, 'GaussianMixture')
plt.subplot(338).set_title('GaussianMixture')
plt.scatter(x_compress[:, 0], x_compress[:, 1], c=y_pred)

print('-' * 50)
