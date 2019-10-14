from matplotlib import pyplot as plt
from sklearn import cluster
from sklearn import datasets
from sklearn import metrics
from sklearn import mixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

#数据集digits
digits = datasets.load_digits()
#评估函数
def functs(labels_true, labels_pred, name):
    print('%-25s\t%.3f\t%.3f\t%.3f' % (
          name,
          metrics.v_measure_score(labels_true, labels_pred),
          metrics.homogeneity_score(labels_true, labels_pred),
          metrics.completeness_score(labels_true, labels_pred),
         ))

#载入digits数据
data = scale(digits.data)
x_compress = PCA(n_components=2).fit_transform(data)
y_true = digits.target
#表格：nmi, homogeneity, completeness的值
print('-' * 50)
print('digits聚类结果')
print("%-25s\t%-5s\t%-5s\t%-5s" % ('', 'NMI', 'Homo', 'Comp'))
#plt.figure(figsize=(20, 20))

y_pred = cluster.KMeans(n_clusters=24, random_state=4).fit_predict(data)
functs(y_true, y_pred, 'K-Means')
plt.subplot(331).set_title('K-Means')
plt.scatter(x_compress[:, 0], x_compress[:, 1], c=y_pred)

y_pred = cluster.AffinityPropagation(damping=0.85, preference=-1200).fit_predict(data)
functs(y_true, y_pred, 'AffinityPropagation')
plt.subplot(332).set_title('AffinityPropagation')
plt.scatter(x_compress[:, 0], x_compress[:, 1], c=y_pred)

y_pred = cluster.MeanShift(bandwidth=4.5).fit_predict(data)
functs(y_true, y_pred, 'MeanShift')
plt.subplot(333).set_title('MeanShift')
plt.scatter(x_compress[:, 0], x_compress[:, 1], c=y_pred)

y_pred = cluster.SpectralClustering(n_clusters=15,
                                    affinity='nearest_neighbors').fit_predict(data)
functs(y_true, y_pred, 'SpectralClustering')
plt.subplot(334).set_title('SpectralClustering')
plt.scatter(x_compress[:, 0], x_compress[:, 1], c=y_pred)

y_pred = cluster.AgglomerativeClustering(n_clusters=18, linkage='ward').fit_predict(data)
functs(y_true, y_pred, 'Ward')
plt.subplot(335).set_title('Ward')
plt.scatter(x_compress[:, 0], x_compress[:, 1], c=y_pred)

y_pred = cluster.AgglomerativeClustering(n_clusters=30, linkage='average',
                                         affinity='cosine').fit_predict(data)
functs(y_true, y_pred, 'AgglomerativeClustering')
plt.subplot(336).set_title('AgglomerativeClustering')
plt.scatter(x_compress[:, 0], x_compress[:, 1], c=y_pred)

y_pred = cluster.DBSCAN(eps=5, min_samples=15).fit_predict(data)
functs(y_true, y_pred, 'DBSCAN')
plt.subplot(337).set_title('DBSCAN')
plt.scatter(x_compress[:, 0], x_compress[:, 1], c=y_pred)

y_pred = mixture.GaussianMixture(n_components=20).fit_predict(data)
functs(y_true, y_pred, 'GaussianMixture')
plt.subplot(338).set_title('GaussianMixture')
plt.scatter(x_compress[:, 0], x_compress[:, 1], c=y_pred)

plt.subplot(339).set_title('TrueLabel')
plt.scatter(x_compress[:, 0], x_compress[:, 1], c=y_true)

print('-' * 50)
plt.show()

