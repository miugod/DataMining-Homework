import numpy as np

from sklearn import datasets, cluster, metrics
from sklearn.decomposition import PCA

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]

news = datasets.fetch_20newsgroups(subset='all', categories=categories,
                                   shuffle=True, random_state=42)

vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000,
                                 min_df=2, stop_words='english',
                                 use_idf=True)
data = vectorizer.fit_transform(news.data)
svd = TruncatedSVD(2)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)
data = lsa.fit_transform(data)
x_compress = PCA(n_components=2).fit_transform(data)
y = news.target
#true_k = np.unique(y_true).shape[0]

print('-' * 50)
print('%-20s\t%-5s\t%-5s\t%-5s' % ('', 'NMI', 'Homo', 'Comp'))

def funct(y, pred, name):
    print('%-20s\t%.3f\t%.3f\t%.3f' % (
          name,
          metrics.v_measure_score(y, pred),
          metrics.homogeneity_score(y, pred),
          metrics.completeness_score(y, pred),
         ))

pred = cluster.KMeans(n_clusters=4, random_state=10).fit_predict(data)
#funct(y, y_pred, 'K-Means')

pred = cluster.AffinityPropagation(damping=0.8, preference=-2000).fit_predict(data)
#funct(y, y_pred, 'AffinityPropagation')

pred = cluster.MeanShift(bandwidth=0.5).fit_predict(data)
funct(y, pred, 'MeanShift')

pred = cluster.SpectralClustering(n_clusters=4).fit_predict(data)
funct(y, pred, 'SpectralClustering')




print('-' * 50)

