# 实验报告

## 实验题目
使用sklearn的各种聚类算法，对20newsgroup和digits数据集进行聚类
## 实验目的
1.对聚类算法进行实践，提高应用能力
2.通过调整参数和运行程序，进一步理解各个算法的原理
## 实验环境
python3.7
## 实验内容
    应用聚类算法
K-Means
Affinity propagation
Mean-shift
Spectral clustering
Ward hierarchical clustering
Agglomerative clustering
DBSCAN
Gaussian mixtures
    使用
NMI
Homogeneity
Completeness
    为评估标准对于
digits
20newsgroup
    数据集进行评估：


## 实验结果
--------------------------------------------------
digits聚类结果
                         	NMI  	Homo 	Comp
K-Means                  	0.721	0.802	0.656
AffinityPropagation      	0.684	0.751	0.627
MeanShift                	0.658	0.954	0.502
SpectralClustering       	0.845	0.901	0.796
Ward                     	0.804	0.827	0.782
AgglomerativeClustering  	0.782	0.887	0.698
DBSCAN                   	0.446	0.354	0.602
GaussianMixture          	0.692	0.742	0.647
--------------------------------------------------

--------------------------------------------------
20newsgroup聚类结果
                    	NMI  	Homo 	Comp
K-Means             	0.444	0.442	0.446
AffinityPropagation 	0.289	1.000	0.169
MeanShift           	0.514	0.385	0.773
SpectralClustering  	0.480	0.444	0.523
Ward                	0.447	0.442	0.452
AgglomerativeClustering	0.460	0.426	0.499
DBSCAN              	0.405	0.436	0.378
GaussianMixture     	0.434	0.433	0.434
--------------------------------------------------
##结果分析
总体而言，由于digits数据集比较简单，所以digits聚类的结果比20newsgroup的聚类结果要好。
经过调参后，meanshift，spectralclustering和AgglomerativeClustering的结果较好，而AffinityPropagation和DBSCAN的结果不太好，且对于参数的依赖性较大。
在运行时，AffinityPropagation和AgglomerativeClustering的运行时间较之其他方法更长，因为他们的时间复杂度较大。