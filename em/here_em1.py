# -*- coding: utf-8 -*-
"""
用全部的特征值矩阵进行聚类训练
"""
import pandas as pd
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# 数据加载，避免中文乱码问题
data_ori = pd.read_csv('./heros.csv', encoding='gb18030')
features = [u'最大生命', u'生命成长', u'初始生命', u'最大法力', u'法力成长', u'初始法力', u'最高物攻', u'物攻成长', u'初始物攻', u'最大物防', u'物防成长', u'初始物防',
            u'最大每5秒回血', u'每5秒回血成长', u'初始每5秒回血', u'最大每5秒回蓝', u'每5秒回蓝成长', u'初始每5秒回蓝', u'最大攻速', u'攻击范围']
data = data_ori[features]


data.loc[:, u'最大攻速'] = data.loc[:, (u'最大攻速')].apply(lambda x: float(x.strip('%')) / 100)
# data.loc[:,(u'最大攻速')].apply(lambda x: float(x.strip('%')) / 100)
data[u'攻击范围'] = data[u'攻击范围'].map({'远程': 1, '近战': 0})
print(data[u'攻击范围'])
# 采用 Z-Score 规范化数据，保证每个特征维度的数据均值为 0，方差为 1
ss = StandardScaler()
data = ss.fit_transform(data)
# 构造 GMM 聚类
# n_components为分组数
gmm = GaussianMixture(n_components=3, covariance_type='full')
gmm.fit(data)
# 训练数据
prediction = gmm.predict(data)
print(prediction)

from sklearn.metrics import calinski_harabaz_score
print('3组的分数')
print(calinski_harabaz_score(data, prediction))


gmm = GaussianMixture(n_components=30, covariance_type='full')
gmm.fit(data)
# 训练数据
prediction = gmm.predict(data)
print(prediction)
#指标分数越高，代表聚类效果越好，也就是相同类中的差异性小，不同类之间的差异性大
from sklearn.metrics import calinski_harabaz_score
print('30组的分数')
print(calinski_harabaz_score(data, prediction))



