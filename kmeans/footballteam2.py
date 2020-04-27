# -*- coding: utf-8 -*-
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
import numpy as np

"""
如果我们把上面的 20 支亚洲球队用 K-Means 划分成 5 类，
在规范化数据的时候采用标准化的方式（即均值为 0，方差为 1）
"""
# 输入数据
data = pd.read_csv('data.csv', encoding='gbk')
train_x = data[["2019年国际排名", "2018世界杯", "2015亚洲杯"]]
df = pd.DataFrame(train_x)
kmeans = KMeans(n_clusters=5)

# 采用Z-Score规范化数据，保证每个特征维度的数据均值为0，方差为1
ss = preprocessing.StandardScaler()
train_x = ss.fit_transform(train_x)

# kmeans 算法
kmeans.fit(train_x)
predict_y = kmeans.predict(train_x)
# 合并聚类结果，插入到原数据中
result = pd.concat((data, pd.DataFrame(predict_y)), axis=1)
result.rename({0: u'聚类'}, axis=1, inplace=True)
print(result)
