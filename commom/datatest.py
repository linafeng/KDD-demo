# -*- coding: utf-8 -*-
"""查看 数据集的特征名称（数据集特征矩阵的
index 名称）"""
#鸢尾花数据集iris
from sklearn.datasets import load_iris
iris = load_iris()
print(iris.feature_names)

#波士顿放假数据集 cart
from sklearn.datasets import load_boston
# 准备数据集
boston = load_boston()
# 探索数据
print(boston.feature_names)


