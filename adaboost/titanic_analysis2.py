# -*- coding: utf-8 -*-
"""Adaboost做预测"""

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import zero_one_loss
import numpy as np

# 数据加载
train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')
# 数据清洗
# 使用平均年龄来填充年龄中的 nan 值
train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
test_data['Age'].fillna(test_data['Age'].mean(), inplace=True)
# 使用票价的均值填充票价中的 nan 值
train_data['Fare'].fillna(train_data['Fare'].mean(), inplace=True)
test_data['Fare'].fillna(test_data['Fare'].mean(), inplace=True)
print(train_data['Embarked'].value_counts())

# 使用登录最多的港口来填充登录港口的 nan 值
train_data['Embarked'].fillna('S', inplace=True)
test_data['Embarked'].fillna('S', inplace=True)

# 特征选择
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
train_features = train_data[features]
train_labels = train_data['Survived']
test_features = test_data[features]

dvec = DictVectorizer(sparse=False)
train_features = dvec.fit_transform(train_features.to_dict(orient='record'))
print(dvec.feature_names_)

# 构造 ID3 决策树
clf = DecisionTreeClassifier()  # (criterion='entropy')
# 决策树训练
clf.fit(train_features, train_labels)
# 得到决策树准确率
acc_decision_tree = round(clf.score(train_features, train_labels), 6)

print(u'cart score 准确率为 %.4lf' % acc_decision_tree)

# 设置 AdaBoost 迭代次数
n_estimators = 300
# 弱分类器
dt_stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
dt_stump.fit(train_features, train_labels)
print(u'决策树弱分类器 score 准确率为 %.4lf' % round(dt_stump.score(train_features, train_labels), 6))
# AdaBoost 分类器
ada = AdaBoostClassifier(base_estimator=dt_stump, n_estimators=n_estimators)
ada.fit(train_features, train_labels)
print(u'AdaBoost score 准确率为 %.4lf' % round(ada.score(train_features, train_labels), 6))
ada_err = np.zeros((n_estimators,))
# 遍历每次迭代的结果 i 为迭代次数, pred_y 为预测结果
# for i, pred_y in enumerate(ada.staged_predict(train_features)):
#     # 统计错误率
#     ada_err[i] = 1-zero_one_loss(pred_y, train_labels)
#
# print(ada_err)
