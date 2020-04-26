# -*- coding: utf-8 -*-
"""
Seaborn 数据集中自带了 car_crashes 数据集，
这是一个国外车祸的数据集，你要如何对这个数据集进行成对关系的探索呢？第二个问题就是，
请你用 Seaborn 画二元变量分布图，
如果想要画散点图，核密度图，Hexbin 图，函数该怎样写？
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#加ssl是因为github的安全验证
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

car_crashes = sns.load_dataset("car_crashes")
# 用 Seaborn 画成对关系
sns.pairplot(car_crashes)
plt.show()

# 用 Seaborn 画二元变量分布图（散点图，核密度图，Hexbin 图）
sns.jointplot(x="speeding", y="alcohol", data=car_crashes, kind='scatter')
sns.jointplot(x="speeding", y="alcohol", data=car_crashes, kind='kde')
sns.jointplot(x="speeding", y="alcohol", data=car_crashes, kind='hex')
plt.show()