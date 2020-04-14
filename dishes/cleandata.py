# -*- coding: utf-8 -*-
"""
原始数据
food	ounces	animal
bacon	4.0	pig
pulled pork	3.0	pig
bacon	NAN	pig
Pastrami	6.0	cow
corned beef	7.5	cow
Bacon	8.0	pig
pastrami	-3.0	cow
honey ham	5.0	pig
nova lox	6.0	salmon

"""
"""
问题阐述

1.大小写转换
2.空值
3.不合法数据 如负数
"""
import pandas as pd
import numpy as np
from pandas import Series, DataFrame

# 导入导出数据
print("导入导出数据")
score = DataFrame(pd.read_excel('data.xlsx'))
print(score)
"""
food名称全部转小写
"""
score['food'] = score['food'].str.lower()

"""
空值检索
print(score.isnull())
print(score.isnull().any())
"""


def double_df(x):
    if x == 'NAN':
        return None;
    return x;
score['ounces'] = score['ounces'].apply(double_df)
score['ounces']=score['ounces'].astype(np.float64)
#找出负数
print("删除负数")
score = score.drop(score[score.ounces < 0].index)
#空值替换
print("空值替换")
print(score['ounces'].mean())
print(score)
score['ounces'].fillna(score['ounces'].mean(), inplace=True)
print(score)


score.to_excel('gooddata.xlsx')

