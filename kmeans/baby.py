# -*- coding: utf-8 -*-
"""
一张 baby.jpg 的图片，请你编写代码用 K-Means 聚类方法将它分割成 16 个部分。
pip3 install pillow
pip3 install scikit-image
"""

# 使用K-means对图像进行聚类，显示分割标识的可视化
import numpy as np
import PIL.Image as image
from sklearn.cluster import KMeans
from sklearn import preprocessing
from skimage import color
from kmeans.imagehelper import load_data

# 加载图像，得到规范化的结果img，以及图像尺寸
img, width, height = load_data('./baby.jpg')

# 用K-Means对图像进行16聚类
kmeans = KMeans(n_clusters=16)
kmeans.fit(img)
label = kmeans.predict(img)
# 将图像聚类结果，转化成图像尺寸的矩阵
label = label.reshape([width, height])
# 将聚类标识矩阵转化为不同颜色的矩阵
label_color = (color.label2rgb(label) * 255).astype(np.uint8)
label_color = label_color.transpose(1, 0, 2)
images = image.fromarray(label_color)
images.save('baby_mark_color.jpg')
