# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 13:59:11 2020

@author: shubham
"""
#Image segmentation of images using DBSCAN algorithm
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

image = cv.imread(r"C:\Users\shubh\Desktop\PMRO\Misc\ImageSegmentation\cars.png")
labImg = cv.cvtColor(image, cv.COLOR_RGB2LAB)
labImg = cv.medianBlur(labImg,3)

#Downsample image
n = 1
while n <= 2:
    labImg = cv.pyrDown(labImg)
    n = n+1

#Squash image feature vector, 3 channels
featureImg = np.reshape(labImg, ([-1, 3]))
row, col, ch = labImg.shape

#flover image
#db = DBSCAN(eps = 5, min_samples = 10, metric = 'euclidean', algorithm = 'auto')
#Highway image
#db = DBSCAN(eps = 5, min_samples = 5, metric = 'euclidean', algorithm = 'brute')
#cars image
db = DBSCAN(eps = 5, min_samples = 10, metric = 'euclidean', algorithm = 'auto')

db.fit(featureImg)
labels = db.labels_
components = db.components_

indices = np.dstack(np.indices(labImg.shape[:2]))
xyColors = np.concatenate((labImg, indices), axis = -1)
featureImage2 = np.reshape(xyColors, ([-1, 5]));
db.fit(featureImage2)
labels2 = db.labels_
components2 = db.components_

figureSize = 10
plt.plot(figsize = (figureSize, figureSize))
plt.subplot(1,2,1)
plt.imshow(image)
plt.subplot(1,2,2)
plt.imshow(np.reshape(labels2, [row, col]), cmap = 'tab20b')
plt.show()
