#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 11:46:14 2018

@author: Carlos A Ariza, PhD
"""

#Montage of images
	
# import the necessary packages
from imutils import build_montages
from imutils import paths
import pandas as pd
import cv2

cv_df = pd.read_csv('../data/cv_feat.csv', index_col=0)
mask = (cv_df['blob_detected'] == True) & (cv_df['label'] == 'Parasitized')
imagesFN = cv_df.loc[mask, ['fn']].sample(n = 6).values[:,0]

images = []
for image in imagesFN:
    image = '../datasets/cell_images/Parasitized/' + str(image)
    im = cv2.imread(image)
#    cv2.imshow('cell', im)
#    cv2.waitKey(0)
    images.append(im)
#print(images[:2])
# construct the montages for the images
montages = build_montages(images, (200, 200), (3,2))
for montage in montages:
        cv2.imshow('montage image', montage)
        cv2.waitKey(0)
        cv2.imwrite('../presentations/TP_blob_montage.png', montage)  
cv2.destroyAllWindows()

