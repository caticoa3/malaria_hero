#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 18:16:05 2017

@author: Carlos Atico Ariza, PhD
"""

import matplotlib.pyplot as plt
import numpy as np
import itertools

def plot_confusion_matrix(cm1, classes, normalize=False, 
                          title='Confusion matrix', cmap=plt.cm.Blues, 
                          gradientbar=False, font={'size':12}):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm1 = cm1.astype('float') / cm1.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
#    else:
#        pass
#        print('Confusion matrix, without normalization')

#    print(cm1)
    plt.imshow(cm1, interpolation='nearest', cmap=cmap)
    plt.title(title, )
    if gradientbar:
        plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes) #rotation=45
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm1.max() / 2.
    for i, j in itertools.product(range(cm1.shape[0]), range(cm1.shape[1])):
        plt.text(j, i, format(cm1[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm1[i, j] > thresh else "black", fontdict = font)

#    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')