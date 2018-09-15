#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 22:04:23 2017

@author: Carlos Atico Ariza, PhD
"""

#This module is for using features from the bottleneck and training ML models 
#to predict the type of webpage.
from visualization import plot_confusion_matrix
import pandas as pd
import numpy as np

from itertools import cycle, chain
#from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, StratifiedKFold
#from sklearn.feature_selection import VarianceThreshold, RFECV 
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer, label_binarize
from sklearn.multiclass import OneVsRestClassifier
le = preprocessing.LabelEncoder()
#import datetime as dt
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
#from sklearn.neural_network import MLPClassifier
#from sklearn.svm import SVC
import matplotlib.pyplot as plt
from scipy import interp
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.cluster import KMeans
import pickle
from PCA_scatter_plot import plot_pca

#Importing the bottleneck features for each image
feat_df = pd.read_csv('filtered_bn_feat.csv',index_col= 0)
mask = feat_df.loc[:,'label'].isin(['single_product','market_place'])
feat_df = feat_df.loc[mask,:]
print('Number of bottleneck features:', feat_df.shape[1])
plt.close('all')
y = feat_df.loc[:,'label'].values
#y.replace('unknown', '', inplace=True)

print('Number of samples for each label \n', feat_df.groupby('label')['label'].count())
X = feat_df.loc[:,'x0':].values
#nulls = BN_featues.isnull().any(axis=1) #checking for nulls in DF

class_names = set(y)
# Binarize the output
print(class_names)
lb = label_binarize(y = y, classes = list(class_names))

#classes.remove('unknown')
#lb.fit(y) #for LabelBinarizer not lable_binerize()
#lb.classes_ #for LabelBinarizer not lable_binerize()
#Split the training data for cross validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=0)

#Feature Scaling?
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

#Dimensionality Reduction
#Princple Component Analysis
from sklearn.decomposition import PCA
#Use n_components = None first to see variability. 
#Then change the number of pca that is reasonable 
pca_none = PCA(n_components = None) # kernel = 'rbf')

X_train1 = pca_none.fit_transform(X_train)
X_test1 = pca_none.transform(X_test)
explained_variance = pca_none.explained_variance_ratio_
plt.figure(0)
plt.plot(explained_variance)
plt.xlabel('n_components')
plt.ylabel('variance')
plt.suptitle('Explained Variance of Principle Components')
plt.show()
plt.savefig('..\analysis\pca_var_vs_ncomp.pdf')
#Applying PCA
pca = PCA(n_components = 300, svd_solver='auto') #, kernel = 'rbf') 
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

#Save feature reduction PCA
save_PCA = '../models/trained_PCA.sav'
pickle.dump(pca, open(save_PCA, 'wb'))

#Pairwise plots of 12 PCA, not this only works with two labels
plt.figure(10)
feat_df_ploting =  pd.DataFrame({'label':y_train})
plot_pca(X=X_train[:,:11], enable_plotting=True,labels_DF=feat_df_ploting)

#Multiclass Classfication
#Let's explore use of Naive Bayes classifiers
classifier = OneVsRestClassifier(GaussianNB())
nbclf = classifier.fit(X_train, y_train)
print('NB test score', classifier.score(X_test, y_test))
y_train_predictions_nbclf = nbclf.predict(X_train)
y_test_predictions_nbclf = nbclf.predict(X_test)
y_predict_prob = nbclf.predict_proba(X_test)
cv_scores = cross_val_score(classifier, X, y)
print('cv_scores', cv_scores)
answer = pd.DataFrame(y_predict_prob, columns = class_names).round(decimals=3) # index= pd.DataFrame(X_test).index.tolist())
#print('One vs Rest - Naive Baise\n', answer.head())

# Confusion Matrix for Naive Baise
cmNB = confusion_matrix(y_test, y_test_predictions_nbclf,labels = list(class_names))

#print(y_test == y_test_predictions_nbclf)

# Plot non-normalized confusion matrix
plt.figure(1)
plot_confusion_matrix(cm1=cmNB, classes=class_names, normalize=False,
                      title='One vs Rest - Naive Baise\nConfusion matrix')

plt.show()

#Set Up ROC Plot to help determine parameter values
# Compute ROC curve and ROC area for each class
#y_decision_func = nbclf.decision_function(X_test)

# Binarize the output
#y_test = label_binarize(y_test, classes=list(class_names))
y_test_Series = pd.Series(y_test)
y_test_dummy = pd.get_dummies(y_test_Series).values
#classes =
y_test_predictions_nbclf = label_binarize(y_test_predictions_nbclf, classes=['single_product','market_place'])

fpr = dict()
tpr = dict()
roc_auc = dict()
print(*range(0))
k = range(len(class_names))
    
for i in k:
    print(i)
    fpr[i], tpr[i], _ = roc_curve(y_test_dummy[:,i], y_predict_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
#    print(_)

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_dummy.ravel(), y_predict_prob.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure(2)
lw = 2
plt.plot(fpr[0], tpr[0], color='darkorange',
         lw=lw, label='ROC curve for \"{0}\" (area = {1:.2f})'.format(list(class_names)[0], roc_auc[0]))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

# First aggregate all false positive rates
n_classes = len(class_names)
all_fpr = np.unique(np.concatenate([fpr[i] for i in k]))

# Then interpolate all ROC curves at this point
mean_tpr = np.zeros_like(all_fpr)
for i in k:
    #print(i)
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure(3)
plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])

for i, color in zip(k, colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve for \"{0}\" (area = {1:0.2f})'
             ''.format(list(class_names)[i], roc_auc[i]))
    
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()


#Adaptive Boosting Classifier
#http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
AdaBoost = AdaBoostClassifier()
AdaBoost.fit(X_train, y_train)
y_predAB = AdaBoost.predict(X_test)
y_predAB_binarized = label_binarize(y_predAB, 
                                          classes=['single_product','market_place'])
#y_predAB = list(chain(*y_predAB))
y_predAB_binarized = list(chain(*y_predAB_binarized))
print('Adaptive Boosting Classifier mean accuracy:', AdaBoost.score(X_test, y_test))
plt.figure(9)
cmAdaBoost = confusion_matrix(y_test, y_predAB,labels = list(class_names))
plot_confusion_matrix(cm1=cmAdaBoost, classes= class_names,
                      title='AdaBoost\nConfusion matrix')
plt.show()

#GradientBoostingClassifier

#More advanced methods of dimensionality reduction
##Recursive feature elimination 
#rfecv = RFECV(GradientBoostingClassifier(random_state=0), step=1, cv=5,
#              scoring='roc_auc')
#rfecv.fit(X_train)
#print(len(list(rfecv.get_support())),'Number of features')
#y_score_rfecv_GBDT = rfecv.predict_proba(X_test)
#y_predict_rfecv_GBDT = rfecv.predict(X_test)

#Let's try unsupervised learning
#K-Means clustering
#X_train1 = preprocessing.scale(X_train1)
reduced_X_train = PCA(n_components=2).fit_transform(X_train1)
reduced_X_test = PCA(n_components=2).fit_transform(X_test)
kmeans = KMeans(n_clusters=2,random_state=0)
kmeans.fit(reduced_X_train)
# #############################################################################
# Visualize the results on PCA-reduced data for K-Means

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_X_train[:, 0].min() - 1, reduced_X_train[:, 0].max() + 1
y_min, y_max = reduced_X_train[:, 1].min() - 1, reduced_X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)

plt.figure(4)
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_X_train[:, 0], reduced_X_train[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

y_train_fct = pd.factorize(y_train)[0]

print('K-Means\ntest score', kmeans.score(reduced_X_test, y_test_dummy))
print("Homogeneity: %0.3f" % metrics.homogeneity_score(y_train_fct, kmeans.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(y_train_fct, kmeans.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(y_train_fct, kmeans.labels_))
######
#https://stats.stackexchange.com/questions/9850/how-to-plot-data-output-of-clustering#10071

from scipy import cluster
plt.figure(6)
#plot variance for each value for 'k' between 1,10
initial = [cluster.vq.kmeans(reduced_X_train,i) for i in range(1,10)]
plt.plot([var for (cent,var) in initial])
plt.show()

cent, var = initial[n_classes-1]
#use vq() to get as assignment for each obs.
plt.figure(4)
assignment,cdist = cluster.vq.vq(reduced_X_train,cent)
plt.scatter(reduced_X_train[:,0], reduced_X_train[:,1], c=assignment)
plt.show()

#######Random Forest Classification##################
#Next, let's try random forest classifier
n, f= 30, 100
RFclf = OneVsRestClassifier(RandomForestClassifier(n_estimators = n, max_features= f))
RFclf.fit(X_train, y_train)

#print('y_train:',y_train)
#Save trained model for later use
save_RF_fn = '../models/trained_RF.sav'
pickle.dump(RFclf, open(save_RF_fn, 'wb'))

y_test_predictions_RF = RFclf.predict(X_test)
y_score_RF = RFclf.predict_proba(X_test)

y_score_answer_RF = RFclf.predict_proba(X_test)
answer_RF = pd.DataFrame(y_score_answer_RF)
#print('Random Forest\n', answer_RF.head())
print('Random Forest test score:', RFclf.score(X_test, y_test))

cmRF = confusion_matrix(y_test, y_test_predictions_RF,labels = list(class_names))

#print(y_test == y_test_predictions_nbclf)

# Plot non-normalized confusion matrix
plt.figure(8)
plot_confusion_matrix(cm1=cmRF, classes= class_names,
                      title='Random Forests; estimators:{0}, max_features: {1}\nConfusion matrix, without normalization'.format(n,f))

plt.show()
###################################################

