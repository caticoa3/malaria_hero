
# coding: utf-8

# # Machine learning with bottleneck features for biological microscopy classification
#
# The goal of this project is to using image (.jpg) analysis to classify cell types. 
# A pre-trained TensorFlow model - Inception Model v3 for TensorFlow - trained to analyze images was used to create features of each webpage. In this notebook are some potential machine learning pipelines that are trained on the bottleneck features, from the TensorFlow model, to classify each webpage.
# Why is this important: this would automate the classification cells eliminating labor intensive classification.
# Let's start comparing three cell types nuerons, oligodendrocytes, and astrocytes.
#
# ## Machine learning algos explored for classification in notebook
# * One vs Rest - Naive Bayes
# * Random Forest
# * Adaptive Boosting

# ### Importing libraries ###

# In[1]:


from visualization import plot_confusion_matrix
import pandas as pd
import numpy as np
import time

from itertools import cycle
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_selection import VarianceThreshold, RFECV
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelBinarizer, label_binarize
from sklearn.multiclass import OneVsRestClassifier
le = preprocessing.LabelEncoder()
import datetime as dt
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from scipy import interp
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.cluster import KMeans
import pickle
from pair_scatter_plots import plot_pca, seaborn_pairwise_plot, caa_plot_pairs

# ### Exploring BN Features ###

# In[2]:

def ML_with_BN_feat(bn_feat_file='../data/factors_n_bn_feat.csv', n_comp=100, 
                    plotting=False):
    if n_comp < 50:
        n_comp = 50
    # Importing the bottleneck features for each image
    feat_df = pd.read_csv(bn_feat_file, index_col=0, dtype='unicode')
#    feat_df = feat_df.iloc[0:300,:]
    mask = feat_df.loc[:, 'label'].isin(['TUJ1', 'RIP', 'GFAP'])
    feat_df = feat_df.loc[mask, :].drop_duplicates()
    print('Number of bottleneck features:', feat_df.shape[1]-7)
    y = feat_df.loc[:,['label','Date','group_idx']].values

    print('Number of samples for each label \n', feat_df.groupby('label')['label'].count())
    X = feat_df.loc[:, 'x0':'x2047'].astype(float).values
#    print(list(feat_df.loc[:, 'x0':].columns))
    
    ##-- Dealing with imbalanced data
    
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=0)
    
    X_resampled, y_resampled = ros.fit_sample(X, y[:,0])
    
    from collections import Counter
    print(sorted(Counter(y_resampled).items()))
    
    X, y = X_resampled, y_resampled
    # checking for nulls in DF
    #nulls = BN_featues.isnull().any(axis=1)
    
    # checking for nulls in DF
    #nulls = BN_featues.isnull().any(axis=1)
    # In[3]:

    class_names = set(feat_df.loc[:,'label'])
    # Binarize the labels
    # print(class_names)
#    lb = label_binarize(y = y, classes = list(class_names))
    # classes.remove('unknown')
    # lb.fit(y) #for LabelBinarizer not lable_binerize()
    # lb.classes_ #for LabelBinarizer not lable_binerize

    # Split the training data for cross validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=0)
   
    ##### Dimensionality Reduction ####

    # In[4]:

    # Princple Component Analysis
    # Use n_components = None first to determine variability of principle components
    # Then limit the number of principle components that are reasonable
    # n_components=None --> min(n observation, n features)
    pca_none = PCA(n_components=None)  
    pca_none.fit_transform(X_train)
#    print(X_test.shape, type(X_test))
#    arr_index = np.where(X_test == '0.1465795w85188675')
#    print('arr_index', arr_index)
#    print('X_test[arr_index]',X_test[arr_index])
    pca_none.transform(X_test)
    explained_variance = pca_none.explained_variance_ratio_
    plt.figure(0)
    plt.plot(explained_variance)
    plt.xlabel('n_components')
    plt.ylabel('variance')
    plt.suptitle('Explained Variance of Principle Components')
    plt.show(block=False)
    plt.savefig('../plots/pca_var_vs_ncomp.png')

    # #### After about 70 components there is very little variance gain  ####
    # Applying Principle Component Decomposition

    # In[5]:


#    n_comp = 11 # the number of Principal Components to project/decompose the data into
    pca = PCA(n_components=n_comp)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    explained_variance1 = pca.explained_variance_ratio_
    plt.figure(10)
    plt.plot(explained_variance1)
    plt.xlabel('n_components')
    plt.ylabel('variance')
    plt.suptitle('Explained Variance of Principle Components')
    plt.show()
    plt.savefig('../plots/pca_var_vs_{}_ncomp.png'.format(n_comp))

    # Save feature reduction PCA
    save_PCA = '../models/trained_PCA.sav'
    pickle.dump(pca, open(save_PCA, 'wb'))

    # In[6]:
    if plotting:
        # Pairwise plots of 11 PCA, note this only works with two labels
        feat_df_ploting = pd.DataFrame({'label': y_train})
        caa_plot_pairs(X_train[:, :11], feat_df_ploting, 'PCA')
        plt.figure(figsize=(16, 24))
        plt.show(block=False)
    
    # In[70]:
    # seaborn plot of PCA
    # need to add columns to pca X_train
    # conver to a dataframe
    #Pairwise plots of 11 components
    pca_DF = pd.DataFrame(X_train[:, :11])
    
    df_y_train = pd.DataFrame(y_train,
                            columns=['label']) #,'Date','group_idx'])
    df_pca_train = pd.concat([df_y_train, pca_DF], axis=1)
#    dates = list(set(df_pca_train['Date']))
    
#    print(list(feat_df.columns))
    feature_names = df_pca_train.columns[3:]
    n_comp = pca_DF.shape[1]
    print('n_comp', n_comp)
    print('feature_names', feature_names)
    print('df_pca_train columns', list(df_pca_train.columns))
    #plot coloring phenotype
#    seaborn_pairwise_plot(df_pca_train, color_index='label',
#                          feature_names=feature_names, n_comp=n_comp)
    #plot coloring experimental condition
#    seaborn_pairwise_plot(df_pca_train, color_index='group_idx',
#                          feature_names=feature_names, n_comp=n_comp)

#    print('dates', dates)
#    for date in dates:
#        try:
#            print(date)
#            df = df_pca_train.loc[df_pca_train['Date']==date,:]
#            seaborn_pairwise_plot(df,color_index='group_idx', 
#                                  feature_names=feature_names,
#                                  n_comp=date)
##            time.sleep(10)
##            plt.show(block=True)
#        except ValueError:
#            print('ValueErorr for', date, 'continuing with next date.')
#            continue

    #Isomap? tsne?
    
    #tsne
    

    # ## Exploring Different Algorithms For Mutliclass Classfication
    # ### OneVsRestClassifier with Naive Baise
    

    
    # In[7]:

    # Set up plot to compare confusion matrices
    fig, axs = plt.subplots(1, 3, sharey=True, figsize=(15, 8.5))

    classifier = OneVsRestClassifier(GaussianNB())
    nbclf = classifier.fit(X_train, df_y_train['label'])
    y_test_predictions_nbclf = nbclf.predict(X_test)
    y_predict_prob = nbclf.predict_proba(X_test)
    cv_scores = cross_val_score(classifier, X, y)
    print('NB cv_scores', cv_scores)
#    answer = pd.DataFrame(y_predict_prob, columns = class_names).round(decimals=3) # index= pd.DataFrame(X_test).index.tolist())
    #print('One vs Rest - Naive Baise\n', answer.head())

    # Confusion Matrix for Naive Baise
    cmNB = confusion_matrix(y_test, y_test_predictions_nbclf, labels=list(class_names))
    plt.subplot(1, 3, 1)
    plot_confusion_matrix(cm1=cmNB, classes=class_names, normalize=False, gradientbar=False,
                          title='One vs Rest - Naive Baise\nConfusion matrix')
    plt.text(0.01, 0, 'NB cv_scores:\n'+ str(cv_scores), ha='left',
             va='bottom', transform= plt.subplot(1,3,1).transAxes)
    # ### Random Forest Classification

    # In[8]:

    # Next, let's try Random Forest Classifier
    if n_comp < 100:
        f = n_comp
    else:
        f=100
    n = 30
    RFclf = OneVsRestClassifier(RandomForestClassifier(n_estimators=n, max_features=f))
    RFclf.fit(X_train, df_y_train['label'])
    y_test_predictions_RF = RFclf.predict(X_test)
#    y_score_RF = RFclf.predict_proba(X_test)
    y_score_answer_RF = RFclf.predict_proba(X_test)
    cv_scores_RF = cross_val_score(RFclf, X, y)
    print('Random Forest cv_scores', cv_scores_RF)
#    answer_RF = pd.DataFrame(y_score_answer_RF)
    save_RF = '../models/trained_RF.sav'
    pickle.dump(RFclf, open(save_RF, 'wb'))
    #print('Random Forest\n', answer_RF.head())

    # confusion matrix
    cmRF = confusion_matrix(y_test, y_test_predictions_RF, labels=list(class_names))
    plt.subplot(1, 3, 2)
    plot_confusion_matrix(cm1=cmRF, classes=class_names, gradientbar=False,
                          title='Random Forests; estimators:{0}, max_features: {1}\nConfusion matrix'.format(n, f))
    plt.text(0.01, 0, 'Random Forest cv_scores:\n'+ str(cv_scores_RF), ha='left',
             va='bottom', transform=plt.subplot(1, 3, 2).transAxes)

    # ### Adaptive Boosting Classifier
    # http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html

    # In[9]:

    AdaBoost = AdaBoostClassifier()
    AdaBoost.fit(X_train, y_train)
    y_predAB = AdaBoost.predict(X_test)
#    y_predAB_binarized = label_binarize(y_predAB,
#                                              classes=['single_product','market_place'])
    cv_scores_AB = cross_val_score(AdaBoost, X, y)
    print('Adaptive Boosting cv_scores', cv_scores_AB)
    save_AdaBoost = '../models/trained_AdaBoost.sav'
    pickle.dump(AdaBoost, open(save_AdaBoost, 'wb'))

    plt.subplot(1, 3, 3)
    cmAdaBoost = confusion_matrix(y_test, y_predAB, labels=list(class_names))
    plot_confusion_matrix(cm1=cmAdaBoost, classes=class_names,
                          title='AdaBoost\nConfusion matrix', gradientbar=False)
    plt.text(0.01, 0, 'Adaptive Boosting cv_scores:\n'+ str(cv_scores_AB), ha='left',
             va='bottom', transform=plt.subplot(1, 3, 3).transAxes)
    plt.savefig('../plots/confusion_matrix_result.png')

    # #### Comparing mean accuracy and confusion matrices of difference classification algorithrms

    # In[10]:
    print('\nOne vs Rest - Naive Baise mean accuracy:', round(classifier.score(X_test, y_test), 4))
    print('Random Forest Classifier mean accuracy:', round(RFclf.score(X_test, y_test), 4))
    print('Adaptive Boosting Classifier mean accuracy:', round(AdaBoost.score(X_test, y_test), 4))
    plt.tight_layout()
    fig.tight_layout()
    print('If launched from command line use ctrl+c to close all plots and finish')
    plt.show()


if __name__ == '__main__':
    import sys
    ML_with_BN_feat(bn_feat_file=sys.argv[1], n_comp=int(sys.argv[2]), plotting=sys.argv[3])
# Command line use:
# python ML_with_BN_features.py ../data/filtered_bn_feat.csv 100 False
