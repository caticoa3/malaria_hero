
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

from scipy import interp
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
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_fscore_support
from sklearn.cluster import KMeans
import pickle
from pair_scatter_plots import plot_pca, seaborn_pairwise_plot, caa_plot_pairs

# ### Exploring BN Features ###

# In[2]:

def ML_with_BN_feat(bn_feat_file='../data/factors_n_bn_feat.csv', n_comp=100, 
                    plotting=False):
    plt.close('all')
    if n_comp < 50:
        n_comp = 50
    # Importing the bottleneck features for each image
    feat_df = pd.read_csv(bn_feat_file, index_col=0, dtype='unicode')
#    feat_df = feat_df.sample(frac=0.05)
    print('Data frame shape:', feat_df.shape)
#    feat_df = feat_df.iloc[0:300,:]
    mask = feat_df.loc[:, 'label'].isin(['Parasitized', 'Uninfected'])
    feat_df = feat_df.loc[mask, :].drop_duplicates()
    print('Number of bottleneck features:', feat_df.shape[1]-7)
    y = feat_df.loc[:,['label']].values
    print(type(y), y.shape)

    print('Number of samples for each label \n', feat_df.groupby('label')['label'].count())
    X = feat_df.loc[:, 'x0':'x2047'].astype(float).values
#    print(list(feat_df.loc[:, 'x0':].columns))
    
    ##-- Dealing with imbalanced data
    
#    from imblearn.over_sampling import RandomOverSampler
#    ros = RandomOverSampler(random_state=0)
#    
#    X_resampled, y_resampled = ros.fit_sample(X, y[:,0])
#    
#    from collections import Counter
#    print(sorted(Counter(y_resampled).items()))
#    
#    X, y = X_resampled, y_resampled
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
    print('...running PCA analysis...''')
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
#    plt.show(block=False)
    plt.savefig('../plots/pca_var_vs_ncomp.png')
    # #### After about 70 components there is very little variance gain  ####
    # Applying Principle Component Decomposition

    # In[5]:


#    n_comp = 11 # the number of Principal Components to project/decompose the data into
    print('...running PCA with', n_comp, 'components')
    pca = PCA(n_components=n_comp)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    explained_variance1 = pca.explained_variance_ratio_
    plt.figure(1)
    plt.plot(explained_variance1)
    plt.xlabel('n_components')
    plt.ylabel('variance')
    plt.suptitle('Explained Variance of Principle Components')
    plt.show(block=False)
    plt.savefig('../plots/pca_var_vs_{}_ncomp.png'.format(n_comp))
    # Save feature reduction PCA
    save_PCA = '../models/trained_PCA.sav'
    pickle.dump(pca, open(save_PCA, 'wb'))

    # In[6]:
    if plotting:
        # Pairwise plots of 11 PCA, note this only works with two labels
        feat_df_ploting = pd.DataFrame({'label': y_train[:,0]})
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
    feature_names = df_pca_train.columns[1:]
    n_comp_pca = pca_DF.shape[1]
    print('n_comp_pca', n_comp_pca)
    print('feature_names', feature_names)
    print('df_pca_train columns', list(df_pca_train.columns))

    plt.close('all')
    
    # Set up plot to compare confusion matrices
    params = {'axes.titlesize': 'x-large',
#            'legend.fontsize': 'large',
#          'figure.figsize': (15, 5),
         'axes.labelsize': 'large',
         'axes.titlesize':'large',
         'xtick.labelsize':'medium',
         'ytick.labelsize':'medium'}
    plt.rcParams.update(params)
    
    fig, axs = plt.subplots(1, 4, sharey=True, figsize=(15, 8.5))
    font = {'linespacing':1.5, #'family': 'serif', 'color':  'darkred', 'weight': 'normal',
            'size': 14}
     
    # ## Exploring Different Algorithms For Mutliclass Classfication 
    
    #Metric in this case is F2 
    from sklearn.metrics import fbeta_score, make_scorer
    ftwo_scorer = make_scorer(fbeta_score, beta=2)
    # In[7.5]:
    # Let's scale the features and plug into logisitc regression classifier
#    from sklearn.preprocessing import StandardScaler
#    X_scaled = StandardScaler().fit_transform(X_train)
    
    from sklearn import linear_model
    log_reg_classifier = linear_model.LogisticRegression(penalty='l2', tol=0.0001, C=1.0, 
                       fit_intercept=True, intercept_scaling=1, 
                       class_weight=None, random_state=None,
                       solver='liblinear', max_iter=100, multi_class='ovr', n_jobs=1)
    log_r = log_reg_classifier.fit(X_train, df_y_train['label'].values)
    
    
    y_test_predictions_log_r = log_r.predict(X_test)
    y_predict_prob_log_r = log_r.predict_proba(X_test)
    # save results into a DF
    results = pd.DataFrame()
    results['y_test'] = y_test[:, 0]
    results['log_r_pred'] = list(y_test_predictions_log_r)
    results['log_r_prob'] = y_predict_prob_log_r[:, 0]
    
    #Perform 3-fold cross validation and return the mean accuracy on each fold    
    cv_scores_lr = cross_val_score(estimator=log_r, X= X_train, y= y_train) #, scoring = ftwo_scorer)
    print('Logistic regression cv_scores', cv_scores_lr)
    
    save_LR = '../models/trained_log_reg.sav'
    pickle.dump(log_reg_classifier, open(save_LR, 'wb'))
    
    # Confusion Matrix for Logistic Regresssion
    cmNB = confusion_matrix(y_test, y_test_predictions_log_r, labels=list(class_names))
    plt.subplot(1, 4, 1);
    plot_confusion_matrix(cm1=cmNB, classes=class_names, normalize=True, gradientbar=False,
                          title='Logistic Regression\n')
    cv_scores_lr = ["{:.2f}".format(x) for x in cv_scores_lr]
        
    p_r_fscore_lr = precision_recall_fscore_support(y_test, y_test_predictions_log_r, 
                                    beta=2.0, labels=['Parasitized'], pos_label='Parasitized',
                                    average='binary')

    print(p_r_fscore_lr[:3])
    plt.text(0.01, -1,'\nCV Scores:\n'+ str(cv_scores_lr) + '\n' + 
             'Precision: {d[0]:.2f}\nRecall: {d[1]:.2f} \nF2 score: {d[2]:.2f} \n'.format(
            d = p_r_fscore_lr[:3]), ha='left', va='bottom', fontdict = font,
            transform= plt.subplot(1,4,1).transAxes)
    
    # In[7]:
    
    # ### OneVsRestClassifier with Naive Bayes    

    classifier = OneVsRestClassifier(GaussianNB())
    nbclf = classifier.fit(X_train, df_y_train['label'].values)
    y_test_predictions_nbclf = nbclf.predict(X_test)
    y_predict_prob = nbclf.predict_proba(X_test)
    # save results into a DF
    results['NB_pred'] = list(y_test_predictions_nbclf)
    results['NB_r_prob'] = y_predict_prob[:, 0]
    
    #Perform 3-fold cross validation and return the mean accuracy on each fold    
    cv_scores = cross_val_score(classifier, X_train, y_train) #default 3-fold cross validation
    print('NB cv_scores', cv_scores) 
#    answer = pd.DataFrame(y_predict_prob, columns = class_names).round(decimals=3) # index= pd.DataFrame(X_test).index.tolist())
    #print('One vs Rest - Naive Bayes\n', answer.head())

    # Confusion Matrix for Naive Bayes
    cmNB = confusion_matrix(y_test, y_test_predictions_nbclf, labels=list(class_names))
    plt.subplot(1, 4, 2);
    plot_confusion_matrix(cm1=cmNB, classes=class_names, normalize=True, gradientbar=False,
                          title='One vs Rest - Naive Bayes\n')
    cv_scores = ["{:.2f}".format(x) for x in cv_scores]
    
    p_r_fscore_NB = precision_recall_fscore_support(y_test, y_test_predictions_nbclf, 
                                    beta=2.0, labels=['Parasitized'], pos_label='Parasitized',
                                    average='binary')
    print(p_r_fscore_NB[:3])
    plt.text(0.01, -1,'\nCV Scores:\n'+ str(cv_scores) + '\n' + 
             'Precision: {d[0]:.2f}\nRecall: {d[1]:.2f} \nF2 score: {d[2]:.2f} \n'.format(
            d = p_r_fscore_NB[:3]), ha='left', va='bottom', fontdict = font,
            transform= plt.subplot(1,4,2).transAxes)
    

    # ### Random Forest Classification

    # In[8]:

    # Next, let's try Random Forest Classifier
    if n_comp < 100:
        f = n_comp
    else:
        f=100
    n = 30
    RFclf = OneVsRestClassifier(RandomForestClassifier(n_estimators=n, max_features=f))
    RFclf.fit(X_train, df_y_train['label'].values)
    y_test_predictions_RF = RFclf.predict(X_test)
#    y_score_RF = RFclf.predict_proba(X_test)
    y_score_answer_RF = RFclf.predict_proba(X_test)

    # save results into a DF
    results['RF'] = list(y_test_predictions_RF)
    results['RF_prob'] = y_score_answer_RF[:, 0]
    
    #Perform 3-fold cross validation and return the mean accuracy on each fold    
    cv_scores_RF = cross_val_score(RFclf, X_train, y_train) #default 3-fold cross validation
    print('Random Forest cv_scores', cv_scores_RF)
#    answer_RF = pd.DataFrame(y_score_answer_RF)
    save_RF = '../models/trained_RF.sav'
    pickle.dump(RFclf, open(save_RF, 'wb'))
    #print('Random Forest\n', answer_RF.head())

    # confusion matrix
    cmRF = confusion_matrix(y_test, y_test_predictions_RF, labels=list(class_names))
    plt.subplot(1, 4, 3)
    plot_confusion_matrix(cm1=cmRF, classes=class_names, normalize=True, gradientbar=False,
                          title='Random Forests\nestimators: {0}\n max_features: {1}\n'.format(n, f))
    cv_scores_RF = ["{:.2f}".format(x) for x in cv_scores_RF]
    
    p_r_fscore_RF = precision_recall_fscore_support(y_test, y_test_predictions_RF, 
                                    beta=2.0, labels=['Parasitized'], pos_label='Parasitized',
                                    average='binary')
    print(p_r_fscore_RF[:3])
    plt.text(0.01, -1,'\nCV Scores:\n'+ str(cv_scores_RF) + '\n' + 
             'Precision: {d[0]:.2f}\nRecall: {d[1]:.2f} \nF2 score: {d[2]:.2f} \n'.format(
            d = p_r_fscore_RF[:3]), ha='left', va='bottom', fontdict = font,
            transform= plt.subplot(1,4,3).transAxes)
    
    # ### Adaptive Boosting Classifier
    # http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html

    # In[9]:

    AdaBoost = AdaBoostClassifier()
    AdaBoost.fit(X_train, y_train)
    y_predAB = AdaBoost.predict(X_test)
    y_predAB_prob = AdaBoost.predict_proba(X_test)
#    y_predAB_binarized = label_binarize(y_predAB,
#                                     classes=['single_product','market_place'])
    # save results into a DF
    results['AB_pred'] = list(y_predAB)
    results['AB_prob'] = y_predAB_prob[:, 0]
    
    results.to_csv('../data/y_test_predictions')
    #Perform 3-fold cross validation and return the mean accuracy on each fold
    cv_scores_AB = cross_val_score(AdaBoost, X_train, y_train) #default 3-fold cross validation
    print('Adaptive Boosting cv_scores', cv_scores_AB)
    save_AdaBoost = '../models/trained_AdaBoost.sav'
    pickle.dump(AdaBoost, open(save_AdaBoost, 'wb'))

    plt.subplot(1, 4, 4)
    cmAdaBoost = confusion_matrix(y_test, y_predAB, labels=list(class_names))
    plot_confusion_matrix(cm1=cmAdaBoost, normalize=True, classes=class_names,
                          title='AdaBoost\n', gradientbar=False)
    cv_scores_AB = ["{:.2f}".format(x) for x in cv_scores_AB]
    
    p_r_fscore_AB = precision_recall_fscore_support(y_test, y_predAB, 
                                    beta=2.0, labels=['Parasitized'], pos_label='Parasitized',
                                    average='binary')
    print(p_r_fscore_AB[:3])
    
    plt.text(0.01, -1,'\nCV Scores:\n'+ str(cv_scores_AB) + '\n' + 
             'Precision: {d[0]:.2f}\nRecall: {d[1]:.2f} \nF2 score: {d[2]:.2f} \n'.format(
            d = p_r_fscore_AB[:3]), ha='left', va='bottom', fontdict = font,
            transform= plt.subplot(1,4,4).transAxes)
        
    # #### Comparing mean accuracy and confusion matrices of difference classification algorithrms

    # In[10]:
    print('\nLogistic Regression mean accuracy:', round(log_reg_classifier.score(X_test, y_test), 4))
    print('One vs Rest - Naive Bayes mean accuracy:', round(classifier.score(X_test, y_test), 4))
    print('Random Forest Classifier mean accuracy:', round(RFclf.score(X_test, y_test), 4))
    print('Adaptive Boosting Classifier mean accuracy:', round(AdaBoost.score(X_test, y_test), 4))
    plt.tight_layout()
    fig.tight_layout()
    plt.savefig('../plots/confusion_matrix_result_1.png')
    plt.show(block=False)
    
    ### -- ROC and AUC
    # Compute ROC curve and area the curve
    plt.figure(12)
#    print('y_test before binirization', y_test[0:4])
    y_test = label_binarize(y_test, classes=['Uninfected','Parasitized'])
#    print('y_test after binirization', y_test[0:4])

#    print(y_predict_prob_log_r[1:4, 0])
    fpr, tpr, thresholds = roc_curve(y_test, y_predict_prob_log_r[:, 0])
    roc_df = pd.DataFrame({'fpr':fpr,'tpr':tpr, 'thresholds':thresholds })
    roc_df.to_csv('../data/roc_data.csv')
#    tprs = [interp(mean_fpr, fpr, tpr)]
#    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic', fontsize=18)
    plt.plot(fpr, tpr, lw=2, color='#3399ff',
             label='AUC = {0:.2f}'.format(roc_auc))

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray',
             label='Chance', alpha=.8)

    plt.ylabel('True Positive Rate',fontsize=14)
    plt.xlabel('False Positive Rate',fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('../plots/ROC_CNN_log_reg.png')
    plt.show()
    plt.close('all')
    print('If launched from command line use ctrl+z to close all plots and finish')


if __name__ == '__main__':
    import sys
    ML_with_BN_feat(bn_feat_file=sys.argv[1], n_comp=int(sys.argv[2]), 
                    plotting=sys.argv[3])
# Command line use:
# python ML_with_BN_features.py ../data/bn_feat.csv 100 False
