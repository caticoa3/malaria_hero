#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 12:01:15 2017

@author: carlos
"""

import pickle
import pandas as pd
from webpage_to_image import jpg_of_webpage
from features_to_DF import gen_bn_features
import sys

#Test data for building pipeline
#d = {'URL' : ['https://nutcasehelmets.com/collections/adult/products/technicolor-with-mips','https://www.amazon.com']}
#test_pipe = pd.DataFrame(d)
#test_pipe.to_csv('test_pipe.csv', index = False)

def web_img_class(csv_file1 = [], #'test_pipe.csv', 
                  trained_model = '../models/trained_RF.sav',
                  features_file1 = 'prod_test_feat.csv',
                  min_samples1 = 0, training1 = False):
    '''This function captures a snapshot of a webpage and classifies the type 
    of webpage using Tensor flow and sklearn libraries.
    
    Arguments    
    csv_file1:      a comma seperated file containing, at minimum a column with 
                    the URL's of websited to be classified.
    trained_model1: the saved, serilized, file which has the trained model 
                    to use for predicting webpage type
    features_file1: the name of the csv file to which bottleneck features from 
                    TensorFlow will be saved.
    min_samples1:   the minimum number of samples to include for each label 
                    when training the model
    training1:      if True the URLs from the csv_file1 will be added to the 
                    training data specified in features_file1. 
                    URLs must be labeled for training. If this is False
                    then a new features_file1 is created, features are 
                    extracted, and saved in features_file1.
    '''
    
    if csv_file1:
        print("Please load a csv file that conatain one column of URL's")

    #Make .png image of URL's in csv file
    image_dir1 = jpg_of_webpage(csv_file = csv_file1, 
                                sheets= ['Sun','Sean','Peter'], 
                                training1 = training1)
    
    #Generate BN features for images save csv file and return a Pandas DF with features.
    print('Starting generating features: gen_bn_features(...)')
    bn_feat = gen_bn_features(image_dir = image_dir1, bn_features_file = features_file1,
                              min_samples = min_samples1, training = training1)
    
    #Split the training data for cross validation
    X_fin = bn_feat.loc[:,'x0':].values
    y_fin = bn_feat.loc[:,'label'].values
    
#    print('len(X_fin)',len(X_fin))
#    print(y_fin)
    #Reduce features
    pca = pickle.load(open('../models/trained_PCA.sav', 'rb'))
    X_reduced = pca.transform(X_fin)
    
    #print(bn_feat)
    #print(head(bn_feat))
    
    #Load the stored trained algorithm
    #http://scikit-learn.org/stable/modules/model_persistence.html
    #loading using pickle
    #https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
    loaded_model = pickle.load(open(trained_model, 'rb'))
    predictions = loaded_model.predict(X_reduced)
#   probabilities = loaded_model.predict_proba(X_reduced)
    if training1:
        score = loaded_model.score(X_reduced, y_fin)
        print('Model accuracy:', score)
#        print('Predicted labels', predictions)
    if not training1:
        print('Predicted labels', predictions)
       
    #Import data from csv into a pandas dataframe
    #Load csv file into pandas dataframe
    #add predicted labels and save a new csv
    urls = pd.read_csv(csv_file1)
    urls = urls.assign(predicted_label= predictions)
    urls.to_csv('../results/predicted_{}'.format(csv_file1))

#    print(urls)
#    print(urls.shape)
#    print(urls.columns)
    #print(urls.to_html())
    return urls.to_html(index=False) #'Modified uploaded file with predictions:\n{0}'.format(urls)
    #Import data from csv into a pandas dataframe
    
    #Output predicted classifications
#web_img_class('test_pipe.csv', '../models/trained_RF.sav', 'prod_test_feat.csv', 0 ,False)

if __name__ == '__main__':
    web_img_class(csv_file1 = sys.argv[1], trained_model = sys.argv[2],
                  features_file1 = sys.argv[3],
                  min_samples1 = int(sys.argv[4]), training1 = int(sys.argv[5]))

#For testing the script on command line:
#python web_img_class_API.py test_pipe.csv ../models/trained_RF.sav prod_test_feat.csv 0 0
