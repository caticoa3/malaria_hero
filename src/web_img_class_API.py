#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 2018

@author: carlos atico ariza
"""
import os
import pickle
import pandas as pd
import re
from features_to_DF import gen_bn_features
import sys

#Test data for building pipeline
#d = {'URL' : ['https://nutcasehelmets.com/collections/adult/products/technicolor-with-mips','https://www.amazon.com']}
#test_pipe = pd.DataFrame(d)
#test_pipe.to_csv('test_pipe.csv', index = False)

def web_img_class(image_dir= [], prediction_csv = 'predictions.csv', 
                  trained_model = '../models/trained_AB.sav',
                  features_file1 = 'prod_test_feat.csv',
                  min_samples1 = 0, training1 = False):
    '''This function captures a snapshot of a webpage and classifies the type 
    of webpage using Tensor flow and sklearn libraries.
    
    Arguments    
    image_dir:      A directory containing .png image of single cells. One cell
                    per image.
    prediction_csv: File name for the csv that in which predictions will be 
                    saved with file names.
    trained_model:  the saved, serilized, file which has the trained model 
                    to use for predicting webpage type
    features_file1: the name of the csv file to which bottleneck features from 
                    TensorFlow will be saved.
    min_samples1:   the minimum number of samples to include for each label 
                    when training the model
    training1:      if True the image_file features will be added to the 
                    training data specified in features_file1. 
                    Images must be in label folders for training. If this is 
                    False then a new features_file1 is created, features are 
                    extracted, and saved in features_file1.
    '''
    
    if image_dir:
        print("Please select a directory containing .png images of single cell.")

    #Generate BN features images file and return a Pandas DF with features.
    print('Starting generating features: gen_bn_features(...)')
    bn_feat = gen_bn_features(image_dir = image_dir, bn_features_file = features_file1,
                              min_samples = min_samples1, training = training1)
    
    #Place feature data in numpy array for use in classification.
    X_fin = bn_feat.loc[:,'x0':].values
#    print('len(X_fin)',len(X_fin))

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
    probabilities = loaded_model.predict_proba(X_reduced)
    positive_prob = probabilities[:,0]
    print('probabilities', positive_prob)
    if training1:
        # Runs if training flag is set to True
        # Place labeles into a array for training
        y_fin = bn_feat.loc[:,'label'].values
        #    print(y_fin)
        score = loaded_model.score(X_reduced, y_fin)
        print('Model accuracy:', score)
#        print('Predicted labels', predictions)
    if not training1:
        print('Predicted labels', predictions)
       
    #Extract file name from bn_feat dataframe into a new df
    #Load csv file into pandas dataframe
    #add predicted labels and save a new csv
    files_processed = bn_feat.loc[:,['fn']]
    
    #file neame processing: C33P1thinF_IMG_20150619_114756a_cell_181.png, C1_thinF_IMG_20150604_104722_cell_81.png, C6NThinF_IMG_20150609_121955_cell_51.png
    def split_it(row):
        c = re.findall('C(\d{1,3})',row['fn'])
        patient = re.findall('P(\d{1,3})',row['fn'])
        cell_no = re.findall('cell_(\d{1,3})',row['fn'])
        for i, x in enumerate([c, patient, cell_no]):
            if x:
                if i==0:
                    row['Slide'] = c[0] 
                if i==1:
                    row['Patient'] = patient[0]
                if i==2:
                    row['Cell'] = cell_no[0]
        return row
    
    files_processed = files_processed.apply(split_it, axis=1)
                
    files_processed = files_processed.assign(Predicted_label= predictions)
    files_processed = files_processed.assign(Parasitized_probability = positive_prob)
    files_processed = files_processed.round({'Parasitized_probability': 4})
    files_processed.to_csv('../results/predicted_{}'.format(prediction_csv))
    
    #Groupping by Patient,and slides to give a better product application 
    #for screening slides.
    def percent_n_sum(row):
        row[u'% Infected Cells'] = 100 * row['Cell']/float(row['Cell'].sum())
        row['Total Cells Examined'] = row['Cell'].sum()
        return row
    cell_counts = files_processed.groupby(by=['Patient','Predicted_label','Slide']).agg({'Cell':'count'})
    totals = cell_counts.groupby(level=0).apply(percent_n_sum).reset_index()
    parasite_mask = totals['Predicted_label'] == 'Parasitized'
    actionable_table = totals.loc[parasite_mask,:].drop(columns=['Slide','Predicted_label','Cell'])
    # Format total cells examined to integer
    actionable_table['Total Cells Examined'] = actionable_table['Total Cells Examined'].astype(int)
    # Sort based on the % of Infected Cell
    actionable_table.sort_values(by = '% Infected Cells', ascending=False, inplace=True)
    # Format to two decimal places
    actionable_table['% Infected Cells'] = actionable_table['% Infected Cells'].map('{:,.2f}'.format).astype(float)
    actionable_table.to_csv('../results/actionable_table.csv')
    print(actionable_table.head())
    
    return files_processed.to_html(index=False), actionable_table, files_processed, bn_feat
    #'Modified uploaded file with predictions:\n{0}'.format(urls)
    #Import data from csv into a pandas dataframe
    
    #Output predicted classifications
#web_img_class('test_pipe.csv', '../models/trained_RF.sav', 'prod_test_feat.csv', 0 ,False)

#For making list of uploaded files given a path
def make_tree(path):
    tree = dict(name=os.path.basename(path), children=[])
    try: lst = os.listdir(path)
    except OSError:
        pass #ignore errors
    else:
        for name in lst:
            fn = os.path.join(path, name)
            if os.path.isdir(fn):
                tree['children'].append(make_tree(fn))
            else:
                tree['children'].append(dict(name=name))
    return tree

if __name__ == '__main__':
    web_img_class(image_dir = sys.argv[1], prediction_csv = sys.argv[2],
                  trained_model = sys.argv[3], features_file1 = sys.argv[4],
                  min_samples1 = int(sys.argv[5]), training1 = int(sys.argv[6]))

#For testing the script on command line:
#python ../datasets/cell_images test_predictions.csv ../models/trained_RF.sav prod_test_feat.csv 0 False
