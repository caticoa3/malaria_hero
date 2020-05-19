#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 2018

@author: carlos atico ariza
"""
import os
import re
import tensorflow as tf
import tflite_runtime.interpreter as tflite
import numpy as np
import pandas as pd

import sys
import gc

#Test data for building pipeline
#d = {'URL' : ['https://nutcasehelmets.com/collections/adult/products/technicolor-with-mips','https://www.amazon.com']}
#test_pipe = pd.DataFrame(d)
#test_pipe.to_csv('test_pipe.csv', index = False)

def tflite_img_class(image_dir= [], prediction_csv = 'predictions.csv',
                  trained_model = '../models/trained_AB.sav',
                  features_file1 = 'prod_test_feat.csv',
                  min_samples1 = 0, training1 = False):
    '''This function classifies if a cell is infected with the Malaria parasite
    using Tensor flow and sklearn libraries.

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

    IMAGE_SIZE = 112
    BATCH_SIZE = 64

    if image_dir:
        print("Please select a directory containing .png images of single cell.")

    print(image_dir)
    path, dirs, files = next(os.walk("/usr/lib"))
    file_count = len(files)

    # run all in one batch: make batch size = file counts
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    unclassified = (image_generator.
                   flow_from_directory(image_dir, target_size=
                                      (IMAGE_SIZE, IMAGE_SIZE),
                                       batch_size=file_count,
                                       shuffle=False))

    # Load TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path=trained_model)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    image_batch, image_name = next(unclassified)

    predictions = []
    for image in image_batch:
        interpreter.set_tensor(input_details[0]['index'], [image])
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        predictions.append(output_data[0])

    predictions = np.vstack(predictions)
    positive_prob = predictions[:,0]
    print(positive_prob)
    classifications = np.argmax(predictions, 1)
    #---
    #Extract file name from bn_feat dataframe into a new df
    #Load csv file into pandas dataframe
    #add predicted labels and save a new csv
    files_processed = pd.DataFrame({'fn': unclassified.filenames,
                                    'Predicted_label': classifications,
                                    'Parasitized_probability': positive_prob,
                                    })

    #example file name: folder\C33P1thinF_IMG_20150619_114756a_cell_181.png,
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
    print(files_processed.head())
    files_processed.to_csv('../results/predicted_{}'.format(prediction_csv))

    summary = pd.DataFrame()
    summary['Total Cells Examined']= (files_processed.groupby(['Patient'])
                                     ['Predicted_label'].count())
    summary[u'% Infected Cells']= (files_processed.groupby('Patient')
                                   ['Predicted_label'].sum()
                                   /summary['Total Cells Examined'])

    print(summary.head())
    #Groupping by Patient,and slides to give a better product application
    #for screening slides.
    def percent_n_sum(row):
        row[u'% Infected Cells'] = 100 * row['Cell']/float(row['Cell'].sum())
        row['Total Cells Examined'] = row['Cell'].sum()
        return row
    print('marker 0')
    cell_counts = files_processed.groupby(by=['Patient','Predicted_label','Slide']).agg({'Cell':'count'})
    print('marker 1')
    totals = cell_counts.groupby(level=0).apply(percent_n_sum).reset_index()
    print('marker 2')

    # Mask to catch uninfected patients: when no cells are deemed parasitized
    uninfected_mask = ((totals['Predicted_label'] == 1) &
                      (totals['% Infected Cells'].astype(int) == 100))
    print('marker 3')

    # Change label to parazitized and % infection to 0 for uninfected patients
    totals.loc[uninfected_mask,['Predicted_label','% Infected Cells']] = 'Parasitized', 0

    # Mask to determine each patients infection level
    parasite_mask = totals['Predicted_label'] == 'Parasitized'
    print('marker 4')
    actionable_table = totals.loc[parasite_mask,:].drop(columns=['Slide','Predicted_label','Cell'])
    # Format total cells examined to integer
    actionable_table['Total Cells Examined'] = actionable_table['Total Cells Examined'].astype(int)
    # Sort based on the % of Infected Cell
    actionable_table.sort_values(by = '% Infected Cells', ascending=False, inplace=True)
    # Format to two decimal places
    actionable_table['% Infected Cells'] = actionable_table['% Infected Cells'].map('{:,.2f}'.format).astype(float)
    actionable_table.to_csv('../results/actionable_table.csv')
    print(actionable_table.head())
    # collect garbage
    del files_processed
    gc.collect()


    return actionable_table # bn_feat, files_processed.to_html(index=False)
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
    tflite_img_class(image_dir = sys.argv[1], prediction_csv = sys.argv[2],
                  trained_model = sys.argv[3], features_file1 = sys.argv[4],
                  min_samples1 = int(sys.argv[5]), training1 = int(sys.argv[6]))

#For testing the script on command line:
#python ../datasets/cell_images test_predictions.csv ../models/trained_RF.sav prod_test_feat.csv 0 False
