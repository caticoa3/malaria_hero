#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: carlos atico ariza
"""
import os
import re
import tensorflow as tf
# import tflite_runtime.interpreter as tflite
import numpy as np
import pandas as pd
import sys
import gc


def tflite_img_class(image_dir=[], prediction_csv='malaria.csv',
                     trained_model='../models/model.tflite',
                     ):
    '''Classifies cells infected with the Malaria parasite
    using Tensor Flow lite

    Arguments
    image_dir:      A directory containing .png images; one cell per image
    prediction_csv: csv file where predictions for each image is saved
    trained_model:  the trained TFLite model modified to detect parasites in
                    single cells
    '''
    IMAGE_SIZE = 112

    if image_dir:
        print("Please select a directory housing .png images of single cell.")


    print(image_dir)
    path, dirs, files = next(os.walk(f'{image_dir}'))
    file_count = len(files)
    print(f'running in one batch: {file_count} images')

    # run all in one batch: make batch size = file counts
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255)

    # remove last folder, which is typically the class of the image
    image_dir = image_dir[:-9]
    # creates a generator that modifies images to 112x112 pixels
    unclassified_img_gen = (image_generator.flow_from_directory(
                            image_dir,
                            target_size=(
                                IMAGE_SIZE,
                                IMAGE_SIZE),
                            batch_size=file_count,
                            shuffle=False))

    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=trained_model)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    image_batch, image_name = next(unclassified_img_gen)

    predictions = []
    for image in image_batch:
        interpreter.set_tensor(input_details[0]['index'], [image])
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        predictions.append(output_data[0])
    predictions = np.vstack(predictions)

    positive_prob = predictions[:, 0]
    # for all images output class- 0: parasitized, 1: uninfected
    classifications = np.argmax(predictions, 1)
    print(f'{classifications=}')

    files_processed = pd.DataFrame({'fn': unclassified_img_gen.filenames,
                                    'Predicted_label': classifications,
                                    'Parasitized_probability': positive_prob
                                    })

    # example file name: folder\C33P1thinF_IMG_20150619_114756a_cell_181.png
    def split_it(row):
        c = re.findall(r'C(\d{1,3})', row['fn'])
        patient = re.findall(r'P(\d{1,3})', row['fn'])
        cell_no = re.findall(r'cell_(\d{1,3})', row['fn'])
        for i, x in enumerate([c, patient, cell_no]):
            if x:
                if i == 0:
                    row['Slide'] = c[0]
                if i == 1:
                    row['Patient'] = patient[0]
                if i == 2:
                    row['Cell'] = cell_no[0]
        return row

    files_processed = files_processed.apply(split_it, axis=1)
    print(files_processed.head())
    files_processed.to_csv(f'../results/predicted_{prediction_csv}')

    # Aggregate results for each patient
    summary = pd.DataFrame()
    summary['Total Cells Examined'] = (files_processed.groupby(['Patient'])
                                       ['Predicted_label'].count())
    summary[u'% Infected Cells'] = 100*(1 - (files_processed.groupby('Patient')
                                        ['Predicted_label'].sum()
                                        / summary['Total Cells Examined']))
    summary['% Infected Cells'] = summary['% Infected Cells'].map(
                                          '{:,.1f}'.format).astype(float)

    summary.sort_values(by='% Infected Cells', ascending=False, inplace=True)
    summary.reset_index(inplace=True)
    summary.to_csv('../results/summary.csv', index=False)
    print(summary.head())

    # collect garbage
    del files_processed
    gc.collect()

    return summary, classifications


if __name__ == '__main__':
    tflite_img_class(image_dir=sys.argv[1], prediction_csv=sys.argv[2],
                     trained_model=sys.argv[3])

# For testing the script on command line:
# python tflite_pred ../datasets/cell_images test_predictions.csv
# ../models/trained_RF.sav
