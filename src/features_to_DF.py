#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 11:32:51 2018
@author: Carlos Atico Ariza, PhD
"""
# Creation of pandas DF with features for each image at botleneck

# imports the FeatureGen class found in the generate_features.py module
import os
from generate_features import FeatureGen
from pathlib import Path
import pandas as pd
import gc
#URL_file = pd.read_csv('../Webcapture/URLs_for_pipeline building.csv')

def gen_bn_features(image_dir='../datasets/cell_images/',
                    bn_features_file='../data/bn_feat.csv', min_samples=0,
                    training=True):
    gen = FeatureGen()
    # for loop picks out jpg captures of webpages saved in labeled
    # subdirecotries of the 'dataset' folder
    saved_feat_data = Path(bn_features_file)
    img_files = Path(image_dir).glob('**/*.png')

    len_img_file = len(list(img_files))
    print(len_img_file)
    print('len_img_file', len_img_file)
    for i, file in enumerate(Path(image_dir).glob('**/*.png')):
        size = os.stat(str(file)).st_size
        if size < 100:
            print(str(file), 'file size: less than 100 Bytes.\n',
                  'Moving on to next image.', flush=True)
            continue
        if (saved_feat_data.is_file() and (i == 0) and training):
            # file exists
            print('Found', bn_features_file, ', which contains features '
                  'extracted from prior images.\nWe will add to those data.')
            feat_df = pd.read_csv(bn_features_file, index_col=0)

            # print(str(file)) #Prints the full path of the file
            print(file.parts[-2])  # Prints the directory name
            # print(file.name)

            # directory name - file.parts[-2] - of the image being classified
            # is the image label. This becomes the first element of every row in the
            # following DF

        # Create column names if dataframe was not previously created and saved as a csv
        elif i == 0:
            print('No prior data file found. '
                  'Let\'s set up the data frame with our column names.')
            clmn_nm = ['label', 'fn']
            #print(file.is_file(), 'image found')
            features = gen.feature_gen(str(file))
            # List of feature names: x0, x1, x2....xn, based on 
            #the number of features (n); will be used to name our DF column.
            for j in range(len(features)):
                # print(len(row_i))
                clmn_nm.append('x{}'.format(str(j)))
            # Instantiates the feat_df with the column names
            feat_df = pd.DataFrame(columns=clmn_nm)
            print('Done')

        if file.name not in feat_df.fn.tolist():
            # print(type(feat_df.fn))
            print('Feature extracted for ', file.name)
            # initiate a list that will become the data for each row in a DF
            # if not training:
            #    row_i = [None,file.name]
            # else:
            row_i = [file.parts[-2], file.name]
            features = gen.feature_gen(str(file))
            row_i.extend(features.tolist())  # adds feature values to row
            feat_df.loc[len(feat_df)] = row_i
            print('Label, file name, and feature values for', file.name,
                  'have been added to the pands DF.', sep=' ')
        else:
            print('The feature values for', file.name,
                  'were previously extracted and saved.',
                  'Hurray, we saved computing time!', sep=' ')

        if i % 40 == 0:
            # Save all extracted features after every forty images
            feat_df.to_csv(bn_features_file)
            print('\nSaved features for', i, 'webpage images;',
                  str(len_img_file - i), 'image ramain.\n', sep=' ')
        elif i > (len_img_file - 40):
            feat_df.to_csv(bn_features_file)
        
        #close the tensor flow session
    gen.session_close()
    
    gc.collect()


    if min_samples > 0:
        # Filter out labels with very few samples (minimum number of samples).
        # Too few sample will likely lead to poor algo training.
        # Set min_sample to 0 if creating features for testing a alreaty trained algo(not used in training algo)
        label_count = feat_df.groupby('label')['fn'].count()
        keep_labels = list(label_count[label_count > min_samples].keys())

        bn_feat_for_ml = feat_df.loc[feat_df['label'].isin(keep_labels)
                                    ].reset_index(drop=True)

        # Save the features of labels that contain more than 10 samples in a new csv file
        print('These labels had more than', min_samples, 'samples:', keep_labels)
        print('Saved to filtered_', bn_features_file)
        bn_feat_for_ml.to_csv('filtered_' + bn_features_file)
        gc.collect()
        return bn_feat_for_ml
    else:
        gc.collect()
        # If no filtering then return all features extacted from images
        return feat_df
#gen_bn_features(image_dir = '../datasets/cell_images/', training = True)


if __name__ == '__main__':
    import sys
    if (sys.argv[3] == 'True'):
        training = True
    elif sys.argv[3] == 'False':
        training = False
    gen_bn_features(image_dir=sys.argv[1],
                    bn_features_file=sys.argv[2],
                    training=training)

# Example of script use at command line:
# python features_to_DF.py ../webcapture/ ../data/bn_feat.csv True
