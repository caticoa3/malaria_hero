#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 11:32:51 2018
@author: Carlos A Ariza, PhD
"""

'''This python script extract Image Features'''

import cv2
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from skimage import measure
from pair_scatter_plots import seaborn_pairwise_plot, caa_plot_pairs

image_dir = '../datasets/cell_images/'
parasitic_example = '../datasets/cell_images/Parasitized/C33P1thinF_IMG_20150619_121229a_cell_177.png'
#normal_example = '../datasets/cell_images/Uninfected/C1_thinF_IMG_20150604_104942_cell_185.png'
image_path = parasitic_example #normal_example
cv_features_file = '../data/cv_feat.csv'
draw_blobs = False
initial_clm_names = ['label','fn','average_blue','average_green','average_red']


#set up pandas df
cv_df = pd.DataFrame(columns=initial_clm_names)
#saved_cv_feat = Path(cv_features_file)
png_files = Path(image_dir).glob('**/*.png')
#print(len(list(png_files)))
#len_png_file = sum(1 for _ in png_files)
#print('len_png_file', len_png_file)

#Blob detector
#deafual parameters
#https://github.com/opencv/opencv/blob/master/modules/features2d/src/blobdetector.cpp#L95
#Need to change filtering by circularity to true

# Set up the detector with default parameters.
detector = cv2.SimpleBlobDetector_create()

for i, file in enumerate(png_files):

    cv_df.loc[i,'label'] = file.parts[-2]
    cv_df.loc[i,'fn'] = file.parts[-1]
#    if i > 3:
#        break
    image = cv2.imread(str(file), -1)
#cv_df['file_name'] = file.parts()
    
#image = cv2.imread(image_path)
#print(image.shape)

    #3D color histogram feature
#    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
#    print("3D histogram shape: {}, with {} values".format(
#          hist.shape, hist.flatten().shape[0]))
#    cv_df.loc[i,'3D_color_hist'] = hist.flatten()

    #Average pixel for RGB ignoring zeros because black areas have been segemented    
    #https://stackoverflow.com/questions/38542548/numpy-mean-of-nonzero-values
    avg_color_per_row = np.true_divide(image.sum(0),(image!=0).sum(0))
    avg_color = np.average(avg_color_per_row, axis=0)
    cv_df.loc[i, ['average_blue','average_green','average_red']] = avg_color
    
    # Detect BLOBs (binary large object .
    keypoints = detector.detect(image)
    
    #simple yes or no if image contains a blog
    n_blobs = len(keypoints)
    add_blob_size = 0
    if n_blobs > 0:
        cv_df.loc[i,'blob_detected'] = 1
        cv_df.loc[i,'num_of_blobs'] = n_blobs
        #Average diameter BLOBs in range(0,keypoints) 
        for blob in list(range(n_blobs)):
#            print(blob)
            add_blob_size += keypoints[blob].size
        cv_df.loc[i, 'average_blob_area'] = add_blob_size/n_blobs
    else:
        cv_df.loc[i,'blob_detected'] = 0

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    if draw_blobs:
        im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]),
                                (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        # Show keypoints
        cv2.imshow("Keypoints", im_with_keypoints)
        cv2.waitKey(0);
        cv2.destroyAllWindows()

    #Next remove black pixels from images and obtain circularity and area
    gray_image = cv2.imread(str(file),0)
    
    #Threshold out pixels that are equal to 0.
    _, thresh = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
    
    #Obtain measurments on cell area
    #http://scikit-image.org/docs/dev/api/skimage.measure.html
    region_areas = []
    region_measurments = measure.regionprops(thresh)
    num_regions = len(region_measurments)
    for region in list(range(num_regions)):
        region_areas.append(region_measurments[region].area)
    largest_region_idx = region_areas.index(max(region_areas))
    print('Largest region\'s index', largest_region_idx)
    cv_df.loc[i, 'cell_area'] = sum(region_areas)
    #Assuming the largest region should be the cell boundry we calculate the eccentricity
    cv_df.loc[i, 'cell_eccentricity'] = measure.regionprops(thresh)[largest_region_idx].eccentricity
    
    #Solidity: Ratio of pixels in the region to pixels of the convex hull image
    cv_df.loc[i, 'cell_solidity'] = measure.regionprops(thresh)[largest_region_idx].solidity

#Normalize some value by area
#cv_df[['norm_avg_blue','norm_avg_green','norm_avg_red']
#         ] = cv_df.loc[:,['average_blue','average_green','average_red']].div(
#             cv_df.seg_area, axis=0)
#    cv_df.drop(columns=['norm_avg_blue','norm_avg_green','norm_avg_red'],inplace=True)

#Reorder columns
cv_df = cv_df[['label', 'fn', 'cell_area','cell_eccentricity','cell_solidity',
               'average_blue', 'average_green', 'average_red', 'blob_detected',
               'num_of_blobs', 'average_blob_area']]
cv_df.to_csv(cv_features_file)

# In[]:

# -- plot histograms of features
cv_df = pd.read_csv(cv_features_file, index_col=0)

label_set = set(cv_df.label)
label_list = list(label_set)
colors_dict = {label_list[0]:'#3399ff', label_list[1]:'#ff9933'}
zorders = {label_list[0]:1, label_list[1]:0 }

cv_df.fillna(0,inplace=True)
cv_df['blob_detected'] = cv_df['blob_detected']*1
#Visualizing color and BLOB features only
#cv_df = cv_df[['label', 'fn', 'average_blue', 'average_green', 'average_red', 
#               'blob_detected', 'num_of_blobs', 'average_blob_area']]
#cv_df.groupby('label').hist(alpha=0.4)
fig, subplots = plt.subplots(2, 5, squeeze=True, figsize=(15,8.5))
subplots = subplots.ravel()
for z, col in enumerate(list(cv_df.columns[2:])):
    ax=subplots[z]
    print(col)
    print(z)
    #Randomly sample a smaller set
    for key, group in cv_df.groupby('label'):
        group[col].plot(ax=ax, color = colors_dict[key], zorder = zorders[key],
                    kind='hist',alpha=0.6, label=col)        
    ax.set_title(col)
    if (z != 0) or (z != 4):
        ax.legend().set_visible(False)
    ax.tick_params(direction = 'in')

# -- adding figure legend
lp = lambda k: plt.plot([], color=colors_dict[k], ms=10, mec="none",
                      label="{}".format(k), ls="", marker="s", alpha = 0.6)[0]
            
handles = [lp(k) for k in label_list]
            
plt.tight_layout()
plt.subplots_adjust(top=0.88, bottom=0.11, left=0.075, right=0.975, 
                    hspace=0.2, wspace=0.345)
plt.figlegend(handles, ['Infected','Normal'], 
              loc='upper right', borderaxespad=0.5, ncol=2, prop={'size': 10})
plt.suptitle('Extracted Features from Segmented Cells', fontsize= 20)

plt.show(fig)
plt.savefig('../plots/basic_feat_hists.png')

# In[]:
# Plot pairwise plots
plt.close()
feat_names = ['cell_area','cell_eccentricity','cell_solidity',
              'average_blue', 'average_green', 'average_red', 'blob_detected',
              'num_of_blobs', 'average_blob_area']

g = seaborn_pairwise_plot(cv_df, color_index='label', 
                          color_order = ['Parasitized','Uninfected'],
                          feature_names=feat_names,
                          n_comp=None)
plt.tight_layout()
plt.subplots_adjust(top=0.94, bottom=0.06, left=0.07, right=0.91, hspace=0.04,
wspace=0.025)
plt.yticks(rotation=30)
plt.ylabel(feat_names,rotation=30)
plt.show()
plt.savefig('../plots/seaborn_pp_basic_feat.png',dpi=450)

#caa_plot_pairs(X=cv_df.loc[:,feat_names], labels_DF=cv_df.loc[:,['label']])
  
#cv2.waitKey(0);
#cv2.destroyAllWindows()

#Extract value of "keypoints"
#https://stackoverflow.com/questions/30807214/opencv-return-keypoints-coordinates-and-area-from-blob-detection-python

# https://www.pyimagesearch.com/2014/01/22/clever-girl-a-guide-to-utilizing-color-histograms-for-computer-vision-and-image-search-engines/
#
#chans = cv2.split(image)
#colors = ("b", "g", "r")
#plt.figure()
#plt.title("'Flattened' Color Histogram")
#plt.xlabel("Bins")
#plt.ylabel("# of Pixels")
#features = []
#
## loop over the image channels
#for (chan, color) in zip(chans, colors):
#	# create a histogram for the current channel and
#	# concatenate the resulting histograms for each
#	# channel
#	hist = cv2.calcHist([chan], [0], None, [256], [1, 256])
#	features.extend(hist)
# 
#	# plot the histogram
#	plt.plot(hist, color = color)
#	plt.xlim([0, 256])
# 
## here we are simply showing the dimensionality of the
## flattened color histogram 256 bins for each channel
## x 3 channels = 768 total values -- in practice, we would
## normally not use 256 bins for each channel, a choice
## between 32-96 bins are normally used, but this tends
## to be application dependent
#print("flattened feature vector size: {}".format(np.array(features).flatten().shape))
