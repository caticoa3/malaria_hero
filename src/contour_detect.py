# coding: utf-8
import os
import numpy as np
import cv2
from pathlib2 import Path
#import re

'''Run on Python2.7'''

image_dir = '../IllumCorr/'

mask_files = []
#check for mask files and make a list without the mask suffix
for i, file in enumerate(Path(image_dir).glob('**/*mask*')):
    #print(file)
    base_fn = str(file).split('_IC_mask')[0]
    #print(base_fn)
    mask_files.append(base_fn)
    
cropped_files = []
#check for mask files and make a list without the crop suffix
for i, file in enumerate(Path(image_dir).glob('**/*+crop*')):
    #print(file)
    base_fn = str(file).split('_IC+crop')[0]
    #print(base_fn)
    cropped_files.append(base_fn)

#make a list of images that were processed to a mask and cropped images containing only + labeled cells
match = set(mask_files) & set(cropped_files)
thefile = open('cell_crop_list.txt', 'w')
for item in match:
    thefile.write("%s\n" % item)
thefile.close()
    
for file in match:
    image = cv2.imread(file + '_IC+crop.tiff', 0)
    print(file + '_IC+crop.tiff')
    mask = cv2.imread(file + '_IC_mask.tiff', 0)
    print(file + '_IC_mask.tiff')
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #crop out each contour into a seperate image
    for i in range(len(contours)):
        #contours = np.array(contours).reshape((-1,1,2)).astype(np.int32)
        blank = np.zeros_like(image)
        cv2.drawContours(blank, contours, i, 255, -1)
        out = np.zeros_like(image)
        out[blank == 255] = image[blank == 255]
        #crop object enclosed by contour
        (x,y) = np.where(blank == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        out = out[topx:bottomx+1, topy:bottomy+1]
        #organize output into phenotype and date
        split_path = file.split('/')
        phenotype = split_path[-2]
        if phenotype in ['TUJ1','RIP','MAP2','GFAP']:
            date = split_path[-3]
            trunc_fn = split_path[-1].split()[0]
            save_fn = date + '_' + trunc_fn + '_' + str(i) + '.jpg'
            os.chdir('../cropped/' + phenotype.upper())
    #       print(os.getcwd())
    #       save_path = '/'.join([phenotype, save_fn])        
            cv2.imwrite(save_fn,out) 
            os.chdir('..')
            #cv2.imshow('out',out)
            #cv2.waitKey(0)
#cv2.destroyAllWindows()

