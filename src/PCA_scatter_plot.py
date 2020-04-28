#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 22:39:05 2018

@author: Carlos A Ariza, PhD
"""
#import pickle as pkl
import pandas as pd
from itertools import combinations
#import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn

#feat_df = pd.read_csv('../data/factors_n_bn_feat.csv',index_col= 0)
##
#feat_df = feat_df.iloc[:,0:20]
#mask = feat_df.loc[:,'label'].isin(['TUJ1','RIP'])
#feat_df = feat_df.loc[mask,:]
#
##
###y = feat_df.loc[:,'label']
##print('Number of samples for each label \n', feat_df.groupby('label')['label'].count())
#X = feat_df.loc[:,'x0':].values
#labels_DF=feat_df.iloc[:,0:7]

def plot_pca(X, labels_DF, enable_plotting=True):
#    X = pd.read_pickle(pickled_PCA_file)
#    labels_DF = pd.read_csv(labels)
    n_comp = X.shape[1]
    label = labels_DF.loc[:,'label']
    label_set = set(label)
    label_list = list(label_set)
    print(label_set)
    # -- plotting
    if (enable_plotting) and (n_comp < 21):
        print('plotting scatter plots...')
        comp_list = list(combinations(range(X.shape[1]), 2))
        if len(comp_list) < 20:
            fig, subplots = plt.subplots(comp_list[-1][0],round(
                                        (len(comp_list)+0.5)/comp_list[-1][0]),
                                         squeeze=True,figsize=(15,8.5))
        else:
            fig, subplots = plt.subplots(comp_list[-1][0]-2,round(
                                         len(comp_list)/(comp_list[-1][0]-2)),
                                         figsize=(15,8.5))
        subplots = subplots.ravel()

        #Make labels to indicate predicted value
        #-1 True positive
        #-2 False positive
        #1 True negative
        #2 False negative
        legend_dict = {label_list[0]:label_list[0],-2:'FP',label_list[1]:label_list[1],2:'FN'}
        if 'y_true' in labels_DF.columns:
            mask_TN = ((y_true_1s == data.loc[:,'label_svm']) & (y_true_1s == 1))
            labels_DF.loc[mask_TN, 'TPTNFPFN']  = 1
            mask_TP = ((y_true_1s == data['label_svm']) & (y_true_1s == -1))
            labels_DF.loc[mask_TP,'TPTNFPFN']  = -1
            mask_FN = ((y_true_1s != data['label_svm']) & (y_true_1s == 1))
            labels_DF.loc[mask_FN,'TPTNFPFN']  = -2
            mask_FP = ((y_true_1s != data['label_svm']) & (y_true_1s == -1))
            labels_DF.loc[mask_FP,'TPTNFPFN']  = 2
            colors = [y_true_1s, labels_DF.loc[:,'TPTNFPFN']]
            order = [1,-1,-2,2] #bottom to top order for overlaping points on plot.
#            print(labels_DF.groupby('TPTNFPFN').size())
        else:
            colors = [label]
            order = [1,-1] #bottom to top order for overlaping points on plot.
        
        def pairwise_scatter_plots(j, comp_list, zorders, grouped,legend_dict):
            for i, n in enumerate(comp_list):
                for key, group in grouped:
                    group.plot(ax=subplots[i], kind='scatter', x=n[0], y=n[1], 
                               label=key, color=colors_dict[key],s=0.7, 
                               zorder=zorders[key],legend=False, alpha=0.6)
                subplots[i].tick_params(labelsize = 5, direction = 'in')
    #                    labelbottom='off', axis='both', which='both',
    #                    bottom='off', top='off')
    #            subplots[i].xaxis.set_ticklabels([])
    #            subplots[i].yaxis.set_ticklabels([])
                subplots[i].xaxis.label.set_visible(False)
                subplots[i].yaxis.label.set_visible(False)
                subplots[i].text(0.5,0.9,'{0} vs {1}'.format(n[0],n[1]),
                        horizontalalignment='center',transform=subplots[i].transAxes,
                        size=8)
            # -- adding figure legend
            lp = lambda i: plt.plot([],color=colors_dict[i],ms=np.sqrt(25), mec="none",
                                    label="{}".format(legend_dict[i]), ls="", marker="o")[0]
            
            handles = [lp(i) for i in np.unique(color)]
    #        plt.figlegend(handles, bbox_to_anchor=(1.05, 0), loc='lower left',
    #                   borderaxespad=0.)
            plt.figlegend(loc='upper left', borderaxespad=0., ncol=4)
            plt.suptitle('{} Principle Components'.format(n_comp))
            fig.tight_layout()
            plt.subplots_adjust(top=0.915, bottom=0.045, left=0.02, right=0.988,
                                hspace=0.2, wspace=0.1)
            fig.savefig('../plots/{0}_PC-scatter_plots_{1}.png'.format(
                                                         n_comp,j), dpi=600)
        
        
        PCAprojectedDF = pd.DataFrame(X)
        colors_dict = {-2:'blue', label_list[0]:'red', label_list[1]:'green', 2:'orange'}
        zorders = {-2:3, label_list[0]:2, label_list[1]:1, 2:4}

        for j, color in enumerate(colors): #Plot with 
            PCAprojectedDF.loc[:,'label'] = color

            grouped = PCAprojectedDF.groupby('label')
            pairwise_scatter_plots(j=j, comp_list=comp_list, zorders=zorders, 
                                   grouped=grouped,legend_dict=legend_dict)
        
        #Make one more plot with the True negatives plotted on top
#        PCAprojectedDF.loc[:,'label'] = labels_DF.loc[:,'TPTNFPFN']
        grouped = PCAprojectedDF.groupby('label')
        zorders = {-2:2, label_list[0]:4, label_list[1]:1, 2:3}
        pairwise_scatter_plots(j=2, comp_list=comp_list, zorders=zorders, 
                               grouped=grouped,legend_dict=legend_dict)
        
            
        print('...scatter plots saved in ../plots/ folder')
        # --seaborn scatter matrix        
##        PCAprojectedDF = feat_df
#        feature_names = feat_df.columns[7:]
#        pp = seaborn.pairplot(feat_df, vars=feature_names, 
#                         hue='group_idx', #hue_order = order, #size=5, #aspect=7,
#                         markers='.',
#                         plot_kws=dict(s=15, linewidth=0),
#                         grid_kws=dict(despine = False))
#        pp.fig.set_size_inches(14.4,14.4) #(14.4,14.4)
##        plt.rcParams['figure.figsize']=(10,10)
##        for ax in pp.diag_axes: ax.set_visible(False)
##        seaborn.axes_style()
#        plt.tight_layout()
#        plt.subplots_adjust(top=0.94,wspace=0.04, hspace=0.04, bottom=0.06, 
#                            left= 0.04)
#        plt.suptitle('{} Principle Components'.format(n_comp))
#        pp.savefig('../plots/{0}_PC-seaborn_pairplots.png'.format(
#                                                     n_comp), dpi=500)
        
#plot_pca(X, enable_plotting=True,labels_DF=labels_DF)

def seaborn_pairwise_plot(feat_df, color_index=None,feature_names=None,
                          n_comp=None):
    pp = seaborn.pairplot(feat_df, vars=feature_names, 
                     hue=color_index, #hue_order = order, #size=5, #aspect=7,
                     markers='.',
                     plot_kws=dict(s=15, linewidth=0),
                     grid_kws=dict(despine = False))
    pp.fig.set_size_inches(14.4,14.4) #(14.4,14.4)
#        plt.rcParams['figure.figsize']=(10,10)
#        for ax in pp.diag_axes: ax.set_visible(False)
#        seaborn.axes_style()
    plt.tight_layout()
    plt.subplots_adjust(top=0.94,wspace=0.04, hspace=0.04, bottom=0.06, 
                        left= 0.04)
    plt.suptitle('{} Principle Components'.format(n_comp))
    pp.savefig('../plots/{0}_PC-seaborn_pairplots.png'.format(n_comp),
               dpi=500)

def caa_plot_pairs(X, labels_DF, **kwargs):
    '''Pairwise plots of features. This is a custom ploting function that 
    differes from standard pairwise scatter matrices plots in libraries like 
    the seaborn or pandas's scatter_matrix(): only scatter plots - no 
    diagonal reduntant lower portion of the square.'''

    n_comp = X.shape[1]
    label = labels_DF.loc[:,'label']
    label_set = set(label)
    label_list = list(label_set)
    print(label_set)
    # -- plotting
    if (n_comp < 21):
        print('plotting scatter plots...')
        comp_list = list(combinations(range(X.shape[1]), 2))
        if len(comp_list) < 20:
            fig, subplots = plt.subplots(comp_list[-1][0],round(
                                        (len(comp_list)+0.5)/comp_list[-1][0]),
                                         squeeze=True,figsize=(15,8.5))
        else:
            fig, subplots = plt.subplots(comp_list[-1][0]-2,round(
                                         len(comp_list)/(comp_list[-1][0]-2)),
                                         figsize=(15,8.5))
        subplots = subplots.ravel()

        #Make labels to indicate predicted value
        #-1 True positive
        #-2 False positive
        #1 True negative
        #2 False negative
        legend_dict = {label_list[0]:label_list[0],-2:'FP',label_list[1]:label_list[1],2:'FN'}
        if 'y_true' in labels_DF.columns:
            mask_TN = ((y_true_1s == data.loc[:,'label_svm']) & (y_true_1s == 1))
            labels_DF.loc[mask_TN, 'TPTNFPFN']  = 1
            mask_TP = ((y_true_1s == data['label_svm']) & (y_true_1s == -1))
            labels_DF.loc[mask_TP,'TPTNFPFN']  = -1
            mask_FN = ((y_true_1s != data['label_svm']) & (y_true_1s == 1))
            labels_DF.loc[mask_FN,'TPTNFPFN']  = -2
            mask_FP = ((y_true_1s != data['label_svm']) & (y_true_1s == -1))
            labels_DF.loc[mask_FP,'TPTNFPFN']  = 2
            colors = [y_true_1s, labels_DF.loc[:,'TPTNFPFN']]
            order = [1,-1,-2,2] #bottom to top order for overlaping points on plot.
#            print(labels_DF.groupby('TPTNFPFN').size())
        else:
            colors = [label]
            order = [1,-1] #bottom to top order for overlaping points on plot.
        
        def pairwise_scatter_plots(j, comp_list, zorders, grouped,legend_dict):
            for i, n in enumerate(comp_list):
                for key, group in grouped:
                    group.plot(ax=subplots[i], kind='scatter', x=n[0], y=n[1], 
                               label=key, color=colors_dict[key],s=0.7, 
                               zorder=zorders[key],legend=False, alpha=0.6)
                subplots[i].tick_params(labelsize = 5, direction = 'in')
    #                    labelbottom='off', axis='both', which='both',
    #                    bottom='off', top='off')
    #            subplots[i].xaxis.set_ticklabels([])
    #            subplots[i].yaxis.set_ticklabels([])
                subplots[i].xaxis.label.set_visible(False)
                subplots[i].yaxis.label.set_visible(False)
                subplots[i].text(0.5,0.9,'{0} vs {1}'.format(n[0],n[1]),
                        horizontalalignment='center',transform=subplots[i].transAxes,
                        size=8)
            # -- adding figure legend
            lp = lambda i: plt.plot([],color=colors_dict[i],ms=np.sqrt(25), mec="none",
                                    label="{}".format(legend_dict[i]), ls="", marker="o")[0]
            
            handles = [lp(i) for i in np.unique(color)]
    #        plt.figlegend(handles, bbox_to_anchor=(1.05, 0), loc='lower left',
    #                   borderaxespad=0.)
            plt.figlegend(loc='upper left', borderaxespad=0., ncol=4)
            plt.suptitle('{} Principle Components'.format(n_comp))
            fig.tight_layout()
            plt.subplots_adjust(top=0.915, bottom=0.045, left=0.02, right=0.988,
                                hspace=0.2, wspace=0.1)
            fig.savefig('../plots/{0}_PC-scatter_plots_{1}.png'.format(
                                                         n_comp,j), dpi=600)
        
        
        PCAprojectedDF = pd.DataFrame(X)
        colors_dict = {-2:'blue', label_list[0]:'red', label_list[1]:'green', 2:'orange'}
        zorders = {-2:3, label_list[0]:2, label_list[1]:1, 2:4}

        for j, color in enumerate(colors): #Plot with 
            PCAprojectedDF.loc[:,'label'] = color

            grouped = PCAprojectedDF.groupby('label')
            pairwise_scatter_plots(j=j, comp_list=comp_list, zorders=zorders, 
                                   grouped=grouped,legend_dict=legend_dict)
        
        #Make one more plot with the True negatives plotted on top
#        PCAprojectedDF.loc[:,'label'] = labels_DF.loc[:,'TPTNFPFN']
        grouped = PCAprojectedDF.groupby('label')
        zorders = {-2:2, label_list[0]:4, label_list[1]:1, 2:3}
        pairwise_scatter_plots(j=2, comp_list=comp_list, zorders=zorders, 
                               grouped=grouped,legend_dict=legend_dict)
        
            
        print('...scatter plots saved in ../plots/ folder')
