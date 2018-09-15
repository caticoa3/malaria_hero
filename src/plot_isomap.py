import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter
from sklearn import manifold
from sklearn.model_selection import train_test_split
from time import time
from sklearn.decomposition import PCA
import pickle
from pair_scatter_plots import caa_plot_pairs

feat_df = pd.read_csv('../data/factors_n_bn_feat.csv', index_col=0, 
                      dtype='unicode')
#feat_df = feat_df.iloc[0:200,:]
y = feat_df.loc[:,['label','Date','group_idx']].values
X = feat_df.loc[:, 'x0':'x2047'].astype(float).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=0)

# load trained PCA model from disk
pca = pickle.load(open('../models/trained_PCA.sav', 'rb'))
X_train_pca = pca.fit_transform(X_train)
X_train_pca = X_train_pca[:, 0:11]

# Hyperparameters dictionary for isomap
hyper_param = {}
path_method = 'auto'
hyper_param['n_neighbors_list'] = [100]
hyper_param['n_components_list'] = [10]
#Setup the plot
(fig, subplots) = plt.subplots(nrows=len(hyper_param['n_components_list']), 
                               ncols=len(hyper_param['n_neighbors_list'])+1,
                               figsize=(15, 8), squeeze=False)

#For ease of printing creating strings out of lists
str_dict = {}
for q, key in enumerate(['n_components_list','n_neighbors_list']):
    str_dict[key] = ', '.join([str(x) for x in hyper_param[key]])

plt.suptitle('{} Comp. isomap'.format(str_dict['n_components_list']))

#Mask classes for colors assignments
red = y_train[:,0] == 'TUJ1'
green = y_train[:,0] == 'RIP'
blue = y_train[:,0] == 'GFAP'

labels = pd.DataFrame(y_train[:,0], columns=['label'])
labels = labels[labels['label'].isin(['TUJ1','RIP','GFAP'])]

for j, n_components in enumerate(hyper_param['n_components_list']):
    print('j',j)
    ax = subplots[j][0]
    ax.scatter(X_train_pca[red, 0], X_train_pca[red, 1], c="r")
    ax.scatter(X_train_pca[green, 0], X_train_pca[green, 1], c="g")
    ax.scatter(X_train_pca[blue, 0], X_train_pca[blue, 1], c="blue")
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')


    for i, n_neighbors in enumerate(hyper_param['n_neighbors_list']):
        ax = subplots[j][i + 1]
    
        t0 = time()
        # All features should be on the same scale - See StandardScaler for 
        # convenient ways of scaling heterogeneous data.
        # Using the exact method so that more than 3 dimensions/n_components can be used
        isomap = manifold.Isomap(n_neighbors=n_neighbors, n_components=n_components,
                                 max_iter=None, path_method= path_method, n_jobs=1)
        
        #Returns a transformed array of n_components
        Y = isomap.fit_transform(X_train_pca)
        t1 = time()

        save_model = '../models/isomap_{}-{}_comp_{}_neighbors.sav'.format(
        path_method, n_components, n_neighbors)
        pickle.dump(pca, open(save_model, 'wb'))

        print("n_comp={0}, n_neighbors={1} in {2:.2g} sec".format(n_components, 
              n_neighbors, t1 - t0))
        ax.set_title("n_comp={0}, n_neighbors={1}".format(n_components, 
                     n_neighbors))
        #Here we plot one pair wise plot of the first two components
        ax.scatter(Y[red, 0], Y[red, 1], c="r")
        ax.scatter(Y[green, 0], Y[green, 1], c="g")
        ax.scatter(Y[blue, 0], Y[blue, 1], c="blue")
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        ax.axis('tight')
        fig.savefig('../plots/isomap_study-{}_comp_{}_neighbors.png'.format(
                    str_dict['n_components_list'], 
                    str_dict['n_neighbors_list'], dpi=600))
        #Plot and save customized pairwise plots
        caa_plot_pairs(Y, labels, 'isomap')
   
plt.show()

#tsne2 = pickle.load(open('../models/5_comp._tsne_exact_1000_iter_100_perplex.sav', 'rb'))
#Y = tsne2.fit_transform(X_train_pca)

