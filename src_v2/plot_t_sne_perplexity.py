"""
t-SNE: Plots

We observe a tendency towards clearer shapes as the preplexity value increases.

The size, the distance and the shape of clusters may vary upon initialization,
perplexity values and does not always convey a meaning.

For further details, "How to Use t-SNE Effectively"
http://distill.pub/2016/misread-tsne/ provides a good discussion of the
effects of various parameters, as well as interactive plots to explore
those effects.
"""

# Author: Narine Kokhlikyan <narine@slice.com>
# License: BSD

print(__doc__)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets
from sklearn.model_selection import train_test_split
from time import time
from sklearn.decomposition import PCA
import pickle
from pair_scatter_plots import caa_plot_pairs

feat_df = pd.read_csv('../data/factors_n_bn_feat.csv', index_col=0, dtype='unicode')
#feat_df = feat_df.iloc[0:200,:]
y = feat_df.loc[:,['label','Date','group_idx']].values
X = feat_df.loc[:, 'x0':'x2047'].astype(float).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# load trained PCA model from disk
pca = pickle.load(open('../models/trained_PCA.sav', 'rb'))
X_train_pca = pca.fit_transform(X_train)
X_train_pca = X_train_pca[:,0:11]

# Hyperparameters for t-SNE
n_components = 5
method= 'exact' # ‘barnes_hut’ #
perplexities = [100] #[5, 30, 50, 100]
iterations = [1000] #[250, 1000, 5000]

#Setup the plot
(fig, subplots) = plt.subplots(nrows=len(iterations), ncols=len(perplexities)+1,
                               figsize=(15, 8), squeeze=False)
plt.suptitle('{} Comp. TSNE hypermater study'.format(n_components))

#Mask classes for colors assignments
red = y_train[:,0] == 'TUJ1'
green = y_train[:,0] == 'RIP'
blue = y_train[:,0] == 'GFAP'

labels = pd.DataFrame(y_train[:,0], columns=['label'])
labels = labels[labels['label'].isin(['TUJ1','RIP','GFAP'])]

for j, iteration in enumerate(iterations):
    print('j',j)
    ax = subplots[j][0]
    ax.scatter(X_train_pca[red, 0], X_train_pca[red, 1], c="r")
    ax.scatter(X_train_pca[green, 0], X_train_pca[green, 1], c="g")
    ax.scatter(X_train_pca[blue, 0], X_train_pca[blue, 1], c="blue")
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')


    for i, perplexity in enumerate(perplexities):
        ax = subplots[j][i + 1]
    
        t0 = time()
        #All features should be on the same scale - See StandardScaler for 
        # convenient ways of scaling heterogeneous data.
        # Using the exact method so that more than 3 dimensions/n_components can be used
        tsne = manifold.TSNE(n_components=n_components, init='random',
                             random_state=0, perplexity=perplexity, 
                             n_iter =  iteration, method=method)
        Y = tsne.fit_transform(X_train_pca)
        t1 = time()

        save_model = '../models/{}_comp._tsne_{}_{}_iter_{}_perplex.sav'.format(
        n_components, method, iteration, perplexity)
        pickle.dump(pca, open(save_model, 'wb'))

        print("perplexity={0}, iter={1} in {2:.2g} sec".format(perplexity, 
              iteration, t1 - t0))
        ax.set_title("Perplexity={0}, iter={1}".format(perplexity,iteration))
        ax.scatter(Y[red, 0], Y[red, 1], c="r")
        ax.scatter(Y[green, 0], Y[green, 1], c="g")
        ax.scatter(Y[blue, 0], Y[blue, 1], c="blue")
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        ax.axis('tight')
        fig.savefig('../plots/{}_C_tsne_HP_study.png'.format(n_components), dpi=600)
        #Plot and save customized pairwise plots
        caa_plot_pairs(Y,labels, 't-SNE 100_per_1000_iterations')
   
plt.show()

#tsne2 = pickle.load(open('../models/5_comp._tsne_exact_1000_iter_100_perplex.sav', 'rb'))
#Y = tsne2.fit_transform(X_train_pca)

