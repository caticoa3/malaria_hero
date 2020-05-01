# coding: utf-8
import pandas as pd
roc =  pd.read_csv('../data/roc_data.csv')
roc.head()
roc[fpr >=0.9]
mask = roc.loc[:,['fpr']]  >= 0.9 
roc[mask]
roc.loc[mask,:]
mask[0:5]
roc.loc[mask, [:]]
roc.loc[mask, :]
a = roc.loc[mask,:]
a.head()
roc.head()
mask = roc.loc[:,['fpr']]  >= 0.9 
roc.shape
roc.loc[mask, :].shape
roc[mask].shape
roc.shape
roc[mask].drop_na()
roc[mask].drop_nan()
roc[mask].drop_nan
roc[mask].na
roc[mask].dropna()
roc
roc[mask].dropna()
roc.shape
roc.dropna()
roc.dropna().shape
roc =  pd.read_csv('../data/roc_data.csv',index_col=0)
mask = roc['tpr']  >= 0.9
roc[mask].shape
roc[mask].df.sort(['tpr'], ascending=[False]).head()
roc[mask].sort(['tpr'], ascending=[False]).head()
roc[mask].sort(['tpr'], ascending=[False])
roc[mask].sort_values(['tpr','threshold'], ascending=[False,False])
roc.columns
roc[mask].sort_values(['tpr','thresholds'], ascending=[False,False])
roc[mask].sort_values(['tpr','thresholds'], ascending=[False,False]).head()
roc[mask].sort_values(['thresholds','tpr'], ascending=[True,False]).head()
roc[mask].sort_values(['thresholds'], ascending=['False']).head()
roc[mask].sort_values(['thresholds'], ascending=['False'])
roc[mask].sort_values(['thresholds'], ascending=[False])
roc[mask].sort_values(['thresholds'], ascending=[False]).head()
mask_fpr = roc['fpr']  >= 0.4
roc[mask].sort_values(['thresholds'], ascending=[False]).head()
roc[mask_fpr].sort_values(['thresholds'], ascending=[False]).head()
mask_fpr = roc['fpr']  <= 0.5 & (roc['fpr']  >= 0.2)
mask_fpr = (roc['fpr']  <= 0.5) & (roc['fpr']  >= 0.2)
roc[mask_fpr].sort_values(['thresholds'], ascending=[False]).head()
mask_fpr = (roc['fpr']  <= 0.5) & (roc['fpr']  >= 0.05)
roc[mask_fpr].sort_values(['thresholds'], ascending=[False]).head()
mask_fpr = (roc['fpr']  <= 0.2) & (roc['fpr']  >= 0.05)
roc[mask_fpr].sort_values(['thresholds'], ascending=[False]).head()
mask_fpr = (roc['fpr']  <= 0.3) & (roc['fpr']  >= 0.05)
roc[mask_fpr].sort_values(['thresholds'], ascending=[False]).head()
mask_fpr = (roc['fpr']  <= 0.3) & (roc['fpr']  >= 0.1)
roc[mask_fpr].sort_values(['thresholds'], ascending=[False]).head()
mask_fpr = (roc['fpr']  <= 0.2) & (roc['fpr']  >= 0.1)
roc[mask_fpr].sort_values(['thresholds'], ascending=[False])
mask_fpr = (roc['fpr']  <= 0.2) & (roc['fpr']  >= 0.05)
roc[mask_fpr].sort_values(['thresholds'], ascending=[False])
roc.plot(y='threshold',x='tpr')
roc.plot(y='thresholds',x='tpr')
roc.plot(y='thresholds',x='fpr')
get_ipython().magic("save 'selecting_threshold.py'")
