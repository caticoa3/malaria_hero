# coding: utf-8

'''Adds the meta into the csv file containing bottle neck features'''

# Pandas 0.23.2
import pandas as pd

feat = pd.read_csv('../data/bn_feat.csv', index_col=0)

#feat_2 = feat.iloc[:,:-18]

exp_key = pd.read_csv('../data/experiment_variables.csv',
                      usecols = ['Metadata_Date','Metadata_Slide',
                                 'Protein','NCCC','Metadata_Pattern'])


df_factors = exp_key.drop_duplicates()

def chop_fn_to_factor(row):
   #print(row)
   stringies =  row['fn'].split('_')
   date = stringies[0]
   slide = stringies[1].split('.')[0]
#   print('date:', date, 'slide:', slide)
   
   #match with the date and slide with the experimental condition
   yo_results = df_factors.loc[(df_factors['Metadata_Date'] == date) &
                  (df_factors['Metadata_Slide'] == slide)]
#   print(yo_results)
   yo_dict = yo_results.iloc[0].to_dict()
#   print(yo_dict)
#   print([yo_dict[x] for x in ['Protein', 'NCCC']])
#   print(yo_dict['Protein'], yo_dict['NCCC'])
#   print(yo_results.loc[0,['Protein','NCCC']])
   return [yo_dict[x] for x in ['Protein', 'NCCC','Metadata_Date',
           'Metadata_Pattern']]

a= feat.apply(chop_fn_to_factor, result_type='expand', axis=1)
a.fillna(0, inplace=True)

feat_fact = a.join(feat)
feat_fact.rename(index=str, columns={0: "Protein", 1: "NCCC", 2: "Date", 3: "Pattern"},inplace=True)



#Grouping experimental conditions
group_vars = ['NCCC','Protein','Pattern']

feat_fact['group_idx'] = pd.Categorical(feat_fact[group_vars].astype(str).apply(
                                        "_".join, 1)).codes

#Reorder columns
reorder = [feat_fact.columns[-1]] + list(feat_fact.columns[:-1])
feat_fact = feat_fact[reorder]
feat_fact.to_csv('../data/factors_n_bn_feat.csv')
#feat_2.to_csv('../data/bn_feat.csv')
