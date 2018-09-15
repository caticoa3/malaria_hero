This code is meant to extract the features from the parasitized and uninfected cells to aid in improved malaria disease screening.
However, you can use these codes as the skeleton to make use of pretrained models as feature extractors for your task of interest.
Simply use the skeleton of feature_extraction.py and extract the features from the most optimal layer from the model of your interest for the underlying data. 
Once you extract the features, modifty the code if you wish, to add your own fully connected layers at the top to train on the underlying data. 
You shall optimize the model hyper-parameters to suit your data.The code has dependencies to load_data and evaluation. 

