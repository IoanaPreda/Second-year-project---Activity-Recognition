# %%
import numpy as np
import pandas as pd
#to find features of test data
import os
import featurefinder
#from sklearn.cluster import KMeans as kmeans
#plot graphs module
import matplotlib.pyplot as plt
#Normalisation modules
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split 
# Classifier module
from sklearn.neighbors import RadiusNeighborsClassifier


# %%
dfTrainingData = pd.read_csv('Data/Train_features.csv')
#df = dfTrainingData.iloc[:,:-1]

# print(dfTrainingData)


# %%
# K Radius Nearest Neighbors Classification

# define dataset
# in iloc format is [intial row: last row, intial column : last column]

features = dfTrainingData.iloc[:,:-1]
task = dfTrainingData.iloc[:,-1:]

# create the model
model = RadiusNeighborsClassifier()

# create pipeline
pipeline = Pipeline(steps=[('norm', MinMaxScaler()),('model',model)])

#fit model
pipeline.fit(features,np.ravel(task))


# %%
#============================================
#Testing/Predicting data 
i = 0
for file in os.listdir("Data/Predict/"):
    path = "Data/Predict/"
    Stan_Dev_names_U, RMS_names_U, Entropy_names_U, JM_names_U, PN_names_U, Stan_Dev_values_U, RMS_values_U,Entropy_values_U, JM_values_U, PN_values_U,Stan_Dev_values_L, RMS_values_L,  Entropy_values_L, JM_values_L, PN_values_L,Stan_Dev_names_L,RMS_names_L, Entropy_names_L, JM_names_L, PN_names_L = featurefinder.FeatureFinder(file,path)
    if (len(Stan_Dev_values_L) > 0 ):
        F_Names = Stan_Dev_names_L + RMS_names_L + Entropy_names_L + JM_names_L + PN_names_L + Stan_Dev_names_U + RMS_names_U + Entropy_names_U + JM_names_U + PN_names_U
        F_Values = Stan_Dev_values_L + RMS_values_L + Entropy_values_L + JM_values_L + PN_values_L + Stan_Dev_values_U + RMS_values_U + Entropy_values_U + JM_values_U + PN_values_U
        dfFeaturesTest =  pd.DataFrame([],columns=F_Names)
        dfFeaturesTest.loc[len(dfFeaturesTest)] = F_Values
        # make a prediction
        i = i + 1
        yhat = pipeline.predict(dfFeaturesTest)

        print('Test Number: %d' % i)
        # summarize prediction
        print('Predicted Class: %s' % yhat)
    else:
        pass


#============================================