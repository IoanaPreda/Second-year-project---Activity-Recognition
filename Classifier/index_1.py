# %% [markdown]
# Import Modules needed

# %%
from pyexpat import features
import numpy as np
import pandas as pd
#to find features of test data
import os
import featurefinder
#plot graphs module
import matplotlib.pyplot as plt
#Normalisation modules
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
# Classifier module
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.preprocessing import LabelEncoder
import mongoInsert

# %% [markdown]
# Load Training Dataset with Features Extracted

# %%
dfTrainingData = pd.read_csv('Data/Train_features.csv')

# %% [markdown]
# K Radius Nearest Neighbors Classification
# 
# ```
# Note: in iloc format is [intial row: last row, intial column : last column]
# ```
# Split training data

# %%
df = dfTrainingData.iloc[:,:-2]
task = dfTrainingData.iloc[:,-1:]


print(df.columns)
# cols = [col for col in df.columns if col not in ['Acc_X-71-JM','Acc_Y-71-JM','Acc_Z-71-JM','Gyr_X-71-JM','Gyr_Y-71-JM','Gyr_Z-71-JM','Acc_X-77-JM','Acc_Y-77-JM','Acc_Z-77-JM','Gyr_X-77-JM','Gyr_Y-77-JM','Gyr_Z-77-JM','Acc_X-71-PN','Acc_Y-71-PN','Acc_Z-71-PN','Gyr_X-71-PN','Gyr_Y-71-PN','Gyr_Z-71-PN','Acc_X-77-PN','Acc_Y-77-PN','Acc_Z-77-PN','Gyr_X-77-PN','Gyr_Y-77-PN','Gyr_Z-77-PN']]
# features = df[cols]

# cols = [col for col in df.columns if col not in ['Acc_X-71-PN','Acc_Y-71-PN','Acc_Z-71-PN','Gyr_X-71-PN','Gyr_Y-71-PN','Gyr_Z-71-PN','Acc_X-77-PN','Acc_Y-77-PN','Acc_Z-77-PN','Gyr_X-77-PN','Gyr_Y-77-PN','Gyr_Z-77-PN']]
# features = df[cols]
features = df

le = LabelEncoder()
features.iloc[:,-1] = le.fit_transform(features.iloc[:,-1])

X_train, X_test, y_train, y_test = train_test_split(features, task, test_size = 0.20)
#We are performing a train test split on the dataset. We are providing the test size as 0.20, that means our training sample contains 320 training set and test sample contains 80 test set

# 'Acc_X-71-JM','Acc_Y-71-JM','Acc_Z-71-JM','Gyr_X-71-JM','Gyr_Y-71-JM','Gyr_Z-71-JM','Acc_X-77-JM','Acc_Y-77-JM','Acc_Z-77-JM','Gyr_X-77-JM','Gyr_Y-77-JM','Gyr_Z-77-JM'
# 'Acc_X-71-PN','Acc_Y-71-PN','Acc_Z-71-PN','Gyr_X-71-PN','Gyr_Y-71-PN','Gyr_Z-71-PN','Acc_X-77-PN','Acc_Y-77-PN','Acc_Z-77-PN','Gyr_X-77-PN','Gyr_Y-77-PN','Gyr_Z-77-PN'

# %% [markdown]
# Create Classifer Model: 

# %%
# create the model
r_n = 0.59825
model = RadiusNeighborsClassifier(radius=r_n)
model.set_params(outlier_label='Unclassified')

# create pipeline
pipeline = Pipeline(steps=[('norm', MinMaxScaler()),('model',model)])

#fit model
pipeline.fit(X_train,np.ravel(y_train))

# %% [markdown]
# Loading Testing Data and using classifer to predict Validation measurements

# %%
i = 0
correctMatch = 0
incorrectMatch = 0

incorrectMatchLift= 0
incorrectMatchRnR = 0
incorrectMatchSwing = 0 
incorrectMatchRotate = 0


# make a prediction
yhat = pipeline.predict(X_test)


# find confusion matrix
cm = confusion_matrix(y_test, yhat,labels=['Lift','Rotate','Swing','R&R'])
ac = accuracy_score(y_test,yhat)
acp = ac*100

print('\n Confusion matrix is: \n        Lift Rotate Swing   R&R \n Lift   [%s   %s      %s      %s] \n Rotate [%s   %s     %s      %s]   \n Swing  [%s    %s      %s      %s] \n R&R    [%s   %s      %s      %s]' % (cm[0][0],cm[0][1],cm[0][2],cm[0][3], cm[1][0], cm[1][1], cm[1][2], cm[1][3], cm[2][0], cm[2][1], cm[2][2], cm[2][3], cm[3][0], cm[3][1], cm[3][2], cm[3][3]))

print('\n Accuracy score is: %f' % acp)

f = open('acc_log.log', 'a')

f.write(f'\n radius is : {r_n} and accuracy is : {acp}% \n')

f.close()

# print(X_test.iloc[0])

# i = 0
# for x in yhat:
#     mongoInsert.insertData(x,X_test,i)
#     i = i + 1
# 
# 
# 
# print('Test Number: %d' % i)
# i = i + 1
# summarize prediction
# print('Predicted Class: %s' % yhat)
# print('Actual Class: %s' % )

# for folder in os.listdir("Data/Predict/"):
#     path = "Data/Predict/" + folder + "/"
#     for file in os.listdir(path):
#         pathfile = "Data/Predict/" + folder + "/"
#         Stan_Dev_names_U, RMS_names_U, Entropy_names_U, JM_names_U, PN_names_U, Stan_Dev_values_U, RMS_values_U,Entropy_values_U, JM_values_U, PN_values_U,Stan_Dev_values_L, RMS_values_L,  Entropy_values_L, JM_values_L, PN_values_L,Stan_Dev_names_L,RMS_names_L, Entropy_names_L, JM_names_L, PN_names_L = featurefinder.FeatureFinder(file,pathfile)
#         if (len(Stan_Dev_values_L) > 0 ):
#             # F_Names = Stan_Dev_names_L + RMS_names_L + Entropy_names_L + JM_names_L + PN_names_L + Stan_Dev_names_U + RMS_names_U + Entropy_names_U + JM_names_U + PN_names_U
#             # F_Values = Stan_Dev_values_L + RMS_values_L + Entropy_values_L + JM_values_L + PN_values_L + Stan_Dev_values_U + RMS_values_U + Entropy_values_U + JM_values_U + PN_values_U
#             F_Names = Stan_Dev_names_L + RMS_names_L + Entropy_names_L  + Stan_Dev_names_U + RMS_names_U + Entropy_names_U
#             F_Values = Stan_Dev_values_L + RMS_values_L + Entropy_values_L  + Stan_Dev_values_U + RMS_values_U + Entropy_values_U
#             dfFeaturesTest =  pd.DataFrame([],columns=F_Names)
#             dfFeaturesTest.loc[len(dfFeaturesTest)] = F_Values
#             # make a prediction
#             yhat = pipeline.predict(dfFeaturesTest)
#             # print('Test Number: %d' % i)
#             i = i + 1
#             # # summarize prediction
#             # print('Predicted Class: %s' % yhat)
#             # print('Actual Class: %s' % folder)
#             if (str(yhat)[2:-2] == str(folder)):
#                 correctMatch = correctMatch + 1
#             else:
# from sklearn.metrics import confusion_matrix,accuracy_score
# cm = confusion_matrix(y_test, y_pred)
# ac = accuracy_score(y_test,y_pred)
#                 incorrectMatch = incorrectMatch + 1
#                 if folder == 'Lift':
#                     incorrectMatchLift = incorrectMatchLift + 1
#                 elif folder == 'R&r':
#                     incorrectMatchRnR = incorrectMatchRnR + 1
#                     print("IS R&R --> Thinks is %s" % str(yhat)[2:-2])
#                 elif folder == 'Rotate':
#                     incorrectMatchRotate = incorrectMatchRotate + 1
#                 elif folder == 'Swing':
#                     incorrectMatchSwing = incorrectMatchSwing + 1
#                 else:
#                     pass
#         else:
#             pass

# %%
# print("Correct matches are : %d " % correctMatch)
# print("Incorrect matches are : %d " % incorrectMatch)
# n = i + 1
# Accuracy = (correctMatch/n)*100
# print("Accuracy Percentage : %f " % Accuracy)
# print("Incorrect matches for Lift are : %d " % incorrectMatchLift)
# print("Incorrect matches for R&R are : %d " % incorrectMatchRnR)
# print("Incorrect matches for Swing are : %d " % incorrectMatchSwing)
# print("Incorrect matches for Rotate are : %d " % incorrectMatchRotate)
# # yat = "['Lift']"
# print(yat[2:-2])


