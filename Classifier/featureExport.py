from matplotlib.cbook import print_cycles
import numpy as np
import pandas as pd
import os
import re
from scipy.stats import entropy
import feLower 
import feUpper 
import featurefinder
#import sklearn

#MT_ csv files are from Collection Day 3 > Vi > Lift Right

# 77 Global reference
# 71 measurement
F_Names = ['Acc_X-71-STD', 'Acc_Y-71-STD', 'Acc_Z-71-STD', 'Gyr_X-71-STD',
       'Gyr_Y-71-STD', 'Gyr_Z-71-STD', 'Acc_X-71-RMS', 'Acc_Y-71-RMS',
       'Acc_Z-71-RMS', 'Gyr_X-71-RMS', 'Gyr_Y-71-RMS', 'Gyr_Z-71-RMS',
       'Acc_X-71-EN', 'Acc_Y-71-EN', 'Acc_Z-71-EN', 'Gyr_X-71-EN',
       'Gyr_Y-71-EN', 'Gyr_Z-71-EN', 'Acc_X-71-JM', 'Acc_Y-71-JM',
       'Acc_Z-71-JM', 'Gyr_X-71-JM', 'Gyr_Y-71-JM', 'Gyr_Z-71-JM',
       'Acc_X-71-PN', 'Acc_Y-71-PN', 'Acc_Z-71-PN', 'Gyr_X-71-PN',
       'Gyr_Y-71-PN', 'Gyr_Z-71-PN', 'Acc_X-77-STD', 'Acc_Y-77-STD',
       'Acc_Z-77-STD', 'Gyr_X-77-STD', 'Gyr_Y-77-STD', 'Gyr_Z-77-STD',
       'Acc_X-77-RMS', 'Acc_Y-77-RMS', 'Acc_Z-77-RMS', 'Gyr_X-77-RMS',
       'Gyr_Y-77-RMS', 'Gyr_Z-77-RMS', 'Acc_X-77-EN', 'Acc_Y-77-EN',
       'Acc_Z-77-EN', 'Gyr_X-77-EN', 'Gyr_Y-77-EN', 'Gyr_Z-77-EN',
       'Acc_X-77-JM', 'Acc_Y-77-JM', 'Acc_Z-77-JM', 'Gyr_X-77-JM',
       'Gyr_Y-77-JM', 'Gyr_Z-77-JM', 'Acc_X-77-PN', 'Acc_Y-77-PN',
       'Acc_Z-77-PN', 'Gyr_X-77-PN', 'Gyr_Y-77-PN', 'Gyr_Z-77-PN','Jerk','Task']
dfFeatures =  pd.DataFrame([],columns=F_Names)

print (" \n Starting Extraction of Features... \n ")

for folder in os.listdir("Data/Lift"):
    path = "Data/Lift/" + folder  + "/"
    if (folder[0:4] == 'Jerk'):
        J_name = ["Jerk"]
        J_value = ["Yes"]
    else:
        J_name = ["Jerk"]
        J_value = ["No"]
    for file in os.listdir("Data/Lift/" + folder):
        Stan_Dev_names_U, RMS_names_U, Entropy_names_U, JM_names_U, PN_names_U, Stan_Dev_values_U, RMS_values_U,Entropy_values_U, JM_values_U, PN_values_U,Stan_Dev_values_L, RMS_values_L,  Entropy_values_L, JM_values_L, PN_values_L,Stan_Dev_names_L,RMS_names_L, Entropy_names_L, JM_names_L, PN_names_L = featurefinder.FeatureFinder(file,path)
        if (len(Stan_Dev_values_L) > 0 ):
            Task_name = ["Task"]
            Task_value = ["Lift"]
            F_Names = Stan_Dev_names_L + RMS_names_L + Entropy_names_L + JM_names_L + PN_names_L + Stan_Dev_names_U + RMS_names_U + Entropy_names_U + JM_names_U + PN_names_U + J_name + Task_name
            F_Values = Stan_Dev_values_L + RMS_values_L + Entropy_values_L + JM_values_L + PN_values_L + Stan_Dev_values_U + RMS_values_U + Entropy_values_U + JM_values_U + PN_values_U + J_value + Task_value
            dfFeatures.loc[len(dfFeatures)] = F_Values
        else:
            pass

print("\n Lift Complete... \n")

for folder in os.listdir("Data/R&R"):
    path = "Data/R&R/" + folder + "/"
    if (folder[0:4] == 'Jerk'):
        J_name = ["Jerk"]
        J_value = ["Yes"]
    else:
        J_name = ["Jerk"]
        J_value = ["No"]
    for file in os.listdir("Data/R&R/" + folder):
        Stan_Dev_names_U, RMS_names_U, Entropy_names_U, JM_names_U, PN_names_U, Stan_Dev_values_U, RMS_values_U,Entropy_values_U, JM_values_U, PN_values_U,Stan_Dev_values_L, RMS_values_L,  Entropy_values_L, JM_values_L, PN_values_L,Stan_Dev_names_L,RMS_names_L, Entropy_names_L, JM_names_L, PN_names_L = featurefinder.FeatureFinder(file,path)
        if (len(Stan_Dev_values_L) > 0 ):
            Task_name = ["Task"]
            Task_value = ["R&R"]
            F_Names = Stan_Dev_names_L + RMS_names_L + Entropy_names_L + JM_names_L + PN_names_L + Stan_Dev_names_U + RMS_names_U + Entropy_names_U + JM_names_U + PN_names_U + J_name + Task_name
            F_Values = Stan_Dev_values_L + RMS_values_L + Entropy_values_L + JM_values_L + PN_values_L + Stan_Dev_values_U + RMS_values_U + Entropy_values_U + JM_values_U + PN_values_U + J_value + Task_value
            dfFeatures.loc[len(dfFeatures)] = F_Values
        else:
            pass

print("\n R&R Complete... \n")

for folder in os.listdir("Data/Swing"):
    path = "Data/Swing/" + folder + "/"
    if (folder[0:4] == 'Jerk'):
        J_name = ["Jerk"]
        J_value = ["Yes"]
    else:
        J_name = ["Jerk"]
        J_value = ["No"]
    for file in os.listdir("Data/Swing/" + folder):
        Stan_Dev_names_U, RMS_names_U, Entropy_names_U, JM_names_U, PN_names_U, Stan_Dev_values_U, RMS_values_U,Entropy_values_U, JM_values_U, PN_values_U,Stan_Dev_values_L, RMS_values_L, Entropy_values_L, JM_values_L, PN_values_L,Stan_Dev_names_L,RMS_names_L, Entropy_names_L, JM_names_L, PN_names_L = featurefinder.FeatureFinder(file,path)
        if (len(Stan_Dev_values_L) > 0 ):
            Task_name = ["Task"]
            Task_value = ["Swing"]
            F_Names = Stan_Dev_names_L + RMS_names_L + Entropy_names_L + JM_names_L + PN_names_L + Stan_Dev_names_U + RMS_names_U + Entropy_names_U + JM_names_U + PN_names_U + J_name + Task_name
            F_Values = Stan_Dev_values_L + RMS_values_L + Entropy_values_L + JM_values_L + PN_values_L + Stan_Dev_values_U + RMS_values_U + Entropy_values_U + JM_values_U + PN_values_U + J_value + Task_value
            dfFeatures.loc[len(dfFeatures)] = F_Values
        else:
            pass

print("\n Swing Complete... \n")


for folder in os.listdir("Data/Rotate"):
    path = "Data/Rotate/" + folder + "/"
    if (folder[0:4] == 'Jerk'):
        J_name = ["Jerk"]
        J_value = ["Yes"]
    else:
        J_name = ["Jerk"]
        J_value = ["No"]
    for file in os.listdir("Data/Rotate/" + folder):
        Stan_Dev_names_U, RMS_names_U, Entropy_names_U, JM_names_U, PN_names_U, Stan_Dev_values_U, RMS_values_U,Entropy_values_U, JM_values_U, PN_values_U,Stan_Dev_values_L, RMS_values_L,  Entropy_values_L, JM_values_L, PN_values_L,Stan_Dev_names_L,RMS_names_L, Entropy_names_L, JM_names_L, PN_names_L = featurefinder.FeatureFinder(file,path)
        if (len(Stan_Dev_values_L) > 0 ):
            Task_name = ["Task"]
            Task_value = ["Rotate"]
            F_Names = Stan_Dev_names_L + RMS_names_L + Entropy_names_L + JM_names_L + PN_names_L + Stan_Dev_names_U + RMS_names_U + Entropy_names_U + JM_names_U + PN_names_U + J_name + Task_name
            F_Values = Stan_Dev_values_L + RMS_values_L + Entropy_values_L + JM_values_L + PN_values_L + Stan_Dev_values_U + RMS_values_U + Entropy_values_U + JM_values_U + PN_values_U + J_value + Task_value
            dfFeatures.loc[len(dfFeatures)] = F_Values
        else:
            pass
        
print("\n Rotate Complete... \n")

print("\n")
print(dfFeatures)
print("\n")
dfFeatures.to_csv('Data\Train_Features.csv',index=False)
print ("\n Extration of Features Done! \n")
