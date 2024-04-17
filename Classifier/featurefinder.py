import re
import os
import pandas as pd
import feLower
import feUpper


def FeatureFinder(file,folder):
    if (re.match("MT\_.*.77\.csv$", file)):
        pathU = folder + file
        dfUpper = pd.read_csv(pathU,skiprows=4,usecols=[1, 2, 3, 4, 5, 6])
        Stan_Dev_names_U, RMS_names_U, Entropy_names_U, JM_names_U, PN_names_U, Stan_Dev_values_U, RMS_values_U,Entropy_values_U, JM_values_U, PN_values_U = feUpper.feUpper(dfUpper)
        pathL = folder + file[0:27] + "1.csv"
        dfLower = pd.read_csv(pathL,skiprows=4,usecols=[1, 2, 3, 4, 5, 6])
        Stan_Dev_values_L, RMS_values_L,  Entropy_values_L, JM_values_L, PN_values_L,Stan_Dev_names_L,RMS_names_L, Entropy_names_L, JM_names_L, PN_names_L = feLower.feLower(dfLower)
    else:
        Stan_Dev_names_U =[]
        RMS_names_U = []
        Entropy_names_U = []
        JM_names_U = []
        PN_names_U = []
        Stan_Dev_values_U = []
        RMS_values_U = []
        Entropy_values_U = []
        JM_values_U = []
        PN_values_U = []
        Stan_Dev_values_L = [] 
        RMS_values_L = []
        Entropy_values_L = []
        JM_values_L = [] 
        PN_values_L = []
        Stan_Dev_names_L = []
        RMS_names_L = [] 
        Entropy_names_L = []
        JM_names_L = [] 
        PN_names_L = []
    return Stan_Dev_names_U, RMS_names_U, Entropy_names_U, JM_names_U, PN_names_U, Stan_Dev_values_U, RMS_values_U,Entropy_values_U, JM_values_U, PN_values_U,Stan_Dev_values_L, RMS_values_L,  Entropy_values_L, JM_values_L, PN_values_L,Stan_Dev_names_L,RMS_names_L, Entropy_names_L, JM_names_L, PN_names_L