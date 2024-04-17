import numpy as np
import pandas as pd
from scipy.stats import entropy

#MT_ csv files are from Collection Day 3 > Vi > Lift Right

# 77 Global reference
# 71 measurement


def feLower(dfLower):
    #Standard_71 
    Stan_Dev_names_L = []
    Stan_Dev_values_L = []

    #Root-Mean-Square_71
    RMS_names_L = []
    RMS_values_L = []

    #Information Entropy_71
    Entropy_names_L = []
    Entropy_values_L = []

    #Jerk Mertic_71
    #The jerk Metric is rms value of the second derivative of the data normalized with respect to the maximum value of the first derivative
    JM_names_L = []
    JM_values_L = []

    #Peak Number_71
    PN_names_L = []
    PN_values_L = []

    for i in dfLower.columns:
        STD = np.std(dfLower[i].values)
        Stan_Dev_names_L.append(i + "-71-STD")
        Stan_Dev_values_L.append(STD)
        rms_L = np.sqrt(np.mean(np.square(dfLower[i].values)))
        RMS_names_L.append(i+"-71-RMS")
        RMS_values_L.append(rms_L)
        if (i[0] == "A"):
            secDev = dfLower[i].diff()/0.01
            JM_names_L.append(i+"-71-JM")
            rmsJM = np.sqrt(np.mean(np.square(secDev.loc[1:].values)))
            JM_values_L.append(rmsJM)
        else:
            secDev = dfLower[i].diff()/0.01
            thirdDev = secDev.diff()/0.01
            rmsJM = np.sqrt(np.mean(np.square(thirdDev.loc[2:].values)))
            JM_names_L.append(i+"-71-JM")
            JM_values_L.append(rmsJM)
        PN_names_L.append(i+"-71-PN")
        PN_Counter_L = []
        for j in range(0,len(dfLower[i])-2):
            if(dfLower[i].loc[j+1] > dfLower[i].loc[j] and dfLower[i].loc[j+1] > dfLower[i].loc[j+2]):
                PN_Counter_L.append("Peak")
            else:
                pass
        PN = len(PN_Counter_L)
        PN_values_L.append(PN)
        Entropy_names_L.append(i+"-71-EN")
        binsN = int(len(dfLower[i])*2)
        counts = dfLower[i].value_counts(normalize=False, sort=False,bins=binsN)
        EN = entropy(counts)
        Entropy_values_L.append(EN)

    return Stan_Dev_values_L, RMS_values_L,  Entropy_values_L,  JM_values_L, PN_values_L,Stan_Dev_names_L,RMS_names_L, Entropy_names_L, JM_names_L, PN_names_L  