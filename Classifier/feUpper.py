import numpy as np
import pandas as pd
from scipy.stats import entropy

#MT_ csv files are from Collection Day 3 > Vi > Lift Right

# 77 Global reference
# 71 measurement

def feUpper(dfUpper):
    #Standard_77 
    Stan_Dev_names_U = []
    Stan_Dev_values_U = []

    #Root-Mean-Square_77
    RMS_names_U = []
    RMS_values_U = []

    #Information Entropy_77
    Entropy_names_U = []
    Entropy_values_U = []

    #Jerk Mertic_77
    #The jerk Metric is rms value of the second derivative of the data normalized with respect to the maximum value of the first derivative
    JM_names_U = []
    JM_values_U = []

    #Peak Number_77
    PN_names_U = []
    PN_values_U = []



    for i in dfUpper.columns:
        STD = np.std(dfUpper[i].values)
        Stan_Dev_names_U.append(i + "-77-STD")
        Stan_Dev_values_U.append(STD)
        rms_U = np.sqrt(np.mean(np.square(dfUpper[i].values)))
        RMS_names_U.append(i+"-77-RMS")
        RMS_values_U.append(rms_U)
        if (i[0] == "A"):
            secDev = dfUpper[i].diff()/0.01
            JM_names_U.append(i+"-77-JM")
            rmsJM = np.sqrt(np.mean(np.square(secDev.loc[1:].values)))
            JM_values_U.append(rmsJM)
        else:
            secDev = dfUpper[i].diff()/0.01
            thirdDev = secDev.diff()/0.01
            rmsJM = np.sqrt(np.mean(np.square(thirdDev.loc[2:].values)))
            JM_names_U.append(i+"-77-JM")
            JM_values_U.append(rmsJM)
        PN_names_U.append(i+"-77-PN")
        PN_Counter_U = []
        for j in range(0,len(dfUpper[i])-2):
            if(dfUpper[i].loc[j+1] > dfUpper[i].loc[j] and dfUpper[i].loc[j+1] > dfUpper[i].loc[j+2]):
                PN_Counter_U.append("Peak")
            else:
                pass
        PN = len(PN_Counter_U)
        PN_values_U.append(PN)
        Entropy_names_U.append(i+"-77-EN")
        binsN = int(len(dfUpper[i])*2)
        counts = dfUpper[i].value_counts(normalize=False, sort=False,bins=binsN)
        EN_U = entropy(counts)
        Entropy_values_U.append(EN_U)
    
    return Stan_Dev_names_U, RMS_names_U, Entropy_names_U, JM_names_U, PN_names_U, Stan_Dev_values_U, RMS_values_U,Entropy_values_U, JM_values_U, PN_values_U