########################################################################################################################
# Fig. 6. Hyperbolic shapes found in the figure reported by Matsushita et al. (00_Kimoto_et_al_(2023)_[_Fig_06_].py)
########################################################################################################################
# Title of the manuscript (research):
# Findings in Travel-related Thrombosis and COVID-19 Related Cardiac Biomarker Data:
# Combination of Novel Review Strategy and Meta-analysis Method Discovered Twenty-eight Days Cycles of Thrombosis and
# Latent Subgroup Patterns
# Author of this research:
# Keiichiro Kimoto, M.Sc.1, 2, Munekazu Yamakuchi, M.D., Ph.D.1, Kazunori Takenouchi, M.D., Ph.D. 1,
# Teruto Hashiguchi, M.D., Ph.D.1
# Author affiliations
# 1 Department of Laboratory and Vascular Medicine, Kagoshima University Graduate School of Medical and Dental Sciences,
# 8-35-1 Sakuragaoka, Kagoshima 890-8544, Japan
# 2 Present Affiliation: External Advisor for Data Strategy Research Institute, Yokohama, Japan
# Corresponding authe: Keiichiro Kimoto, M.Sc.
#
# Program version:1.0
# Author of this program: Keiichiro Kimoto, M.Sc. in Biol. (Kagoshima University, Data Strategy Research Institute)
#
########################################################################################################################
# References:
# The Relationship of COVID-19 Severity with Cardiovascular Disease and Its Traditional Risk Factors:
# A Systematic Review and Meta-Analysis
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7546112/
# Glob Heart. 2020; 15(1): 64.
# Published online 2020 Sep 22. doi: 10.5334/gh.814
# PMCID: PMC7546112
# PMID: 33150129
########################################################################################################################

########################################################################################################################
# List of Import
########################################################################################################################
import pprint as pp
import pandas as pd
import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import matplotlib

from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN

import os
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option("display.width", 300)

########################################################################################################################
# For Manuscripts to Scientific Jounals
########################################################################################################################
print(matplotlib.get_cachedir())
#/home/kimoto/.cache/matplotlib: fontlist-v310.json  fontlist-v330.json  tex.cache#
print(matplotlib.matplotlib_fname())
print(matplotlib.rcParams["font.family"])
print(matplotlib.rcParams["font.sans-serif"])
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rcParams['font.serif'] = "Arial"
########################################################################################################################
# Making Data Sets
########################################################################################################################

###### "Cao J et al.", "Intensive Care Med", ######
list_1A = [1, 15, 2, "Cao J et al.", "Intensive Care Med", "A", "hypertension", "Yes", "Yes", 2,
           66, 31, 35, 18, 10, 8, 84, 18, 66]
list_1B = [1, 15, 2, "Cao J et al.", "Intensive Care Med", "B", "diabetes", "Yes", "Yes", 2,
           66, 31, 35, 18, 4, 14, 84, 7, 77]
list_1C = [1, 15, 2, "Cao J et al.", "Intensive Care Med", "C", "CVD", "Yes", "Yes", 2,
           66, 31, 35, 18, 2, 16, 84, 3, 81]

###### "Deng Y et al.", "Chin Med J (Engl)" ######
list_2A = [2, 20, 7, "Deng Y et al.", "Chin Med J (Engl)", "A", "hypertension", "Yes", "Yes", 2,
           69, 40, 29, 109, 40, 69, 116, 18, 98]
list_2B = [2, 20, 7, "Deng Y et al.", "Chin Med J (Engl)", "B", "diabetes", "Yes", "Yes", 2,
           69, 40, 29, 109, 17, 92, 116, 9, 107]
list_2C = [2, 20, 7, "Deng Y et al.", "Chin Med J (Engl)", "C", "CVD", "Yes", "Yes", 2,
           69, 40, 29, 109, 13, 96, 116, 4, 112]

###### "Guan W et al.", "N Eng J Med" ######
list_3A = [3, 21, 8, "Guan W et al.", "N Eng J Med", "A", "hypertension", "No", "Yes", 1,
           52, 45, 7, 173, 41, 132, 926, 124, 802]
list_3B = [3, 21, 8, "Guan W et al.", "N Eng J Med", "B", "diabetes", "No", "Yes", 1,
           52, 45, 7, 173, 28, 145, 926, 53, 873]
list_3C = [3, 21, 8, "Guan W et al.", "N Eng J Med", "C", "CVD", "Yes", "Yes", 1,
           52, 45, 7, 173, 10, 163, 926, 17, 909]

###### "Guo T et al.", "JAMA Cardiol" ######
list_4A = [4, 22, 9, "Guo T et al.", "JAMA Cardiol", "A", "hypertension", "No", "Yes", 1,
           71.4, 53.53, 17.87, 52, 33, 19, 135, 28, 107]
list_4B = [4, 22, 9, "Guo T et al.", "JAMA Cardiol", "B", "diabetes", "No", "Yes", 1,
           71.4, 53.53, 17.87, 52, 16, 36, 135, 12, 123]
list_4C = [4, 22, 9, "Guo T et al.", "JAMA Cardiol", "C", "CVD", "No", "Yes", 1,
           71.4, 53.53, 17.87, 52, 36, 16, 135, 12, 147]

###### "Wang D et al.", "JAMA" ######
list_5A = [5, 32, 19, "Wang D et al.", "JAMA", "A", "hypertension", "Yes", "Yes", 1,
           66, 51, 15, 36, 21, 15, 102, 22, 80]
list_5B = [5, 32, 19, "Wang D et al.", "JAMA", "B", "diabetes", "Yes", "Yes", 1,
           66, 51, 15, 36, 8, 28, 102, 6, 96]
list_5C = [5, 32, 19, "Wang D et al.", "JAMA", "C", "CVD", "Yes", "Yes", 1,
           66, 51, 15, 36, 9, 27, 102, 11, 91]

###### "Wang L et al.", "J Infect Mar" ######
list_6A = [6, 33, 20, "Wang L et al.", "J Infect Mar", "A", "hypertension", "Yes", "No", 1,
           76, 68, 8, 65, 32, 33, 274, 106, 168]
list_6B = [6, 33, 20, "Wang L et al.", "J Infect Mar", "B", "diabetes", "Yes", "No", 1,
           76, 68, 8, 65, 11, 54, 274, 43, 231]
list_6C = [6, 33, 20, "Wang L et al.", "J Infect Mar", "C", "CVD", "Yes", "No", 1,
           76, 68, 8, 65, 21, 44, 274, 32, 242]

###### "Wu C et al.", "JAMA Intern Med" ######
list_7A = [7, 34, 21, "Wu C et al.", "JAMA Intern Med", "A", "hypertension", "Yes", "Yes", 1,
           58.5, 48, 10.5, 84, 23, 61, 117, 16, 101]
list_7B = [7, 34, 21, "Wu C et al.", "JAMA Intern Med", "B", "diabetes", "Yes", "Yes", 1,
           58.5, 48, 10.5, 84, 16, 68, 117, 6, 111]
list_7C = [7, 34, 21, "Wu C et al.", "JAMA Intern Med", "C", "CVD", "No", "Yes", 1,
           58.5, 48, 10.5, 84, 5, 79, 117, 3, 114]

###### "Yuan M et al.", "PLoS One" ######
list_8A = [8, 35, 23, "Yuan M et al.", "PLoS One", "A", "hypertension", "Yes", "No", 0,
           68, 55, 13, 10, 5, 5, 17, 0, 17]
list_8B = [8, 35, 23, "Yuan M et al.", "PLoS One", "B", "diabetes", "Yes", "No", 0,
           68, 55, 13, 10, 6, 4, 17, 0, 17]
list_8C = [8, 35, 23, "Yuan M et al.", "PLoS One", "C", "CVD", "Yes", "No", 0,
           68, 55, 13, 10, 3, 7, 17, 0, 17]

###### "Zhou F et al.", "Lancet" ######
list_9A = [9, 37, 25, "Zhou F et al.", "Lancet", "A", "hypertension", "No", "Yes", 1,
           69, 52, 17, 54, 26, 28, 137, 32, 105]
list_9B = [9, 37, 25, "Zhou F et al.", "Lancet", "B", "diabetes", "No", "Yes", 1,
           69, 52, 17, 54, 17, 37, 137, 19, 118]
list_9C = [9, 37, 25, "Zhou F et al.", "Lancet", "C", "CVD", "Yes", "Yes", 1,
           69, 52, 17, 54, 13, 41, 137, 2, 135]

OriginalDataSet = pd.DataFrame(
    [list_1A, list_1B, list_1C, list_2A, list_2B, list_2C, list_3A, list_3B, list_3C, list_4A, list_4B, list_4C,
     list_5A, list_5B, list_5C, list_6A, list_6B, list_6C, list_7A, list_7B, list_7C, list_8A, list_8B, list_8C,
     list_9A, list_9B, list_9C],
   index=["1A", "1B", "1C", "2A", "2B", "2C", "3A", "3B", "3C", "4A", "4B", "4C", "5A", "5B", "5C", "6A", "6B", "6C",
          "7A", "7B", "7C", "8A", "8B", "8C", "9A", "9B", "9C"],
   columns=["Ca-No.", "Ref. No.", "Fig. 5", "Auther", "Jounal", "Sub-C", "Comorbidity", "Frag1", "Frag2", "Frag3",
            "Age (S)", "Age (Non-S)", "Age (Diff)", "Severe", "S_Yes", "S_No", "Non-severe", "NS_Yes",
            "NS_No"])

OriginalDataSet["OR"] = (OriginalDataSet["S_Yes"] / OriginalDataSet["S_No"]) / (OriginalDataSet["NS_Yes"] / OriginalDataSet["NS_No"])
OriginalDataSet["OR_U"] = np.exp(np.log(OriginalDataSet["OR"]) + 1.96*np.sqrt((1 / OriginalDataSet["S_Yes"]) + (1 / OriginalDataSet["S_No"]) + (1 / OriginalDataSet["NS_Yes"]) + (1 / OriginalDataSet["NS_No"])))
OriginalDataSet["OR_L"] = np.exp(np.log(OriginalDataSet["OR"]) - 1.96*np.sqrt((1 / OriginalDataSet["S_Yes"]) + (1 / OriginalDataSet["S_No"]) + (1 / OriginalDataSet["NS_Yes"]) + (1 / OriginalDataSet["NS_No"])))

OriginalDataSet["LN(OR)"] = np.log(OriginalDataSet["OR"])
OriginalDataSet["LN(U)"] = np.log(OriginalDataSet["OR_U"])
OriginalDataSet["LN(L)"] = np.log(OriginalDataSet["OR_L"])

OriginalDataSet["LN(U)diff"] = OriginalDataSet["LN(U)"] - OriginalDataSet["LN(OR)"]
OriginalDataSet["LN(L)diff"] = OriginalDataSet["LN(L)"] - OriginalDataSet["LN(OR)"]

OriginalDataSet["LN(RR)"] = np.log(
                                    (OriginalDataSet["S_Yes"] / (OriginalDataSet["S_Yes"] + OriginalDataSet["NS_Yes"]))
                                    / (OriginalDataSet["S_No"]/(OriginalDataSet["S_No"] + OriginalDataSet["NS_No"]))
                                    )


OriginalDataSet["OR - Abs( LN(L) - LN(OR) )"] =\
    OriginalDataSet["OR"] - np.abs( OriginalDataSet["LN(L)"] - OriginalDataSet["LN(OR)"] )

OriginalDataSet["OR + Abs( LN(U) - LN(OR) )"] =\
    OriginalDataSet["OR"] + np.abs( OriginalDataSet["LN(U)"] - OriginalDataSet["LN(OR)"] )

AnalysisDataSet_A = OriginalDataSet[(OriginalDataSet["Sub-C"] == "A") & (OriginalDataSet["Frag1"] == "Yes") & (OriginalDataSet["Frag2"] == "Yes")]
AnalysisDataSet_B = OriginalDataSet[(OriginalDataSet["Sub-C"] == "B") & (OriginalDataSet["Frag1"] == "Yes") & (OriginalDataSet["Frag2"] == "Yes")]
AnalysisDataSet_C = OriginalDataSet[(OriginalDataSet["Sub-C"] == "C") & (OriginalDataSet["Frag1"] == "Yes") & (OriginalDataSet["Frag2"] == "Yes")]

RegressionDataSet_0 = pd.DataFrame([["", "", "", "", "", "Origin", 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], index=["O"], columns=["Ref. No.", "Fig. 5", "Auther", "Jounal", "Sub-C", "Comorbidity", "Frag3", "OR", "Age (Diff)", "LN(OR)", "LN(U)", "LN(L)", "LN(U)diff", "LN(L)diff", "LN(RR)"])

RegressionDataSet_A = pd.concat([RegressionDataSet_0, AnalysisDataSet_A.loc[:, ["Ref. No.", "Fig. 5", "Auther", "Jounal", "Sub-C", "Comorbidity", "Frag3", "Age (Diff)", "OR", "LN(OR)", "LN(U)", "LN(L)", "LN(U)diff", "LN(L)diff", "LN(RR)", "OR - Abs( LN(L) - LN(OR) )", "OR + Abs( LN(U) - LN(OR) )"]]])
RegressionDataSet_B = pd.concat([RegressionDataSet_0, AnalysisDataSet_B.loc[:, ["Ref. No.", "Fig. 5", "Auther", "Jounal", "Sub-C", "Comorbidity", "Frag3", "Age (Diff)", "OR", "LN(OR)", "LN(U)", "LN(L)", "LN(U)diff", "LN(L)diff", "LN(RR)", "OR - Abs( LN(L) - LN(OR) )", "OR + Abs( LN(U) - LN(OR) )"]]])
RegressionDataSet_C = pd.concat([RegressionDataSet_0, AnalysisDataSet_C.loc[:, ["Ref. No.", "Fig. 5", "Auther", "Jounal", "Sub-C", "Comorbidity", "Frag3", "Age (Diff)", "OR", "LN(OR)", "LN(U)", "LN(L)", "LN(U)diff", "LN(L)diff", "LN(RR)", "OR - Abs( LN(L) - LN(OR) )", "OR + Abs( LN(U) - LN(OR) )"]]])

print("")
print("########## Whole of Data (AnalysisDataSet) ##########")
print(OriginalDataSet)
print("")
print("########## Data Set A (AnalysisDataSet_A) ##########")
print(AnalysisDataSet_A)
print("")
print("########## Data Set B (AnalysisDataSet_B) ##########")
print(AnalysisDataSet_B)
print("")
print("########## Data Set C (AnalysisDataSet_C) ##########")
print(AnalysisDataSet_C)
print("")
print("########## Data Set A (RegressionDataSet_A) ##########")
print(RegressionDataSet_A)
print("")
print("########## Data Set B (RegressionDataSet_B) ##########")
print(RegressionDataSet_B)
print("")
print("########## Data Set C (RegressionDataSet_C) ##########")
print(RegressionDataSet_C)

########################################################################################################################
# Regression Analysis
########################################################################################################################
############################################################################################################
# Definition of Regression Function
############################################################################################################
def reg_func_s2(parameter: ["k1", "k2", "l", "m"], x, y):
    k1 = parameter[0]
    k2 = parameter[1]
    l = 1.0
    m = parameter[2]
    _Output_ = y - (k1 / (1 + np.exp(-l * (x - m))) + k2)
    return _Output_

def reg_func_ciU(parameter: ["a", "c"], x, y, k1, k2, l, m):
    a = parameter[0]
    c = parameter[1]
    _Output_ = y - (k1 / (1 + np.exp(-l * (x - m))) + k2 + (abs(a)*(x - m)**2 + c))
    return _Output_

def reg_func_ciL(parameter: ["a", "c"], x, y, k1, k2, l, m):
    a = parameter[0]
    c = parameter[1]
    _Output_ = y - (k1 / (1 + np.exp(-l * (x - m))) + k2 + (-abs(a)*(x - m)**2 + c))
    return _Output_

############################################################################################################
# Non-linear Regression Analysis"
############################################################################################################
print("")
print("########## Evaluation for Result of Regression Analysis ##########")
############################################################################################################
# Sub-group A
############################################################################################################
###Sub-group A-1
x = RegressionDataSet_A.loc[(RegressionDataSet_A["Frag3"] == 0) | (RegressionDataSet_A["Frag3"] == 1), "Age (Diff)"]
y = RegressionDataSet_A.loc[(RegressionDataSet_A["Frag3"] == 0) | (RegressionDataSet_A["Frag3"] == 1), "OR"]
parameter = [0.00, 0.00, 0.00]; k1 = parameter[0]; k2 = parameter[1]; l = 1.0; m = parameter[2]
ResultA1S_1 = optimize.leastsq(reg_func_s2, parameter, args=(x, y), full_output=True)
ResultA1S_2 = ResultA1S_1[0]
ResultA1S_3 = [ResultA1S_2[0], ResultA1S_2[1], 1.00, ResultA1S_2[2]]
print("A1S", "Evaluation of Result:", ResultA1S_1[-1], "(1, 2, 3 or 4, the solution was found.)")

x = RegressionDataSet_A.loc[(RegressionDataSet_A["Frag3"] == 1), "Age (Diff)"]
y = RegressionDataSet_A.loc[(RegressionDataSet_A["Frag3"] == 1), "OR + Abs( LN(U) - LN(OR) )"]
parameter = [0.0, 0.0]; k1 = ResultA1S_3[0]; k2 = ResultA1S_3[1]; l = ResultA1S_3[2]; m = ResultA1S_3[3]
a = parameter[0]; c = parameter[1]
ResultA1CIU_1 = optimize.leastsq(reg_func_ciU, parameter, args=(x, y, k1, k2, l, m), full_output=True)
ResultA1CIU_2 = ResultA1CIU_1[0]
ResultA1CIU_3 = [ResultA1S_3[0], ResultA1S_3[1], ResultA1S_3[2], ResultA1S_3[3], ResultA1CIU_2[0], ResultA1CIU_2[1]]
print("A1U", "Evaluation of Result:", ResultA1CIU_1[-1], "(1, 2, 3 or 4, the solution was found.)")

x = RegressionDataSet_A.loc[(RegressionDataSet_A["Frag3"] == 1), "Age (Diff)"]
y = RegressionDataSet_A.loc[(RegressionDataSet_A["Frag3"] == 1), "OR - Abs( LN(L) - LN(OR) )"]
parameter = [0.0, 0.0]; k1 = ResultA1S_3[0]; k2 = ResultA1S_3[1]; l = ResultA1S_3[2]; m = ResultA1S_3[3]
a = parameter[0]; c = parameter[1]
ResultA1CIL_1 = optimize.leastsq(reg_func_ciL, parameter, args=(x, y, k1, k2, l, m), full_output=True)
ResultA1CIL_2 = ResultA1CIL_1[0]
ResultA1CIL_3 = [ResultA1S_3[0], ResultA1S_3[1], ResultA1S_3[2], ResultA1S_3[3], ResultA1CIL_2[0], ResultA1CIL_2[1]]
print("A1L", "Evaluation of Result:", ResultA1CIL_1[-1], "(1, 2, 3 or 4, the solution was found.)")

###Sub-group A-2
x = RegressionDataSet_A.loc[(RegressionDataSet_A["Frag3"] == 0) | (RegressionDataSet_A["Frag3"] == 2), "Age (Diff)"]
y = RegressionDataSet_A.loc[(RegressionDataSet_A["Frag3"] == 0) | (RegressionDataSet_A["Frag3"] == 2), "OR"]
parameter = [0.00, 0.00, 40.0]; k1 = parameter[0]; k2 = parameter[1]; l = 1.0; m = parameter[2]
ResultA2S_1 = optimize.leastsq(reg_func_s2, parameter, args=(x, y), full_output=True)
ResultA2S_2 = ResultA2S_1[0]
ResultA2S_3 = [ResultA2S_2[0], ResultA2S_2[1], 1.00, ResultA2S_2[2]]
print("A2S", "Evaluation of Result:", ResultA2S_1[-1], "(1, 2, 3 or 4, the solution was found.)")

x = RegressionDataSet_A.loc[(RegressionDataSet_A["Frag3"] == 2), "Age (Diff)"]
y = RegressionDataSet_A.loc[(RegressionDataSet_A["Frag3"] == 2), "OR + Abs( LN(U) - LN(OR) )"]
parameter = [0.0, 0.0]; k1 = ResultA2S_3[0]; k2 = ResultA2S_3[1]; l = ResultA2S_3[2]; m = ResultA2S_3[3]
a = parameter[0]; c = parameter[1]
ResultA2CIU_1 = optimize.leastsq(reg_func_ciU, parameter, args=(x, y, k1, k2, l, m), full_output=True)
ResultA2CIU_2 = ResultA2CIU_1[0]
ResultA2CIU_3 = [ResultA2S_3[0], ResultA2S_3[1], ResultA2S_3[2], ResultA2S_3[3], ResultA2CIU_2[0], ResultA2CIU_2[1]]
print("A2U", "Evaluation of Result:", ResultA2CIU_1[-1], "(1, 2, 3 or 4, the solution was found.)")

x = RegressionDataSet_A.loc[(RegressionDataSet_A["Frag3"] == 2), "Age (Diff)"]
y = RegressionDataSet_A.loc[(RegressionDataSet_A["Frag3"] == 2), "OR - Abs( LN(L) - LN(OR) )"]
parameter = [0.0, 0.0]; k1 = ResultA2S_3[0]; k2 = ResultA2S_3[1]; l = ResultA2S_3[2]; m = ResultA2S_3[3]
a = parameter[0]; c = parameter[1]
ResultA2CIL_1 = optimize.leastsq(reg_func_ciL, parameter, args=(x, y, k1, k2, l, m), full_output=True)
ResultA2CIL_2 = ResultA2CIL_1[0]
ResultA2CIL_3 = [ResultA2S_3[0], ResultA2S_3[1], ResultA2S_3[2], ResultA2S_3[3], ResultA2CIL_2[0], ResultA2CIL_2[1]]
print("A2L", "Evaluation of Result:", ResultA2CIL_1[-1], "(1, 2, 3 or 4, the solution was found.)")

############################################################################################################
# Sub-group B
############################################################################################################
###Sub-group B-1
x = RegressionDataSet_B.loc[(RegressionDataSet_B["Frag3"] == 0) | (RegressionDataSet_B["Frag3"] == 1), "Age (Diff)"]
y = RegressionDataSet_B.loc[(RegressionDataSet_B["Frag3"] == 0) | (RegressionDataSet_B["Frag3"] == 1), "OR"]
parameter = [0.00, 0.00, 0.00]; k1 = parameter[0]; k2 = parameter[1]; l = 1.0; m = parameter[2]
ResultB1S_1 = optimize.leastsq(reg_func_s2, parameter, args=(x, y), full_output=True)
ResultB1S_2 = ResultB1S_1[0]
ResultB1S_3 = [ResultB1S_2[0], ResultB1S_2[1], 1.00, ResultB1S_2[2]]
print("B1S", "Evaluation of Result:", ResultB1S_1[-1], "(1, 2, 3 or 4, the solution was found.)")

x = RegressionDataSet_B.loc[(RegressionDataSet_B["Frag3"] == 1), "Age (Diff)"]
y = RegressionDataSet_B.loc[(RegressionDataSet_B["Frag3"] == 1), "OR + Abs( LN(U) - LN(OR) )"]
parameter = [0.0, 0.0]; k1 = ResultB1S_3[0]; k2 = ResultB1S_3[1]; l = ResultB1S_3[2]; m = ResultB1S_3[3]
a = parameter[0]; c = parameter[1]
ResultB1CIU_1 = optimize.leastsq(reg_func_ciU, parameter, args=(x, y, k1, k2, l, m), full_output=True)
ResultB1CIU_2 = ResultB1CIU_1[0]
ResultB1CIU_3 = [ResultB1S_3[0], ResultB1S_3[1], ResultB1S_3[2], ResultB1S_3[3], ResultB1CIU_2[0], ResultB1CIU_2[1]]
print("B1U", "Evaluation of Result:", ResultB1CIU_1[-1], "(1, 2, 3 or 4, the solution was found.)")

x = RegressionDataSet_B.loc[(RegressionDataSet_B["Frag3"] == 1), "Age (Diff)"]
y = RegressionDataSet_B.loc[(RegressionDataSet_B["Frag3"] == 1), "OR - Abs( LN(L) - LN(OR) )"]
parameter = [0.0, 0.0]; k1 = ResultB1S_3[0]; k2 = ResultB1S_3[1]; l = ResultB1S_3[2]; m = ResultB1S_3[3]
a = parameter[0]; c = parameter[1]
ResultB1CIL_1 = optimize.leastsq(reg_func_ciL, parameter, args=(x, y, k1, k2, l, m), full_output=True)
ResultB1CIL_2 = ResultB1CIL_1[0]
ResultB1CIL_3 = [ResultB1S_3[0], ResultB1S_3[1], ResultB1S_3[2], ResultB1S_3[3], ResultB1CIL_2[0], ResultB1CIL_2[1]]
print("B1L", "Evaluation of Result:", ResultB1CIL_1[-1], "(1, 2, 3 or 4, the solution was found.)")

###Sub-group B-2
x = RegressionDataSet_B.loc[(RegressionDataSet_B["Frag3"] == 0) | (RegressionDataSet_B["Frag3"] == 2), "Age (Diff)"]
y = RegressionDataSet_B.loc[(RegressionDataSet_B["Frag3"] == 0) | (RegressionDataSet_B["Frag3"] == 2), "OR"]
parameter = [0.00, 0.00, 40.0]; k1 = parameter[0]; k2 = parameter[1]; l = 1.0; m = parameter[2]
ResultB2S_1 = optimize.leastsq(reg_func_s2, parameter, args=(x, y), full_output=True)
ResultB2S_2 = ResultB2S_1[0]
ResultB2S_3 = [ResultB2S_2[0], ResultB2S_2[1], 1.00, ResultB2S_2[2]]
print("B2S", "Evaluation of Result:", ResultB2S_1[-1], "(1, 2, 3 or 4, the solution was found.)")

x = RegressionDataSet_B.loc[(RegressionDataSet_B["Frag3"] == 2), "Age (Diff)"]
y = RegressionDataSet_B.loc[(RegressionDataSet_B["Frag3"] == 2), "OR + Abs( LN(U) - LN(OR) )"]
parameter = [0.0, 0.0]; k1 = ResultB2S_3[0]; k2 = ResultB2S_3[1]; l = ResultB2S_3[2]; m = ResultB2S_3[3]
a = parameter[0]; c = parameter[1]
ResultB2CIU_1 = optimize.leastsq(reg_func_ciU, parameter, args=(x, y, k1, k2, l, m), full_output=True)
ResultB2CIU_2 = ResultB2CIU_1[0]
ResultB2CIU_3 = [ResultB2S_3[0], ResultB2S_3[1], ResultB2S_3[2], ResultB2S_3[3], ResultB2CIU_2[0], ResultB2CIU_2[1]]
print("B2U", "Evaluation of Result:", ResultB2CIU_1[-1], "(1, 2, 3 or 4, the solution was found.)")

x = RegressionDataSet_B.loc[(RegressionDataSet_B["Frag3"] == 2), "Age (Diff)"]
y = RegressionDataSet_B.loc[(RegressionDataSet_B["Frag3"] == 2), "OR - Abs( LN(L) - LN(OR) )"]
parameter = [0.0, 0.0]; k1 = ResultB2S_3[0]; k2 = ResultB2S_3[1]; l = ResultB2S_3[2]; m = ResultB2S_3[3]
a = parameter[0]; c = parameter[1]
ResultB2CIL_1 = optimize.leastsq(reg_func_ciL, parameter, args=(x, y, k1, k2, l, m), full_output=True)
ResultB2CIL_2 = ResultB2CIL_1[0]
ResultB2CIL_3 = [ResultB2S_3[0], ResultB2S_3[1], ResultB2S_3[2], ResultB2S_3[3], ResultB2CIL_2[0], ResultB2CIL_2[1]]
print("B2L", "Evaluation of Result:", ResultB2CIL_1[-1], "(1, 2, 3 or 4, the solution was found.)")

############################################################################################################
# Sub-group C
############################################################################################################
###Sub-group C-1
x = RegressionDataSet_C.loc[(RegressionDataSet_C["Frag3"] == 0) | (RegressionDataSet_C["Frag3"] == 1), "Age (Diff)"]
y = RegressionDataSet_C.loc[(RegressionDataSet_C["Frag3"] == 0) | (RegressionDataSet_C["Frag3"] == 1), "OR"]
parameter = [21.00, 0.00, 5.00]; k1 = parameter[0]; k2 = parameter[1]; l = 1.0; m = parameter[2]
ResultC1S_1 = optimize.leastsq(reg_func_s2, parameter, args=(x, y), full_output=True)
ResultC1S_2 = ResultC1S_1[0]
ResultC1S_3 = [ResultC1S_2[0], ResultC1S_2[1], 1.00, ResultC1S_2[2]]
print("C1S", "Evaluation of Result:", ResultC1S_1[-1], "(1, 2, 3 or 4, the solution was found.)")

x = RegressionDataSet_C.loc[(RegressionDataSet_C["Frag3"] == 1), "Age (Diff)"]
y = RegressionDataSet_C.loc[(RegressionDataSet_C["Frag3"] == 1), "OR + Abs( LN(U) - LN(OR) )"]
parameter = [0.0, 0.0]; k1 = ResultC1S_3[0]; k2 = ResultC1S_3[1]; l = ResultC1S_3[2]; m = ResultC1S_3[3]
a = parameter[0]; c = parameter[1]
ResultC1CIU_1 = optimize.leastsq(reg_func_ciU, parameter, args=(x, y, k1, k2, l, m), full_output=True)
ResultC1CIU_2 = ResultC1CIU_1[0]
ResultC1CIU_3 = [ResultC1S_3[0], ResultC1S_3[1], ResultC1S_3[2], ResultC1S_3[3], ResultC1CIU_2[0], ResultC1CIU_2[1]]
print("C1U", "Evaluation of Result:", ResultC1CIU_1[-1], "(1, 2, 3 or 4, the solution was found.)")

x = RegressionDataSet_C.loc[(RegressionDataSet_C["Frag3"] == 1), "Age (Diff)"]
y = RegressionDataSet_C.loc[(RegressionDataSet_C["Frag3"] == 1), "OR - Abs( LN(L) - LN(OR) )"]
parameter = [0.0, 0.0]; k1 = ResultC1S_3[0]; k2 = ResultC1S_3[1]; l = ResultC1S_3[2]; m = ResultC1S_3[3]
a = parameter[0]; c = parameter[1]
ResultC1CIL_1 = optimize.leastsq(reg_func_ciL, parameter, args=(x, y, k1, k2, l, m), full_output=True)
ResultC1CIL_2 = ResultC1CIL_1[0]
ResultC1CIL_3 = [ResultC1S_3[0], ResultC1S_3[1], ResultC1S_3[2], ResultC1S_3[3], ResultC1CIL_2[0], ResultC1CIL_2[1]]
print("C1L", "Evaluation of Result:", ResultC1CIL_1[-1], "(1, 2, 3 or 4, the solution was found.)")

###Sub-group C-2
x = RegressionDataSet_C.loc[(RegressionDataSet_C["Frag3"] == 0) | (RegressionDataSet_C["Frag3"] == 2), "Age (Diff)"]
y = RegressionDataSet_C.loc[(RegressionDataSet_C["Frag3"] == 0) | (RegressionDataSet_C["Frag3"] == 2), "OR"]
parameter = [0.00, 0.00, 40.0]; k1 = parameter[0]; k2 = parameter[1]; l = 1.0; m = parameter[2]
ResultC2S_1 = optimize.leastsq(reg_func_s2, parameter, args=(x, y), full_output=True)
ResultC2S_2 = ResultC2S_1[0]
ResultC2S_3 = [ResultC2S_2[0], ResultC2S_2[1], 1.00, ResultC2S_2[2]]
print("C2S", "Evaluation of Result:", ResultC2S_1[-1], "(1, 2, 3 or 4, the solution was found.)")

x = RegressionDataSet_C.loc[(RegressionDataSet_C["Frag3"] == 2), "Age (Diff)"]
y = RegressionDataSet_C.loc[(RegressionDataSet_C["Frag3"] == 2), "OR + Abs( LN(U) - LN(OR) )"]
parameter = [0.0, 0.0]; k1 = ResultC2S_3[0]; k2 = ResultC2S_3[1]; l = ResultC2S_3[2]; m = ResultC2S_3[3]
a = parameter[0]; c = parameter[1]
ResultC2CIU_1 = optimize.leastsq(reg_func_ciU, parameter, args=(x, y, k1, k2, l, m), full_output=True)
ResultC2CIU_2 = ResultC2CIU_1[0]
ResultC2CIU_3 = [ResultC2S_3[0], ResultC2S_3[1], ResultC2S_3[2], ResultC2S_3[3], ResultC2CIU_2[0], ResultC2CIU_2[1]]
print("C2U", "Evaluation of Result:", ResultC2CIU_1[-1], "(1, 2, 3 or 4, the solution was found.)")

x = RegressionDataSet_C.loc[(RegressionDataSet_C["Frag3"] == 2), "Age (Diff)"]
y = RegressionDataSet_C.loc[(RegressionDataSet_C["Frag3"] == 2), "OR - Abs( LN(L) - LN(OR) )"]
parameter = [0.0, 0.0]; k1 = ResultC2S_3[0]; k2 = ResultC2S_3[1]; l = ResultC2S_3[2]; m = ResultC2S_3[3]
a = parameter[0]; c = parameter[1]
ResultC2CIL_1 = optimize.leastsq(reg_func_ciL, parameter, args=(x, y, k1, k2, l, m), full_output=True)
ResultC2CIL_2 = ResultC2CIL_1[0]
ResultC2CIL_3 = [ResultC2S_3[0], ResultC2S_3[1], ResultC2S_3[2], ResultC2S_3[3], ResultC2CIL_2[0], ResultC2CIL_2[1]]
print("C2L", "Evaluation of Result:", ResultC2CIL_1[-1], "(1, 2, 3 or 4, the solution was found.)")

RegressionCoef = pd.DataFrame([ResultA1S_3, ResultA1CIU_3, ResultA1CIL_3, ResultA2S_3, ResultA2CIU_3, ResultA2CIL_3,
                      ResultB1S_3, ResultB1CIU_3, ResultB1CIL_3, ResultB2S_3, ResultB2CIU_3, ResultB2CIL_3,
                      ResultC1S_3, ResultC1CIU_3, ResultC1CIL_3, ResultC2S_3, ResultC2CIU_3, ResultC2CIL_3],
                     index=["AS1", "AU1", "AL1", "AS2", "AU2", "AL2",
                            "BS1", "BU1", "BL1", "BS2", "BU2", "BL2",
                            "CS1", "CU1", "CL1", "CS2", "CU2", "CL2"],
                     columns=["k1", "k2", "l", "m", "a", "c"])
print("")
print("########## Data Sets of Regression Coefficients (RegressionCoef) ##########")
print(RegressionCoef)

Interval_AS1 = [0.00, 19.0]
Interval_AU1 = [0.00, 19.0]
Interval_AL1 = [0.00, 19.0]
Interval_AS2 = [0.00, 37.5]
Interval_AU2 = [22.5, 37.5]
Interval_AL2 = [22.5, 37.5]
Interval_BS1 = [0.00, 19.0]
Interval_BU1 = [0.00, 19.0]
Interval_BL1 = [0.00, 19.0]
Interval_BS2 = [0.00, 37.5]
Interval_BU2 = [22.5, 37.5]
Interval_BL2 = [22.5, 37.5]
Interval_CS1 = [0.00, 19.0]
Interval_CU1 = [0.00, 19.0]
Interval_CL1 = [0.00, 19.0]
Interval_CS2 = [0.00, 37.5]
Interval_CU2 = [0.00, 37.5]
Interval_CL2 = [0.00, 37.5]

Inter = pd.DataFrame([Interval_AS1, Interval_AU1, Interval_AL1, Interval_AS2, Interval_AU2,Interval_AL2,
                      Interval_BS1, Interval_BU1, Interval_BL1, Interval_BS2, Interval_BU2,Interval_BL2,
                      Interval_CS1, Interval_CU1, Interval_CL1, Interval_CS2, Interval_CU2, Interval_CL2 ],
                     index=["AS1", "AU1", "AL1", "AS2", "AU2", "AL2",
                            "BS1", "BU1", "BL1", "BS2", "BU2", "BL2",
                            "CS1", "CU1", "CL1", "CS2", "CU2", "CL2"],
                     columns=["x1", "x2"])

RegC = pd.merge(RegressionCoef, Inter, left_index=True, right_index=True)

print("")
print("########## Data Sets of Regression Coefficients and Interval of Curve (RegC) ##########")
print(RegC)

PAU1 = [0.005, ResultA1S_3[3], 0.625, -2.5, 27.5]
PAL1 = [-0.005, ResultA1S_3[3], -0.625, -2.5, 27.5]
PAU2 = [0.01, ResultA2S_3[3], 0.625, 17.25, 40.0]
PAL2 = [-0.01, ResultA2S_3[3], -0.625, 17.25, 40.0]
PBU1 = [0.005, ResultB1S_3[3], 0.875, -2.5, 25.0]
PBL1 = [-0.005, ResultB1S_3[3], -0.875, -2.5, 25.0]
PBU2 = [0.01, ResultB2S_3[3], 0.875, 17.25, 40.0]
PBL2 = [-0.01, ResultB2S_3[3], -0.875, 17.25, 40.0]
PCU1 = [0.01, ResultC1S_3[3], 0.65, -2.5, 25.0]
PCL1 = [-0.01, ResultC1S_3[3], -0.65, -2.5, 25.0]
PCU2 = [0.015, ResultC2S_3[3], 1.15, 17.25, 40.0]
PCL2 = [-0.015, ResultC2S_3[3], -1.15, 17.25, 40.0]

Para = pd.DataFrame([PAU1, PAL1, PAU2, PAL2, PBU1, PBL1, PBU2, PBL2, PCU1, PCL1, PCU2, PCL2],
                    index=["AU1", "AL1", "AU2", "AL2", "BU1", "BL1", "BU2", "BL2", "CU1", "CL1", "CU2", "CL2"],
                    columns=["a", "m", "c", "x1", "x2"])
print("")
print("########## Data Sets of Parabola for Explanation (Para) ##########")
print(Para)

LineA = [0.033, 0.38, 0.00, 40.0, AnalysisDataSet_A["Age (Diff)"][3], AnalysisDataSet_A["Age (Diff)"][0]]
LineB = [0.017, 0.58, 0.00, 40.0, AnalysisDataSet_B["Age (Diff)"][3], AnalysisDataSet_B["Age (Diff)"][0]]
LineC = [0.024, 1.04, 0.00, 40.0, AnalysisDataSet_C["Age (Diff)"][4], AnalysisDataSet_C["Age (Diff)"][0]]

Line = pd.DataFrame([LineA, LineB, LineC],
                    index=["A", "B", "C"],
                    columns=["a", "b", "x1", "x2", "x1'", "x2'"])

print("")
print("##########  Data Sets of Regression Line (Line) ##########")
print(Line)

########################################################################################################################
# Figures
########################################################################################################################
Figure_object = plt.figure(1, figsize=(11.69, 8.27), dpi=400, edgecolor="black", linewidth=0.5)
#plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.25, hspace=None)
plt.subplots_adjust(left=0.125, bottom=0.11, right=0.9, top=0.88, wspace=0.2 ,hspace=0.2)

plt.figtext(0.0440, 0.9700, "a", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")
plt.figtext(0.3750, 0.9700, "b", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")
plt.figtext(0.7060, 0.9700, "c", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")

plt.figtext(0.0440, 0.6500, "d", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")
plt.figtext(0.3750, 0.6500, "e", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")
plt.figtext(0.7060, 0.6500, "f", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")

plt.figtext(0.0440, 0.3250, "g", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")
plt.figtext(0.3750, 0.3250, "h", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")
plt.figtext(0.7060, 0.3250, "i", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")


gs_master = matplotlib.gridspec.GridSpec(nrows=3, ncols=1, height_ratios=[1, 1, 1])
gs_1 = matplotlib.gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=3, subplot_spec=gs_master[0])
gs_2 = matplotlib.gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=3, subplot_spec=gs_master[1])
gs_3 = matplotlib.gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=3, subplot_spec=gs_master[2])

#Axes_obj_01 = Figure_object.add_subplot(gs_1[0])
#Axes_obj_02 = Figure_object.add_subplot(gs_1[1])
#Axes_obj_03 = Figure_object.add_subplot(gs_1[2])
#Axes_obj_04 = Figure_object.add_subplot(gs_2[0])
#Axes_obj_05 = Figure_object.add_subplot(gs_2[1])
#Axes_obj_06 = Figure_object.add_subplot(gs_2[2])
#Axes_obj_07 = Figure_object.add_subplot(gs_3[0])
#Axes_obj_08 = Figure_object.add_subplot(gs_3[1])
#Axes_obj_09 = Figure_object.add_subplot(gs_3[2])

Axes_obj_01 = plt.axes([0.0606258, 0.723834-0.04, 0.226041*1.17, 0.232232*1.17])
Axes_obj_02 = plt.axes([0.391534, 0.723834-0.04, 0.227741*1.17, 0.232232*1.17])
Axes_obj_03 = plt.axes([0.722442, 0.723834-0.04, 0.224586*1.17, 0.232232*1.17])
Axes_obj_04 = plt.axes([0.0606258, 0.396547, 0.264726, 0.232232])
Axes_obj_05 = plt.axes([0.391534, 0.396547, 0.264726, 0.232232])
Axes_obj_06 = plt.axes([0.722442, 0.396547, 0.264726, 0.232232])
Axes_obj_07 = plt.axes([0.0606258, 0.0692597, 0.264726, 0.232232])
Axes_obj_08 = plt.axes([0.391534, 0.0692597, 0.264726, 0.232232])
Axes_obj_09 = plt.axes([0.722442, 0.0692597, 0.264726, 0.232232])
########################################################################################################################
############################################################################################################
# (a) Matsushita et al. Figure 5A
############################################################################################################
Axes_obj_01.set_title("Matsushita et al. Figure 5A", fontweight="normal", fontsize=11.4230)
Axes_obj_01.tick_params(length=0.0); Axes_obj_01.set_xticklabels([]); Axes_obj_01.set_yticklabels([])
img_01 = plt.imread("PMC7546112_Figure_5A.jpg"); Axes_obj_01.imshow(img_01)
############################################################################################################
# (b) Matsushita et al. Figure 5B
############################################################################################################
Axes_obj_02.set_title("Matsushita et al. Figure 5B", fontweight="normal", fontsize=11.4230)
Axes_obj_02.tick_params(length=0.0); Axes_obj_02.set_xticklabels([]); Axes_obj_02.set_yticklabels([])
img_02 = plt.imread("PMC7546112_Figure_5B.jpg"); Axes_obj_02.imshow(img_02)
############################################################################################################
# (c) Matsushita et al. Figure 5C
############################################################################################################
Axes_obj_03.set_title("Matsushita et al. Figure 5C", fontweight="normal", fontsize=11.4230)
Axes_obj_03.tick_params(length=0.0); Axes_obj_03.set_xticklabels([]); Axes_obj_03.set_yticklabels([])
img_03 = plt.imread("PMC7546112_Figure_5C.jpg"); Axes_obj_03.imshow(img_03)
########################################################################################################################
############################################################################################################
# "(d) Hypertension (C.I.)":
############################################################################################################
Axes_obj_04.set_title("Hypertension (LN(C.I.) of OR)", fontweight="normal", fontsize=11.4230)
Axes_obj_04.set_xlabel("Difference, Age"); Axes_obj_04.set_ylabel("log, C.I. of OR")
Axes_obj_04.set_xlim(-2.5, 42.5); Axes_obj_04.set_ylim(-2.75, 2.25)
Axes_obj_04.set_xticks([0, 5, 10, 15, 20, 25, 30, 35, 40])
Axes_obj_04.set_yticks([-2.0, -1.0, 0.0, 1.0, 2.0])
Axes_obj_04.set_xticklabels([0, 5, 10, 15, 20, 25, 30, 35, 40])
Axes_obj_04.set_yticklabels([-2.0, -1.0, 0.0, 1.0, 2.0])

for i in range(0, 9):
    Axes_obj_04.vlines(x=0.0+5.0*i, ymin=-2.0, ymax=2.0, color="gray", linestyle="dotted", linewidth=0.5)
for i in range(0, 9):
    Axes_obj_04.hlines(y=-2.0 + 0.5*i, xmin=0.0, xmax=40.0, color="gray", linestyle="dotted", linewidth=0.5)
Axes_obj_04.hlines(y=0.0, xmin=0.0, xmax=40.0, color="black", linestyle='solid', linewidth=1.0)
#Axes_obj_04.vlines(x=20.0, ymin=-2.0, ymax=2.0, color="black", linestyle='solid', linewidth=1.0)

for i in range(0, len(AnalysisDataSet_A["Age (Diff)"])):
    if AnalysisDataSet_A["Frag3"][i] == 1:
        Axes_obj_04.scatter(AnalysisDataSet_A["Age (Diff)"][i], 0.0,
                            color="black", s=50, zorder=3)
        Axes_obj_04.scatter(AnalysisDataSet_A["Age (Diff)"][i], AnalysisDataSet_A["LN(U)diff"][i],
                            color="black", marker='_', s=50, zorder=3)
        Axes_obj_04.scatter(AnalysisDataSet_A["Age (Diff)"][i], AnalysisDataSet_A["LN(L)diff"][i],
                            color="black", marker='_', s=50, zorder=3)
        Axes_obj_04.vlines(x=AnalysisDataSet_A["Age (Diff)"][i],
                           ymin=AnalysisDataSet_A["LN(L)diff"][i], ymax=AnalysisDataSet_A["LN(U)diff"][i],
                           color="black", linestyle='solid', linewidth=1.6318, zorder=3)
    if AnalysisDataSet_A["Frag3"][i] == 2:
        Axes_obj_04.scatter(AnalysisDataSet_A["Age (Diff)"][i], 0.0,
                            color="black", s=50, zorder=3)
        Axes_obj_04.scatter(AnalysisDataSet_A["Age (Diff)"][i], AnalysisDataSet_A["LN(U)diff"][i],
                            color="black", marker='_', s=50, zorder=3)
        Axes_obj_04.scatter(AnalysisDataSet_A["Age (Diff)"][i], AnalysisDataSet_A["LN(L)diff"][i],
                            color="black", marker='_', s=50, zorder=3)
        Axes_obj_04.vlines(x=AnalysisDataSet_A["Age (Diff)"][i],
                           ymin=AnalysisDataSet_A["LN(L)diff"][i], ymax=AnalysisDataSet_A["LN(U)diff"][i],
                           color="black", linestyle='solid', linewidth=1.6318, zorder=3)

Axes_obj_04.text(AnalysisDataSet_A["Age (Diff)"][0], AnalysisDataSet_A["LN(U)diff"][0] + 0.25,
                 AnalysisDataSet_A["Fig. 5"][0], ha="center", va="center", size=9.7912, color="black", fontweight="normal")
Axes_obj_04.text(AnalysisDataSet_A["Age (Diff)"][1], AnalysisDataSet_A["LN(U)diff"][1] + 0.25,
                 AnalysisDataSet_A["Fig. 5"][1], ha="center", va="center",  size=9.7912, color="black", fontweight="normal")
Axes_obj_04.text(AnalysisDataSet_A["Age (Diff)"][2], AnalysisDataSet_A["LN(U)diff"][2] + 0.25,
                 AnalysisDataSet_A["Fig. 5"][2], ha="center", va="center",  size=9.7912, color="black", fontweight="normal")
Axes_obj_04.text(AnalysisDataSet_A["Age (Diff)"][3], AnalysisDataSet_A["LN(U)diff"][3] + 0.25,
                 AnalysisDataSet_A["Fig. 5"][3], ha="center", va="center", size=9.7912, color="black", fontweight="normal")
#Axes_obj_04.text(AnalysisDataSet_A["Age (Diff)"][4], AnalysisDataSet_A["LN(U)diff"][4] + 0.25,
#                 AnalysisDataSet_A["Fig. 5"][4], ha="center", va="center",  size=9.7912, color="black", fontweight="normal")

Axes_obj_04.annotate(text=" ? ",
                     xy=(RegC.at["AS1", "m"]-2.5, 1.5),
                     ha='center', va='center', size=13.0549, color='red', fontweight="normal", alpha=1.0,
                     bbox=dict(boxstyle='square', edgecolor='black', fc='white', alpha=0.0))
Axes_obj_04.annotate(text=" ? ",
                     xy=(RegC.at["AS1", "m"]-2.5, -1.5),
                     ha='center', va='center', size=13.0549, color='red', fontweight="normal", alpha=1.0,
                     bbox=dict(boxstyle='square', edgecolor='black', fc='white', alpha=0.0))
Axes_obj_04.annotate(text=" ? ",
                     xy=(RegC.at["AS1", "m"]+2.75, -2.375),
                     ha='center', va='center', size=13.0549, color='red', fontweight="normal", alpha=1.0,
                     bbox=dict(boxstyle='square', edgecolor='black', fc='white', alpha=0.0))
Axes_obj_04.annotate(text=" ? ",
                     xy=(RegC.at["AS2", "m"], 1.5),
                     ha='center', va='center', size=13.0549, color='blue', fontweight="normal", alpha=1.0,
                     bbox=dict(boxstyle='square', edgecolor='black', fc='white', alpha=0.0))
Axes_obj_04.annotate(text=" ? ",
                     xy=(RegC.at["AS2", "m"], -1.5),
                     ha='center', va='center', size=13.0549, color='blue', fontweight="normal", alpha=1.0,
                     bbox=dict(boxstyle='square', edgecolor='black', fc='white', alpha=0.0))
Axes_obj_04.annotate(text=" ? ",
                     xy=(RegC.at["AS2", "m"]+2.75, -2.375),
                     ha='center', va='center', size=13.0549, color='blue', fontweight="normal", alpha=1.0,
                     bbox=dict(boxstyle='square', edgecolor='black', fc='white', alpha=0.0))

x = np.linspace(Para.at["AU1", "x1"], Para.at["AU1", "x2"], 100)
y = Para.at["AU1", "a"]*(x - Para.at["AU1", "m"])**2 + Para.at["AU1", "c"]
Axes_obj_04.plot(x, y, color="red", linestyle="dashed", linewidth=1.0)

x = np.linspace(Para.at["AL1", "x1"], Para.at["AL1", "x2"], 100)
y = Para.at["AL1", "a"]*(x - Para.at["AL1", "m"])**2 + Para.at["AL1", "c"]
Axes_obj_04.plot(x, y, color="red", linestyle="dashed", linewidth=1.0)

x = np.linspace(Para.at["AU2", "x1"], Para.at["AU2", "x2"], 100)
y = Para.at["AU2", "a"]*(x - Para.at["AU2", "m"])**2 + Para.at["AU2", "c"]
Axes_obj_04.plot(x, y, color="blue", linestyle="dashed", linewidth=1.0)

x = np.linspace(Para.at["AL2", "x1"], Para.at["AL2", "x2"], 100)
y = Para.at["AL2", "a"]*(x - Para.at["AL2", "m"])**2 + Para.at["AL2", "c"]
Axes_obj_04.plot(x, y, color="blue", linestyle="dashed", linewidth=1.0)

x = np.linspace(0.00, 20.0, 100)
y = (RegC.at["AS1", "k1"]*RegC.at["AS1", "l"]*np.exp(-RegC.at["AS1", "l"]*(x - RegC.at["AS1", "m"])))/\
    (np.exp(-RegC.at["AS1", "l"]*(x - RegC.at["AS1", "m"]))+1)**2
Axes_obj_04.plot(x, y*0.5-2.75, color="red", linewidth=1.6318)

x = np.linspace(20.0, 40.0, 100)
y = (RegC.at["AS2", "k1"]*RegC.at["AS2", "l"]*np.exp(-RegC.at["AS2", "l"]*(x - RegC.at["AS2", "m"])))/\
    (np.exp(-RegC.at["AS2", "l"]*(x - RegC.at["AS2", "m"]))+1)**2
Axes_obj_04.plot(x, y*0.5-2.75, color="blue", linewidth=1.6318)

############################################################################################################
# "(e) Diabetes (C.I.)":
############################################################################################################
Axes_obj_05.set_title("Diabetes (LN(C.I.) of OR)", fontweight="normal", fontsize=11.4230)
Axes_obj_05.set_xlabel("Difference, Age"); Axes_obj_05.set_ylabel("log, C.I. of OR")
Axes_obj_05.set_xlim(-2.5, 42.5); Axes_obj_05.set_ylim(-2.75, 2.25)
Axes_obj_05.set_xticks([0, 5, 10, 15, 20, 25, 30, 35, 40])
Axes_obj_05.set_yticks([-2.0, -1.0, 0.0, 1.0, 2.0])
Axes_obj_05.set_xticklabels([0, 5, 10, 15, 20, 25, 30, 35, 40])
Axes_obj_05.set_yticklabels([-2.0, -1.0, 0.0, 1.0, 2.0])

for i in range(0, 9):
    Axes_obj_05.vlines(x=0.0+5.0*i, ymin=-2.0, ymax=2.0, color="gray", linestyle="dotted", linewidth=0.5)
for i in range(0, 9):
    Axes_obj_05.hlines(y=-2.0 + 0.5*i, xmin=0.0, xmax=40.0, color="gray", linestyle="dotted", linewidth=0.5)
Axes_obj_05.hlines(y=0.0, xmin=0.0, xmax=40.0, color="black", linestyle='solid', linewidth=1.0)
#Axes_obj_05.vlines(x=20.0, ymin=-2.0, ymax=2.0, color="black", linestyle='solid', linewidth=1.0)

for i in range(0, len(AnalysisDataSet_B["Age (Diff)"])):
    if AnalysisDataSet_B["Frag3"][i] == 1:
        Axes_obj_05.scatter(AnalysisDataSet_B["Age (Diff)"][i], 0.0,
                            color="black", s=50, zorder=3)
        Axes_obj_05.scatter(AnalysisDataSet_B["Age (Diff)"][i], AnalysisDataSet_B["LN(U)diff"][i],
                            color="black", marker='_', s=50, zorder=3)
        Axes_obj_05.scatter(AnalysisDataSet_B["Age (Diff)"][i], AnalysisDataSet_B["LN(L)diff"][i],
                            color="black", marker='_', s=50, zorder=3)
        Axes_obj_05.vlines(x=AnalysisDataSet_B["Age (Diff)"][i],
                           ymin=AnalysisDataSet_B["LN(L)diff"][i], ymax=AnalysisDataSet_B["LN(U)diff"][i],
                           color="black", linestyle='solid', linewidth=1.6318, zorder=3)
    if AnalysisDataSet_B["Frag3"][i] == 2:
        Axes_obj_05.scatter(AnalysisDataSet_B["Age (Diff)"][i], 0.0,
                            color="black", s=50, zorder=3)
        Axes_obj_05.scatter(AnalysisDataSet_B["Age (Diff)"][i], AnalysisDataSet_B["LN(U)diff"][i],
                            color="black", marker='_', s=50, zorder=3)
        Axes_obj_05.scatter(AnalysisDataSet_B["Age (Diff)"][i], AnalysisDataSet_B["LN(L)diff"][i],
                            color="black", marker='_', s=50, zorder=3)
        Axes_obj_05.vlines(x=AnalysisDataSet_B["Age (Diff)"][i],
                           ymin=AnalysisDataSet_B["LN(L)diff"][i], ymax=AnalysisDataSet_B["LN(U)diff"][i],
                           color="black", linestyle='solid', linewidth=1.6318, zorder=3)

Axes_obj_05.text(AnalysisDataSet_B["Age (Diff)"][0], AnalysisDataSet_B["LN(U)diff"][0] + 0.25,
                 AnalysisDataSet_B["Fig. 5"][0], ha="center", va="center",  size=9.7912, color="black", fontweight="normal")
Axes_obj_05.text(AnalysisDataSet_B["Age (Diff)"][1], AnalysisDataSet_B["LN(U)diff"][1] + 0.25,
                 AnalysisDataSet_B["Fig. 5"][1], ha="center", va="center",  size=9.7912, color="black", fontweight="normal")
Axes_obj_05.text(AnalysisDataSet_B["Age (Diff)"][2], AnalysisDataSet_B["LN(U)diff"][2] + 0.25,
                 AnalysisDataSet_B["Fig. 5"][2], ha="center", va="center",  size=9.7912, color="black", fontweight="normal")
Axes_obj_05.text(AnalysisDataSet_B["Age (Diff)"][3], AnalysisDataSet_B["LN(U)diff"][3] + 0.375,
                 AnalysisDataSet_B["Fig. 5"][3], ha="center", va="center",  size=9.7912, color="black", fontweight="normal")
#Axes_obj_05.text(AnalysisDataSet_B["Age (Diff)"][4], AnalysisDataSet_B["LN(U)diff"][4] + 0.25,
#                 AnalysisDataSet_B["Fig. 5"][4], ha="center", va="center",  size=9.7912, color="black", fontweight="normal")

Axes_obj_05.annotate(text=" ? ",
                     xy=(RegC.at["BS1", "m"], 1.625),
                     ha='center', va='center', size=13.0549, color='red', fontweight="normal", alpha=1.0,
                     bbox=dict(boxstyle='square', edgecolor='black', fc='white', alpha=0.0))
Axes_obj_05.annotate(text=" ? ",
                     xy=(RegC.at["BS1", "m"], -1.625),
                     ha='center', va='center', size=13.0549, color='red', fontweight="normal", alpha=1.0,
                     bbox=dict(boxstyle='square', edgecolor='black', fc='white', alpha=0.0))
Axes_obj_05.annotate(text=" ? ",
                     xy=(RegC.at["BS1", "m"]+2.75, -2.375),
                     ha='center', va='center', size=13.0549, color='red', fontweight="normal", alpha=1.0,
                     bbox=dict(boxstyle='square', edgecolor='black', fc='white', alpha=0.0))
Axes_obj_05.annotate(text=" ? ",
                     xy=(RegC.at["BS2", "m"], 1.625),
                     ha='center', va='center', size=13.0549, color='blue', fontweight="normal", alpha=1.0,
                     bbox=dict(boxstyle='square', edgecolor='black', fc='white', alpha=0.0))
Axes_obj_05.annotate(text=" ? ",
                     xy=(RegC.at["BS2", "m"], -1.625),
                     ha='center', va='center', size=13.0549, color='blue', fontweight="normal", alpha=1.0,
                     bbox=dict(boxstyle='square', edgecolor='black', fc='white', alpha=0.0))
Axes_obj_05.annotate(text=" ? ",
                     xy=(RegC.at["BS2", "m"]+2.75, -2.375),
                     ha='center', va='center', size=13.0549, color='blue', fontweight="normal", alpha=1.0,
                     bbox=dict(boxstyle='square', edgecolor='black', fc='white', alpha=0.0))

x = np.linspace(Para.at["BU1", "x1"], Para.at["BU1", "x2"], 100)
y = Para.at["BU1", "a"]*(x - Para.at["BU1", "m"])**2 + Para.at["BU1", "c"]
Axes_obj_05.plot(x, y, color="red", linestyle="dashed", linewidth=1.0)

x = np.linspace(Para.at["BL1", "x1"], Para.at["BL1", "x2"], 100)
y = Para.at["BL1", "a"]*(x - Para.at["BL1", "m"])**2 + Para.at["BL1", "c"]
Axes_obj_05.plot(x, y, color="red", linestyle="dashed", linewidth=1.0)

x = np.linspace(Para.at["BU2", "x1"], Para.at["BU2", "x2"], 100)
y = Para.at["BU2", "a"]*(x - Para.at["BU2", "m"])**2 + Para.at["BU2", "c"]
Axes_obj_05.plot(x, y, color="blue", linestyle="dashed", linewidth=1.0)

x = np.linspace(Para.at["BL2", "x1"], Para.at["BL2", "x2"], 100)
y = Para.at["BL2", "a"]*(x - Para.at["BL2", "m"])**2 + Para.at["BL2", "c"]
Axes_obj_05.plot(x, y, color="blue", linestyle="dashed", linewidth=1.0)

x = np.linspace(0.00, 20.0, 100)
y = (RegC.at["BS1", "k1"]*RegC.at["BS1", "l"]*np.exp(-RegC.at["BS1", "l"]*(x - RegC.at["BS1", "m"])))/\
    (np.exp(-RegC.at["BS1", "l"]*(x - RegC.at["BS1", "m"]))+1)**2
Axes_obj_05.plot(x, y*0.5-2.75, color="red", linewidth=1.6318)

x = np.linspace(20.0, 40.0, 100)
y = (RegC.at["BS2", "k1"]*RegC.at["BS2", "l"]*np.exp(-RegC.at["BS2", "l"]*(x - RegC.at["BS2", "m"])))/\
    (np.exp(-RegC.at["BS2", "l"]*(x - RegC.at["BS2", "m"]))+1)**2
Axes_obj_05.plot(x, y*0.5-2.75, color="blue", linewidth=1.6318)
############################################################################################################
# "(f) CVD (C.I.)":
############################################################################################################
Axes_obj_06.set_title("CVD (LN(C.I.) of OR)", fontweight="normal", fontsize=11.4230)
Axes_obj_06.set_xlabel("Difference, Age"); Axes_obj_06.set_ylabel("log, C.I. of OR")
Axes_obj_06.set_xlim(-2.5, 42.5); Axes_obj_06.set_ylim(-3.75, 3.25)
Axes_obj_06.set_xticks([0, 5, 10, 15, 20, 25, 30, 35, 40])
Axes_obj_06.set_yticks([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
Axes_obj_06.set_xticklabels([0, 5, 10, 15, 20, 25, 30, 35, 40])
Axes_obj_06.set_yticklabels([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])

for i in range(0, 9):
    Axes_obj_06.vlines(x=0.0+5.0*i, ymin=-3.0, ymax=3.0, color="gray", linestyle="dotted", linewidth=0.5)
for i in range(0, 13):
    Axes_obj_06.hlines(y=-3.0 + 0.5*i, xmin=0.0, xmax=40.0, color="gray", linestyle="dotted", linewidth=0.5)
Axes_obj_06.hlines(y=0.0, xmin=0.0, xmax=40.0, color="black", linestyle='solid', linewidth=1.0)
#Axes_obj_06.vlines(x=20.0, ymin=-3.0, ymax=3.0, color="black", linestyle='solid', linewidth=1.0)

for i in range(0, len(AnalysisDataSet_C["Age (Diff)"])):
    if AnalysisDataSet_C["Frag3"][i] == 1:
        Axes_obj_06.scatter(AnalysisDataSet_C["Age (Diff)"][i], 0.0,
                            color="black", s=50, zorder=3)
        Axes_obj_06.scatter(AnalysisDataSet_C["Age (Diff)"][i], AnalysisDataSet_C["LN(U)diff"][i],
                            color="black", marker='_', s=50, zorder=3)
        Axes_obj_06.scatter(AnalysisDataSet_C["Age (Diff)"][i], AnalysisDataSet_C["LN(L)diff"][i],
                            color="black", marker='_', s=50, zorder=3)
        Axes_obj_06.vlines(x=AnalysisDataSet_C["Age (Diff)"][i],
                           ymin=AnalysisDataSet_C["LN(L)diff"][i], ymax=AnalysisDataSet_C["LN(U)diff"][i],
                           color="black", linestyle='solid', linewidth=1.6318, zorder=3)
    if AnalysisDataSet_C["Frag3"][i] == 2:
        Axes_obj_06.scatter(AnalysisDataSet_C["Age (Diff)"][i], 0.0,
                            color="black", s=50, zorder=3)
        Axes_obj_06.scatter(AnalysisDataSet_C["Age (Diff)"][i], AnalysisDataSet_C["LN(U)diff"][i],
                            color="black", marker='_', s=50, zorder=3)
        Axes_obj_06.scatter(AnalysisDataSet_C["Age (Diff)"][i], AnalysisDataSet_C["LN(L)diff"][i],
                            color="black", marker='_', s=50, zorder=3)
        Axes_obj_06.vlines(x=AnalysisDataSet_C["Age (Diff)"][i],
                           ymin=AnalysisDataSet_C["LN(L)diff"][i], ymax=AnalysisDataSet_C["LN(U)diff"][i],
                           color="black", linestyle='solid', linewidth=1.6318, zorder=3)

Axes_obj_06.text(AnalysisDataSet_C["Age (Diff)"][0], AnalysisDataSet_C["LN(U)diff"][0] + 0.375,
                 AnalysisDataSet_C["Fig. 5"][0], ha="center", va="center",  size=9.7912, color="black", fontweight="normal")
Axes_obj_06.text(AnalysisDataSet_C["Age (Diff)"][1], AnalysisDataSet_C["LN(U)diff"][1] + 0.375,
                 AnalysisDataSet_C["Fig. 5"][1], ha="center", va="center",  size=9.7912, color="black", fontweight="normal")
Axes_obj_06.text(AnalysisDataSet_C["Age (Diff)"][2] - 0.375, AnalysisDataSet_C["LN(U)diff"][2] + 0.375,
                 AnalysisDataSet_C["Fig. 5"][2], ha="center", va="center",  size=9.7912, color="black", fontweight="normal")
Axes_obj_06.text(AnalysisDataSet_C["Age (Diff)"][3], AnalysisDataSet_C["LN(U)diff"][3] + 0.375,
                 AnalysisDataSet_C["Fig. 5"][3], ha="center", va="center",  size=9.7912, color="black", fontweight="normal")
Axes_obj_06.text(AnalysisDataSet_C["Age (Diff)"][4] + 0.375, AnalysisDataSet_C["LN(U)diff"][4] + 0.375,
                 AnalysisDataSet_C["Fig. 5"][4], ha="center", va="center",  size=9.7912, color="black", fontweight="normal")
#Axes_obj_06.text(AnalysisDataSet_C["Age (Diff)"][5], AnalysisDataSet_C["LN(U)diff"][5] + 0.375,
#                 AnalysisDataSet_C["Fig. 5"][5], ha="center", va="center",  size=9.7912, color="black", fontweight="normal")

Axes_obj_06.annotate(text=" ? ",
                     xy=(RegC.at["CS1", "m"], 2.0),
                     ha='center', va='center', size=13.0549, color='red', fontweight="normal", alpha=1.0,
                     bbox=dict(boxstyle='square', edgecolor='black', fc='white', alpha=0.0))
Axes_obj_06.annotate(text=" ? ",
                     xy=(RegC.at["CS1", "m"], -2.0),
                     ha='center', va='center', size=13.0549, color='red', fontweight="normal", alpha=1.0,
                     bbox=dict(boxstyle='square', edgecolor='black', fc='white', alpha=0.0))
Axes_obj_06.annotate(text=" ? ",
                     xy=(RegC.at["CS1", "m"]+3.75, -3.375),
                     ha='center', va='center', size=13.0549, color='red', fontweight="normal", alpha=1.0,
                     bbox=dict(boxstyle='square', edgecolor='black', fc='white', alpha=0.0))
Axes_obj_06.annotate(text=" ? ",
                     xy=(RegC.at["BS2", "m"], 2.0),
                     ha='center', va='center', size=13.0549, color='blue', fontweight="normal", alpha=1.0,
                     bbox=dict(boxstyle='square', edgecolor='black', fc='white', alpha=0.0))
Axes_obj_06.annotate(text=" ? ",
                     xy=(RegC.at["BS2", "m"], -2.0),
                     ha='center', va='center', size=13.0549, color='blue', fontweight="normal", alpha=1.0,
                     bbox=dict(boxstyle='square', edgecolor='black', fc='white', alpha=0.0))
Axes_obj_06.annotate(text=" ? ",
                     xy=(RegC.at["BS2", "m"]+3.75, -3.375),
                     ha='center', va='center', size=13.0549, color='blue', fontweight="normal", alpha=1.0,
                     bbox=dict(boxstyle='square', edgecolor='black', fc='white', alpha=0.0))

x = np.linspace(Para.at["CU1", "x1"], Para.at["CU1", "x2"], 100)
y = Para.at["CU1", "a"]*(x - Para.at["CU1", "m"])**2 + Para.at["CU1", "c"]
Axes_obj_06.plot(x, y, color="red", linestyle="dashed", linewidth=1.0)

x = np.linspace(Para.at["CL1", "x1"], Para.at["CL1", "x2"], 100)
y = Para.at["CL1", "a"]*(x - Para.at["CL1", "m"])**2 + Para.at["CL1", "c"]
Axes_obj_06.plot(x, y, color="red", linestyle="dashed", linewidth=1.0)

x = np.linspace(Para.at["CU2", "x1"], Para.at["CU2", "x2"], 100)
y = Para.at["CU2", "a"]*(x - Para.at["BU2", "m"])**2 + Para.at["CU2", "c"]
Axes_obj_06.plot(x, y, color="blue", linestyle="dashed", linewidth=1.0)

x = np.linspace(Para.at["CL2", "x1"], Para.at["CL2", "x2"], 100)
y = Para.at["CL2", "a"]*(x - Para.at["BL2", "m"])**2 + Para.at["CL2", "c"]
Axes_obj_06.plot(x, y, color="blue", linestyle="dashed", linewidth=1.0)

x = np.linspace(0.00, 20.0, 100)
y = (RegC.at["CS1", "k1"]*RegC.at["CS1", "l"]*np.exp(-RegC.at["CS1", "l"]*(x - RegC.at["CS1", "m"])))/\
    (np.exp(-RegC.at["CS1", "l"]*(x - RegC.at["CS1", "m"]))+1)**2
Axes_obj_06.plot(x, y*(1/3)-3.75, color="red", linewidth=1.6318)

x = np.linspace(20.0, 40.0, 100)
y = (RegC.at["BS2", "k1"]*RegC.at["BS2", "l"]*np.exp(-RegC.at["BS2", "l"]*(x - RegC.at["BS2", "m"])))/\
    (np.exp(-RegC.at["BS2", "l"]*(x - RegC.at["BS2", "m"]))+1)**2
Axes_obj_06.plot(x, y*(1/3)-3.75, color="blue", linewidth=1.6318)
########################################################################################################################
############################################################################################################
# "(g) Hypertension (Estimated Curve)":
############################################################################################################
Axes_obj_07.set_title("Hypertension (OR + LN(C.I.))", fontweight="normal", fontsize=11.4230)
Axes_obj_07.set_xlabel("Difference, Age"); Axes_obj_07.set_ylabel("OR")
Axes_obj_07.set_xlim(-2.5, 42.5); Axes_obj_07.set_ylim(-2.0, 8.0)
Axes_obj_07.set_xticks([0, 5, 10, 15, 20, 25, 30, 35, 40])
Axes_obj_07.set_yticks([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
Axes_obj_07.set_xticklabels([0, 5, 10, 15, 20, 25, 30, 35, 40])
Axes_obj_07.set_yticklabels([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

for i in range(0, 9):
    Axes_obj_07.vlines(x=0.0+5.0*i, ymin=-1.0, ymax=7.0, color="gray", linestyle="dotted", linewidth=0.5)
for i in range(0, 17):
    Axes_obj_07.hlines(y=-1.0 + 0.5*i, xmin=0.0, xmax=40.0, color="gray", linestyle="dotted", linewidth=0.5)

Axes_obj_07.hlines(y=1.0, xmin=0.0, xmax=40.0, color="black", linestyle='solid', linewidth=1.0)
Axes_obj_07.scatter([0.0], [1.0], color="black", facecolor="black", s=50, zorder=5)

for i in range(0, len(AnalysisDataSet_A["Age (Diff)"])):
    if AnalysisDataSet_A["Frag3"][i] == 1:
        Axes_obj_07.scatter(AnalysisDataSet_A["Age (Diff)"][i], AnalysisDataSet_A["OR"][i],
                            color="red", s=50)
        Axes_obj_07.scatter(AnalysisDataSet_A["Age (Diff)"][i], AnalysisDataSet_A["OR + Abs( LN(U) - LN(OR) )"][i],
                            color="red", marker='_', s=50)
        Axes_obj_07.scatter(AnalysisDataSet_A["Age (Diff)"][i], AnalysisDataSet_A["OR - Abs( LN(L) - LN(OR) )"][i],
                            color="red", marker='_', s=50)
        Axes_obj_07.vlines(x=AnalysisDataSet_A["Age (Diff)"][i],
                           ymin=AnalysisDataSet_A["OR - Abs( LN(L) - LN(OR) )"][i],
                           ymax=AnalysisDataSet_A["OR + Abs( LN(U) - LN(OR) )"][i],
                           color="red", linestyle='solid', linewidth=1.6318)

    if AnalysisDataSet_A["Frag3"][i] == 2:
        Axes_obj_07.scatter(AnalysisDataSet_A["Age (Diff)"][i], AnalysisDataSet_A["OR"][i],
                            color="blue", s=50)
        Axes_obj_07.scatter(AnalysisDataSet_A["Age (Diff)"][i], AnalysisDataSet_A["OR + Abs( LN(U) - LN(OR) )"][i],
                            color="blue", marker='_', s=50)
        Axes_obj_07.scatter(AnalysisDataSet_A["Age (Diff)"][i], AnalysisDataSet_A["OR - Abs( LN(L) - LN(OR) )"][i],
                            color="blue", marker='_', s=50)
        Axes_obj_07.vlines(x=AnalysisDataSet_A["Age (Diff)"][i],
                           ymin=AnalysisDataSet_A["OR - Abs( LN(L) - LN(OR) )"][i],
                           ymax=AnalysisDataSet_A["OR + Abs( LN(U) - LN(OR) )"][i],
                           color="blue", linestyle='solid', linewidth=1.6318)

Axes_obj_07.text(AnalysisDataSet_A["Age (Diff)"][0], AnalysisDataSet_A["OR + Abs( LN(U) - LN(OR) )"][0] + 0.375+0.25,
                 AnalysisDataSet_A["Fig. 5"][0], ha="center", va="center",  size=9.7912, color="black", fontweight="normal")
Axes_obj_07.text(AnalysisDataSet_A["Age (Diff)"][1], AnalysisDataSet_A["OR + Abs( LN(U) - LN(OR) )"][1] + 0.375+0.25,
                 AnalysisDataSet_A["Fig. 5"][1], ha="center", va="center",  size=9.7912, color="black", fontweight="normal")
Axes_obj_07.text(AnalysisDataSet_A["Age (Diff)"][2], AnalysisDataSet_A["OR + Abs( LN(U) - LN(OR) )"][2] + 0.375+0.25,
                 AnalysisDataSet_A["Fig. 5"][2], ha="center", va="center",  size=9.7912, color="black", fontweight="normal")
Axes_obj_07.text(AnalysisDataSet_A["Age (Diff)"][3], AnalysisDataSet_A["OR + Abs( LN(U) - LN(OR) )"][3] + 0.375+0.5,
                 AnalysisDataSet_A["Fig. 5"][3], ha="center", va="center",  size=9.7912, color="black", fontweight="normal")
#Axes_obj_07.text(AnalysisDataSet_A["Age (Diff)"][4], AnalysisDataSet_A["LN(U)"][4] + 0.375,
#                 AnalysisDataSet_A["Fig. 5"][4], ha="center", va="center",  size=9.7912, color="black", fontweight="normal")

xs1 = np.linspace(RegC.at["AS1", "x1"], RegC.at["AS1", "x2"], 100)
ys1 = RegC.at["AS1", "k1"] / (1 + np.exp(-RegC.at["AS1", "l"] * (xs1 - RegC.at["AS1", "m"]))) + RegC.at["AS1", "k2"]
Axes_obj_07.plot(xs1, ys1, color="red", linewidth=1.6318)

xu1 = np.linspace(RegC.at["AU1", "x1"], RegC.at["AU1", "x2"], 100)
yu1 = RegC.at["AU1", "k1"] / (1 + np.exp(-RegC.at["AU1", "l"] * (xu1 - RegC.at["AU1", "m"]))) + RegC.at["AU1", "k2"]\
    + (abs(RegC.at["AU1", "a"])*(xu1 - RegC.at["AU1", "m"])**2 + RegC.at["AU1", "c"])
Axes_obj_07.plot(xu1, yu1, color="red", linewidth=0.5)

xl1 = np.linspace(RegC.at["AL1", "x1"], RegC.at["AL1", "x2"], 100)
yl1 = RegC.at["AL1", "k1"] / (1 + np.exp(-RegC.at["AL1", "l"] * (xl1 - RegC.at["AL1", "m"]))) + RegC.at["AL1", "k2"]\
    + (-abs(RegC.at["AL1", "a"])*(xl1 - RegC.at["AL1", "m"])**2 + RegC.at["AL1", "c"])
Axes_obj_07.plot(xl1, yl1, color="red", linewidth=0.5)

xs2 = np.linspace(RegC.at["AS2", "x1"], RegC.at["AS2", "x2"], 100)
ys2 = RegC.at["AS2", "k1"] / (1 + np.exp(-RegC.at["AS2", "l"] * (xs2 - RegC.at["AS2", "m"]))) + RegC.at["AS2", "k2"]
Axes_obj_07.plot(xs2, ys2, color="blue", linewidth=1.6318)

xu2 = np.linspace(RegC.at["AU2", "x1"], RegC.at["AU2", "x2"], 100)
yu2 = RegC.at["AU2", "k1"] / (1 + np.exp(-RegC.at["AU2", "l"] * (xu2 - RegC.at["AU2", "m"]))) + RegC.at["AU2", "k2"]\
    + (abs(RegC.at["AU2", "a"])*(xu2 - RegC.at["AU2", "m"])**2 + RegC.at["AU2", "c"])
Axes_obj_07.plot(xu2, yu2, color="blue", linewidth=0.5)

xl2 = np.linspace(RegC.at["AL2", "x1"], RegC.at["AL2", "x2"], 100)
yl2 = RegC.at["AL2", "k1"] / (1 + np.exp(-RegC.at["AL2", "l"] * (xl2 - RegC.at["AL2", "m"]))) + RegC.at["AL2", "k2"]\
    + (-abs(RegC.at["AL2", "a"])*(xl2 - RegC.at["AL2", "m"])**2 + RegC.at["AL2", "c"])
Axes_obj_07.plot(xl2, yl2, color="blue", linewidth=0.5)

xd1 = np.linspace(0.00, 20.0, 100)
yd1 = (RegC.at["AS1", "k1"]*RegC.at["AS1", "l"]*np.exp(-RegC.at["AS1", "l"]*(xd1 - RegC.at["AS1", "m"])))/\
    (np.exp(-RegC.at["AS1", "l"]*(xd1 - RegC.at["AS1", "m"]))+1)**2
Axes_obj_07.plot(xd1, yd1-2.0, color="red", linewidth=1.6318)

xd2 = np.linspace(20.0, 40.0, 100)
yd2 = (RegC.at["AS2", "k1"]*RegC.at["AS2", "l"]*np.exp(-RegC.at["AS2", "l"]*(xd2 - RegC.at["AS2", "m"])))/\
    (np.exp(-RegC.at["AS2", "l"]*(xd2 - RegC.at["AS2", "m"]))+1)**2
Axes_obj_07.plot(xd2, yd2-2.0, color="blue", linewidth=1.6318)

Axes_obj_07.fill_between(xu2, yu2, yl2, facecolor="blue", alpha=0.05)
Axes_obj_07.fill_between(xu1, yu1, yl1, facecolor="red", alpha=0.05)
Axes_obj_07.fill_between(xu2, yu2, yl2, facecolor="blue", alpha=0.05)
Axes_obj_07.fill_between(xd1, yd1-2.0, -2.0*np.ones(100), facecolor="red", alpha=0.05)
Axes_obj_07.fill_between(xd2, yd2-2.0, -2.0*np.ones(100), facecolor="blue", alpha=0.05)
############################################################################################################
# "(h) Diabetes (Estimated Curve)":
############################################################################################################
Axes_obj_08.set_title("Diabetes (OR + LN(C.I.))", fontweight="normal", fontsize=11.4230)
Axes_obj_08.set_xlabel("Difference, Age"); Axes_obj_08.set_ylabel("OR")
Axes_obj_08.set_xlim(-2.5, 42.5); Axes_obj_08.set_ylim(-2.0, 7.5)
Axes_obj_08.set_xticks([0, 5, 10, 15, 20, 25, 30, 35, 40])
Axes_obj_08.set_yticks([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
Axes_obj_08.set_xticklabels([0, 5, 10, 15, 20, 25, 30, 35, 40])
Axes_obj_08.set_yticklabels([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

for i in range(0, 9):
    Axes_obj_08.vlines(x=0.0+5.0*i, ymin=-1.0, ymax=6.5, color="gray", linestyle="dotted", linewidth=0.5)
for i in range(0, 16):
    Axes_obj_08.hlines(y=-1.0 + 0.5*i, xmin=0.0, xmax=40.0, color="gray", linestyle="dotted", linewidth=0.5)

Axes_obj_08.hlines(y=1.0, xmin=0.0, xmax=40.0, color="black", linestyle='solid', linewidth=1.0)

Axes_obj_08.scatter([0.0], [1.0], color="black", facecolor="black", s=50, zorder=5)

for i in range(0, len(AnalysisDataSet_B["Age (Diff)"])):
    if AnalysisDataSet_B["Frag3"][i] == 1:
        Axes_obj_08.scatter(AnalysisDataSet_B["Age (Diff)"][i], AnalysisDataSet_B["OR"][i],
                            color="red", s=50)
        Axes_obj_08.scatter(AnalysisDataSet_B["Age (Diff)"][i], AnalysisDataSet_B["OR + Abs( LN(U) - LN(OR) )"][i],
                            color="red", marker='_', s=50)
        Axes_obj_08.scatter(AnalysisDataSet_B["Age (Diff)"][i], AnalysisDataSet_B["OR - Abs( LN(L) - LN(OR) )"][i],
                            color="red", marker='_', s=50)
        Axes_obj_08.vlines(x=AnalysisDataSet_B["Age (Diff)"][i],
                           ymin=AnalysisDataSet_B["OR - Abs( LN(L) - LN(OR) )"][i],
                           ymax=AnalysisDataSet_B["OR + Abs( LN(U) - LN(OR) )"][i],
                           color="red", linestyle='solid', linewidth=1.6318)

    if AnalysisDataSet_B["Frag3"][i] == 2:
        Axes_obj_08.scatter(AnalysisDataSet_B["Age (Diff)"][i], AnalysisDataSet_B["OR"][i],
                            color="blue", s=50)
        Axes_obj_08.scatter(AnalysisDataSet_B["Age (Diff)"][i], AnalysisDataSet_B["OR + Abs( LN(U) - LN(OR) )"][i],
                            color="blue", marker='_', s=50)
        Axes_obj_08.scatter(AnalysisDataSet_B["Age (Diff)"][i], AnalysisDataSet_B["OR - Abs( LN(L) - LN(OR) )"][i],
                            color="blue", marker='_', s=50)
        Axes_obj_08.vlines(x=AnalysisDataSet_B["Age (Diff)"][i],
                           ymin=AnalysisDataSet_B["OR - Abs( LN(L) - LN(OR) )"][i],
                           ymax=AnalysisDataSet_B["OR + Abs( LN(U) - LN(OR) )"][i],
                           color="blue", linestyle='solid', linewidth=1.6318)

Axes_obj_08.text(AnalysisDataSet_B["Age (Diff)"][0], AnalysisDataSet_B["OR + Abs( LN(U) - LN(OR) )"][0] + 0.375+0.25,
                 AnalysisDataSet_B["Fig. 5"][0], ha="center", va="center",  size=9.7912, color="black", fontweight="normal")
Axes_obj_08.text(AnalysisDataSet_B["Age (Diff)"][1], AnalysisDataSet_B["OR + Abs( LN(U) - LN(OR) )"][1] + 0.375+0.25,
                 AnalysisDataSet_B["Fig. 5"][1], ha="center", va="center",  size=9.7912, color="black", fontweight="normal")
Axes_obj_08.text(AnalysisDataSet_B["Age (Diff)"][2], AnalysisDataSet_B["OR + Abs( LN(U) - LN(OR) )"][2] + 0.375+0.25,
                 AnalysisDataSet_B["Fig. 5"][2], ha="center", va="center",  size=9.7912, color="black", fontweight="normal")
Axes_obj_08.text(AnalysisDataSet_B["Age (Diff)"][3], AnalysisDataSet_B["OR + Abs( LN(U) - LN(OR) )"][3] + 0.375+0.25,
                 AnalysisDataSet_B["Fig. 5"][3], ha="center", va="center",  size=9.7912, color="black", fontweight="normal")
#Axes_obj_08.text(AnalysisDataSet_B["Age (Diff)"][4], AnalysisDataSet_B["LN(U)"][4] + 0.375,
#                 AnalysisDataSet_B["Fig. 5"][4], ha="center", va="center",  size=9.7912, color="black", fontweight="normal")

xs1 = np.linspace(RegC.at["BS1", "x1"], RegC.at["BS1", "x2"], 100)
ys1 = RegC.at["BS1", "k1"] / (1 + np.exp(-RegC.at["BS1", "l"] * (xs1 - RegC.at["BS1", "m"]))) + RegC.at["BS1", "k2"]
Axes_obj_08.plot(xs1, ys1, color="red", linewidth=1.6318)

xu1 = np.linspace(RegC.at["BU1", "x1"], RegC.at["BU1", "x2"], 100)
yu1 = RegC.at["BU1", "k1"] / (1 + np.exp(-RegC.at["BU1", "l"] * (xu1 - RegC.at["BU1", "m"]))) + RegC.at["BU1", "k2"]\
    + (abs(RegC.at["BU1", "a"])*(xu1 - RegC.at["BU1", "m"])**2 + RegC.at["BU1", "c"])
Axes_obj_08.plot(xu1, yu1, color="red", linewidth=0.5)

xl1 = np.linspace(RegC.at["BL1", "x1"], RegC.at["BL1", "x2"], 100)
yl1 = RegC.at["BL1", "k1"] / (1 + np.exp(-RegC.at["BL1", "l"] * (xl1 - RegC.at["BL1", "m"]))) + RegC.at["BL1", "k2"]\
    + (-abs(RegC.at["BL1", "a"])*(xl1 - RegC.at["BL1", "m"])**2 + RegC.at["BL1", "c"])
Axes_obj_08.plot(xl1, yl1, color="red", linewidth=0.5)

xs2 = np.linspace(RegC.at["BS2", "x1"], RegC.at["BS2", "x2"], 100)
ys2 = RegC.at["BS2", "k1"] / (1 + np.exp(-RegC.at["BS2", "l"] * (xs2 - RegC.at["BS2", "m"]))) + RegC.at["BS2", "k2"]
Axes_obj_08.plot(xs2, ys2, color="blue", linewidth=1.6318)

xu2 = np.linspace(RegC.at["BU2", "x1"], RegC.at["BU2", "x2"], 100)
yu2 = RegC.at["BU2", "k1"] / (1 + np.exp(-RegC.at["BU2", "l"] * (xu2 - RegC.at["BU2", "m"]))) + RegC.at["BU2", "k2"]\
    + (abs(RegC.at["BU2", "a"])*(xu2 - RegC.at["BU2", "m"])**2 + RegC.at["BU2", "c"])
Axes_obj_08.plot(xu2, yu2, color="blue", linewidth=0.5)

xl2 = np.linspace(RegC.at["BL2", "x1"], RegC.at["BL2", "x2"], 100)
yl2 = RegC.at["BL2", "k1"] / (1 + np.exp(-RegC.at["BL2", "l"] * (xl2 - RegC.at["BL2", "m"]))) + RegC.at["BL2", "k2"]\
    + (-abs(RegC.at["BL2", "a"])*(xl2 - RegC.at["BL2", "m"])**2 + RegC.at["BL2", "c"])
Axes_obj_08.plot(xl2, yl2, color="blue", linewidth=0.5)

xd1 = np.linspace(0.00, 20.0, 100)
yd1 = (RegC.at["BS1", "k1"]*RegC.at["BS1", "l"]*np.exp(-RegC.at["BS1", "l"]*(xd1 - RegC.at["BS1", "m"])))/\
    (np.exp(-RegC.at["BS1", "l"]*(xd1 - RegC.at["BS1", "m"]))+1)**2
Axes_obj_08.plot(xd1, yd1-2.0, color="red", linewidth=1.6318)

xd2 = np.linspace(20.0, 40.0, 100)
yd2 = (RegC.at["BS2", "k1"]*RegC.at["BS2", "l"]*np.exp(-RegC.at["BS2", "l"]*(xd2 - RegC.at["BS2", "m"])))/\
    (np.exp(-RegC.at["BS2", "l"]*(xd2 - RegC.at["BS2", "m"]))+1)**2
Axes_obj_08.plot(xd2, yd2-2.0, color="blue", linewidth=1.6318)

Axes_obj_08.fill_between(xu2, yu2, yl2, facecolor="blue", alpha=0.05)
Axes_obj_08.fill_between(xu1, yu1, yl1, facecolor="red", alpha=0.05)
Axes_obj_08.fill_between(xu2, yu2, yl2, facecolor="blue", alpha=0.05)
Axes_obj_08.fill_between(xd1, yd1-2.0, -2.5*np.ones(100), facecolor="red", alpha=0.05)
Axes_obj_08.fill_between(xd2, yd2-2.0, -2.5*np.ones(100), facecolor="blue", alpha=0.05)
############################################################################################################
# "(i) CVD (Estimated Curve)":
############################################################################################################
Axes_obj_09.set_title("CVD (OR + LN(C.I.))", fontweight="normal", fontsize=11.4230)
Axes_obj_09.set_xlabel("Difference, Age"); Axes_obj_09.set_ylabel("OR")
Axes_obj_09.set_xlim(-2.5, 42.5); Axes_obj_09.set_ylim(-5.0, 30.0)
Axes_obj_09.set_xticks([0, 5, 10, 15, 20, 25, 30, 35, 40])
Axes_obj_09.set_yticks([0.0, 5.0, 10.0, 15.0, 20.0, 25.0])
Axes_obj_09.set_xticklabels([0, 5, 10, 15, 20, 25, 30, 35, 40])
Axes_obj_09.set_yticklabels([0.0, 5.0, 10.0, 15.0, 20.0, 25.0])

for i in range(0, 9):
    Axes_obj_09.vlines(x=0.0+5.0*i, ymin=-2.5, ymax=27.5, color="gray", linestyle="dotted", linewidth=0.5)
for i in range(0, 13):
    Axes_obj_09.hlines(y=-2.5 + 2.5*i, xmin=0.0, xmax=40.0, color="gray", linestyle="dotted", linewidth=0.5)

Axes_obj_09.hlines(y=1.0, xmin=0.0, xmax=40.0, color="black", linestyle='solid', linewidth=1.0)

Axes_obj_09.scatter([0.0], [1.0], color="black", facecolor="black", s=50, zorder=5)

for i in range(0, len(AnalysisDataSet_C["Age (Diff)"])):
    if AnalysisDataSet_C["Frag3"][i] == 1:
        Axes_obj_09.scatter(AnalysisDataSet_C["Age (Diff)"][i], AnalysisDataSet_C["OR"][i],
                            color="red", s=50)
        Axes_obj_09.scatter(AnalysisDataSet_C["Age (Diff)"][i], AnalysisDataSet_C["OR + Abs( LN(U) - LN(OR) )"][i],
                            color="red", marker='_', s=50)
        Axes_obj_09.scatter(AnalysisDataSet_C["Age (Diff)"][i], AnalysisDataSet_C["OR - Abs( LN(L) - LN(OR) )"][i],
                            color="red", marker='_', s=50)
        Axes_obj_09.vlines(x=AnalysisDataSet_C["Age (Diff)"][i],
                           ymin=AnalysisDataSet_C["OR - Abs( LN(L) - LN(OR) )"][i],
                           ymax=AnalysisDataSet_C["OR + Abs( LN(U) - LN(OR) )"][i],
                           color="red", linestyle='solid', linewidth=1.6318)
    if AnalysisDataSet_C["Frag3"][i] == 2:
        Axes_obj_09.scatter(AnalysisDataSet_C["Age (Diff)"][i], AnalysisDataSet_C["OR"][i],
                            color="blue", s=50)
        Axes_obj_09.scatter(AnalysisDataSet_C["Age (Diff)"][i], AnalysisDataSet_C["OR + Abs( LN(U) - LN(OR) )"][i],
                            color="blue", marker='_', s=50)
        Axes_obj_09.scatter(AnalysisDataSet_C["Age (Diff)"][i], AnalysisDataSet_C["OR - Abs( LN(L) - LN(OR) )"][i],
                            color="blue", marker='_', s=50)
        Axes_obj_09.vlines(x=AnalysisDataSet_C["Age (Diff)"][i],
                           ymin=AnalysisDataSet_C["OR - Abs( LN(L) - LN(OR) )"][i],
                           ymax=AnalysisDataSet_C["OR + Abs( LN(U) - LN(OR) )"][i],
                           color="blue", linestyle='solid', linewidth=1.6318)

Axes_obj_09.text(AnalysisDataSet_C["Age (Diff)"][0], AnalysisDataSet_C["OR + Abs( LN(U) - LN(OR) )"][0] + 0.375+1.25,
                 AnalysisDataSet_C["Fig. 5"][0], ha="center", va="center",  size=9.7912, color="black", fontweight="normal")
Axes_obj_09.text(AnalysisDataSet_C["Age (Diff)"][1], AnalysisDataSet_C["OR + Abs( LN(U) - LN(OR) )"][1] + 0.375+1.25,
                 AnalysisDataSet_C["Fig. 5"][1], ha="center", va="center",  size=9.7912, color="black", fontweight="normal")
Axes_obj_09.text(AnalysisDataSet_C["Age (Diff)"][2]- 0.375, AnalysisDataSet_C["OR + Abs( LN(U) - LN(OR) )"][2] + 0.375+1.25,
                 AnalysisDataSet_C["Fig. 5"][2], ha="center", va="center",  size=9.7912, color="black", fontweight="normal")
Axes_obj_09.text(AnalysisDataSet_C["Age (Diff)"][3], AnalysisDataSet_C["OR + Abs( LN(U) - LN(OR) )"][3] + 0.375+1.25,
                 AnalysisDataSet_C["Fig. 5"][3], ha="center", va="center",  size=9.7912, color="black", fontweight="normal")
Axes_obj_09.text(AnalysisDataSet_C["Age (Diff)"][4], AnalysisDataSet_C["OR + Abs( LN(U) - LN(OR) )"][4] + 0.375+1.25,
                 AnalysisDataSet_C["Fig. 5"][4], ha="center", va="center",  size=9.7912, color="black", fontweight="normal")
#Axes_obj_09.text(AnalysisDataSet_C["Age (Diff)"][5], AnalysisDataSet_C["LN(U)"][5] + 0.375,
#                 AnalysisDataSet_C["Fig. 5"][5], ha="center", va="center",  size=9.7912, color="black", fontweight="normal")

xs1 = np.linspace(RegC.at["CS1", "x1"], RegC.at["CS1", "x2"], 100)
ys1 = RegC.at["CS1", "k1"] / (1 + np.exp(-RegC.at["CS1", "l"] * (xs1 - RegC.at["CS1", "m"]))) + RegC.at["CS1", "k2"]
Axes_obj_09.plot(xs1, ys1, color="red", linewidth=1.6318)

xu1 = np.linspace(RegC.at["CU1", "x1"], RegC.at["CU1", "x2"], 100)
yu1 = RegC.at["CU1", "k1"] / (1 + np.exp(-RegC.at["CU1", "l"] * (xu1 - RegC.at["CU1", "m"]))) + RegC.at["CU1", "k2"]\
    + (abs(RegC.at["CU1", "a"])*(xu1 - RegC.at["CU1", "m"])**2 + RegC.at["CU1", "c"])
#Axes_obj_09.plot(xu1, yu1, color="red", linewidth=0.5)

xl1 = np.linspace(RegC.at["CL1", "x1"], RegC.at["CL1", "x2"], 100)
yl1 = RegC.at["CL1", "k1"] / (1 + np.exp(-RegC.at["CL1", "l"] * (xl1 - RegC.at["CL1", "m"]))) + RegC.at["CL1", "k2"]\
    + (-abs(RegC.at["CL1", "a"])*(xl1 - RegC.at["CL1", "m"])**2 + RegC.at["CL1", "c"])
#Axes_obj_09.plot(xl1, yl1, color="red", linewidth=0.5)

xs2 = np.linspace(RegC.at["CS2", "x1"], RegC.at["CS2", "x2"], 100)
ys2 = RegC.at["CS2", "k1"] / (1 + np.exp(-RegC.at["CS2", "l"] * (xs2 - RegC.at["CS2", "m"]))) + RegC.at["CS2", "k2"]
Axes_obj_09.plot(xs2, ys2, color="blue", linewidth=1.6318)

xu2 = np.linspace(RegC.at["CU2", "x1"], RegC.at["CU2", "x2"], 100)
yu2 = RegC.at["CU2", "k1"] / (1 + np.exp(-RegC.at["CU2", "l"] * (xu2 - RegC.at["CU2", "m"]))) + RegC.at["CU2", "k2"]\
    + (abs(RegC.at["CU2", "a"])*(xu2 - RegC.at["CU2", "m"])**2 + RegC.at["CU2", "c"])
#Axes_obj_09.plot(xu2, yu2, color="blue", linewidth=0.5)

xl2 = np.linspace(RegC.at["CL2", "x1"], RegC.at["CL2", "x2"], 100)
yl2 = RegC.at["CL2", "k1"] / (1 + np.exp(-RegC.at["CL2", "l"] * (xl2 - RegC.at["CL2", "m"]))) + RegC.at["CL2", "k2"]\
    + (-abs(RegC.at["CL2", "a"])*(xl2 - RegC.at["CL2", "m"])**2 + RegC.at["CL2", "c"])
#Axes_obj_09.plot(xl2, yl2, color="blue", linewidth=0.5)

xd1 = np.linspace(0.00, 20.0, 100)
yd1 = (RegC.at["CS1", "k1"]*RegC.at["CS1", "l"]*np.exp(-RegC.at["CS1", "l"]*(xd1 - RegC.at["CS1", "m"])))/\
    (np.exp(-RegC.at["CS1", "l"]*(xd1 - RegC.at["CS1", "m"]))+1)**2
Axes_obj_09.plot(xd1, yd1-5.0, color="red", linewidth=1.6318)

xd2 = np.linspace(0.00, 40.0, 100)
yd2 = (RegC.at["CS2", "k1"]*RegC.at["CS2", "l"]*np.exp(-RegC.at["CS2", "l"]*(xd2 - RegC.at["CS2", "m"])))/\
    (np.exp(-RegC.at["CS2", "l"]*(xd2 - RegC.at["CS2", "m"]))+1)**2
Axes_obj_09.plot(xd2, yd2-5.0, color="blue", linewidth=1.6318)


#Axes_obj_09.fill_between(xu1, yu1, yl1, facecolor="red", alpha=0.05)
#Axes_obj_09.fill_between(xu2, yu2, yl2, facecolor="blue", alpha=0.05)
Axes_obj_09.fill_between(xd1, yd1-5.0, -5.0*np.ones(100), facecolor="red", alpha=0.05)
Axes_obj_09.fill_between(xd2, yd2-5.0, -5.0*np.ones(100), facecolor="blue", alpha=0.05)
############################################################################################################

#Figure_object.tight_layout()
my_path = os.path.abspath("Figures")
plt.savefig(os.path.join(my_path, "00_Kimoto_et_al_(2023)_[_Fig_06_].png"))

from PIL import Image # py -m pip install pillow
img = Image.open(os.path.join(my_path, "00_Kimoto_et_al_(2023)_[_Fig_06_].png"))
img_resize = img.resize(size=(2866, 2016))
my_path2 = os.path.abspath("Figures/Fixed_Version/B6")
img_resize.save(os.path.join(my_path2, "00_Kimoto_et_al_(2023)_[_Fig_06_]_B6.png"))

print("")
print("########## Figure Information ##########")
print("figure.figsize:", plt.rcParams["figure.figsize"])
print("figure.dpi:", plt.rcParams["figure.dpi"])
print("figure.autolayout:", plt.rcParams["figure.autolayout"])
print("figure.subplot.left:", plt.rcParams["figure.subplot.left"])
print("figure.subplot.bottom:", plt.rcParams["figure.subplot.bottom"])
print("figure.subplot.right:", plt.rcParams["figure.subplot.right"])
print("figure.subplot.top:", plt.rcParams["figure.subplot.top"])
print("figure.subplot.wspace:", plt.rcParams["figure.subplot.wspace"])
print("figure.subplot.hspace:", plt.rcParams["figure.subplot.hspace"])

print("Re-setting: " + "plt.subplots_adjust(" +
       "left=" + str(plt.rcParams["figure.subplot.left"]) + " ,"
       + "bottom=" + str(plt.rcParams["figure.subplot.bottom"]) + " ,"
       + "right=" + str(plt.rcParams["figure.subplot.right"]) + " ,"
       + "top=" + str(plt.rcParams["figure.subplot.top"]) + " ,"
       + "wspace=" + str(plt.rcParams["figure.subplot.wspace"]) + " ,"
       + "hspace=" + str(plt.rcParams["figure.subplot.hspace"]) + ")")

print("")
print("########## Axes Information (1) ##########")
print("Axes Information, fig.axes:", Figure_object.axes)
print("")
print("########## Axes Information (2) ##########")
for i in range(0, len(Figure_object.axes)):
    print("Axes Information:", i, Figure_object.axes[i].title)
    print("                 ", " ", Figure_object.axes[i])

#plt.show()
########################################################################################################################