########################################################################################################################
# Fig. 3. Re-analysis based on the proposed ideas. (00_Kimoto_et_al_(2023)_[_Fig_03_])
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
# Corresponding author: Keiichiro Kimoto, M.Sc.
#
# Program version:1.0
# Author of this program: Keiichiro Kimoto, M.Sc. in Biol. (Kagoshima University, Data Strategy Research Institute)
#
########################################################################################################################
# References:
#
########################################################################################################################
# Contents
########################################################################################################################
# Data Review Results (DataReviewDataSet)
# Data Set (OriginalDataSet)
# Flag Data Set (FragDataSet)
# Label Data Set (LabelDataSet)
# Integrating Data Sets
# Regression Analysis
#    Definition of Regression Function
#    Linear Regression Analysis
#    Non-linear Regression Analysis on "Whole Data"
#    Non-linear Regression Analysis on "Risk Group A"
#    Non-linear Regression Analysis on "Risk Group  B"
#    Evaluation of Result on the Regression Analysis
#    Coefficient Matrix (Data Set)
# Values for Line and Curves
#    Regression Line on Outline
#    Regression Line on Whole Data
#    Regression Curve on Whole Data
#    Regression Line on Whole Data (Chandra's Line)
#    Regression Curves on Risk Group A
#    Regression Curves on Risk Group B
#    Risk Group A & Risk Group B (Distribution)
# Figures
#    (a) "Result of Source Data Review", Source Documents
#          Basic Informations on This Figure
#          Basic Items
#          Display of Data (1): Scatter Plot(s)
#          Display of Data (2): Plot(s)
#          Regression Line on Outline
#          Regression Line on Whole Data
#          Regression Curve on Whole Data
#          Regression Line on Whole Data (Chandra's Line)
#          Vertical Lines
#          Filling Aria
#          Annotations (1): Texts
#          Annotations: Legend
#    (c) "Estimated Distributions of Latent High-risk Populations"
#          Basic Informations on This Figure
#          Basic Items
#          Display of Data (1): Scatter Plot(s)
#          Display of Data (2): Plot(s)
#          Regression Curves on Risk Group A
#          Regression Curves on Risk Group B
#          Risk Group A & Risk Group B
#          Vertical Lines
#          Filling Aria
#          Annotations (1): Texts
#          Annotations (2): Texts & Allows
#          Annotations (3): Another Curve
#    (b) "Flow of Data Selection for Regression Analysis"
########################################################################################################################
# List of Import
########################################################################################################################
import os
import sys
from decimal import Decimal, ROUND_HALF_UP

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as optimize
from PIL import Image  # py -m pip install pillow

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option("display.width", 300)

########################################################################################################################
# For Manuscripts to Scientific Journals
########################################################################################################################
print(matplotlib.get_cachedir())
# /home/kimoto/.cache/matplotlib: fontlist-v310.json  fontlist-v330.json  tex.cache#
print(matplotlib.matplotlib_fname())
print(matplotlib.rcParams["font.family"])
print(matplotlib.rcParams["font.sans-serif"])
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rcParams['font.sans-serif'] = ["Arial"]
########################################################################################################################
# Making Data Sets
########################################################################################################################
############################################################################################################
# Data Review Results (DataReviewDataSet) #
############################################################################################################
header_line_1 = "------------------------------------------------------------------------------------"
header = ''.join(
    ["", "Label", "       ", "Author", "        ", "Year", "    ", "Data Source", "      ", "Duration", "   ", "Type",
     "  ", "Adj.(Data Review)", " "])
header_line_2 = "------------------------------------------------------------------------------------"
PN02_R = ''.join(
    ["", "K(2h)", "  ", "Kuipers et al.", "     ", "2007", " ", "Table 4 (0–4 hr)", "    ", "2", " ", "(2.2)", "", "hr",
     "  ", "Air", "  ", "Yes (OK)", "         "])
PN03_R = ''.join(
    ["", "K(6h)", "  ", "Kuipers et al.", "     ", "2007", " ", "Table 4 (4–8 hr)", "    ", "6", " ", "(5.8)", "", "hr",
     "  ", "Air", "  ", "Yes (OK)", "         "])
PN04_R = ''.join(
    ["", "C(6h)", "  ", "Cannegieter et al.", " ", "2006", " ", "Table 2 (4-8 hr)", "    ", "6", " ", "(6.2)", "", "hr",
     "  ", "Any", "  ", "NO?", "                "])
PN05_R = ''.join(
    ["", "K(10h)", " ", "Kuipers et al.", "     ", "2007", " ", "Table 4 (8-12 hr)", "  ", "10", " ", "(9.8)", "", "hr",
     "  ", "Air", "  ", "Yes (OK)", "         "])
PN06_R = ''.join(
    ["", "C(10h)", " ", "Cannegieter et al.", " ", "2006", " ", "Table 2 (8-12 hr)", "  ", "10", "", "(10.2)", "", "hr",
     "  ", "Any", "  ", "NO?", "                "])
PN07_R = ''.join(
    ["", "P(12h)", " ", "Parkin et al.", "      ", "2006", " ", "Table 5 (>8 hr)", "    ", "12", "", "(11.8)", "", "hr",
     "  ", "Air", "  ", "Yes (conditional)", ""])
PN08_R = ''.join(
    ["", "M(12h)", " ", "Martinelli et al.", "  ", "2003", " ", "Text (13.5 hr)", "     ", "12", "", "(12.2)", "", "hr",
     "  ", "Air", "  ", "Yes (NG)", "         "])
PN09_R = ''.join(
    ["", "K(14h)", " ", "Kuipers et al.", "     ", "2007", " ", "Table 4 (12-16 hr)", " ", "14", "  ", "(14)", "", "hr",
     "  ", "Air", "  ", "Yes (OK) ", "        "])
PN10_R = ''.join(
    ["", "C(16h)", " ", "Cannegieter et al.", " ", "2006", " ", "Table 2 (>12 hr)", "   ", "16", "  ", "(16)", "", "hr",
     "  ", "Any", "  ", "NO?", "               "])
PN11_R = ''.join(
    ["", "K(20h)", " ", "Kuipers et al.", "     ", "2007", " ", "Table 4 (>16 hr)", "   ", "20", "  ", "(20)", "", "hr",
     "  ", "Air", "  ", "Yes (OK)", "         "])
footer_line = "------------------------------------------------------------------------------------------"

PN02_R_2 = ''.join(["", "K(2h)", "  ", "Yes (OK)", "   "])
PN03_R_2 = ''.join(["", "K(6h)", "  ", "Yes (OK)", "   "])
PN04_R_2 = ''.join(["", "C(6h)", "  ", "NO?", "         "])
PN05_R_2 = ''.join(["", "K(10h)", " ", "Yes (OK)", "   "])
PN06_R_2 = ''.join(["", "C(10h)", " ", "NO?", "         "])
PN07_R_2 = ''.join(["", "P(12h)", " ", "Yes (cond.)", ""])
PN08_R_2 = ''.join(["", "M(12h)", " ", "Yes (NG)", "   "])
PN09_R_2 = ''.join(["", "K(14h)", " ", "Yes (OK)", "   "])
PN10_R_2 = ''.join(["", "C(16h)", " ", "NO?", "         "])
PN11_R_2 = ''.join(["", "K(20h)", " ", "Yes (OK)", "   "])

Legend_K = ''.join(["K", ":", " ", "Kuipers et al.", "    ", "(2007)"])
Legend_C = ''.join(["C", ":", " ", "Cannegieter et al.", "", "(2006)"])
Legend_P = ''.join(["P", ":", " ", "Parkin et al.", "     ", "(2006)"])
Legend_M = ''.join(["M", ":", " ", "Martinelli et al. ", "", "(2003)"])

ForList = ["header_line_1", "header", "header_line_2", "PN02_R", "PN03_R", "PN04_R", "PN05_R", "PN06_R", "PN07_R",
           "PN08_R", "PN09_R", "PN10_R", "PN11_R", "footer_line", "header_line_1_2", "header_2", "header_line_2_2",
           "PN02_R_2", "PN03_R_2", "PN04_R_2", "PN05_R_2", "PN06_R_2", "PN07_R_2", "PN08_R_2", "PN09_R_2", "PN10_R_2",
           "PN11_R_2", "footer_line_2", "Legend_K", "Legend_C", "Legend_P", "Legend_M"]

print("##### Data Review Results (DataReviewDataSet) #####")
print(header_line_1)
print(header)
print(header_line_2)
print(PN02_R)
print(PN03_R)
print(PN04_R)
print(PN05_R)
print(PN06_R)
print(PN07_R)
print(PN08_R)
print(PN09_R)
print(PN10_R)
print(PN11_R)
print(footer_line)

print(Legend_K)
print(Legend_C)
print(Legend_P)
print(Legend_M)

############################################################################################################
# Data Set (OriginalDataSet) #
############################################################################################################
PN01 = [1, "Ca", 2009, "None", 0.0, 0.0, 1.0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0]
PN02 = [2, "K", 2007, "Air", 2.0, 2.0, 0.4, 0, 0, 0, 0, 1.9, 0.1]
PN03 = [3, "K", 2007, "Air", 6.0, 5.8, 2.3, 0, 0, 0, 0, 5.9, 0.9]
PN04 = [4, "C", 2006, "Any", 6.0, 6.2, 2.0, 36, 18, 1673, 1724]  # Need Deviation1
PN05 = [5, "K", 2007, "Air", 10.0, 9.8, 2.2, 0, 0, 0, 0, 5.4, 0.9]
PN06 = [6, "C", 2006, "Any", 10.0, 10.2, 1.8, 33, 18, 1673, 1724]  # Need Deviation1
PN07 = [7, "P", 2006, "Air", 12.0, 11.8, 7.9, 0, 0, 0, 0, 55.1, 1.1]
PN08 = [8, "M", 2003, "Air", 12.0, 12.2, 3.0, 0, 0, 0, 0, 9.5, 0.9]
PN09 = [9, "K", 2007, "Air", 14.0, 14.0, 5.3, 0, 0, 0, 0, 12.4, 2.3]
PN10 = [10, "C", 2006, "Any", 16.0, 16.0, 2.8, 25, 9, 1673, 1724]  # Need Deviation1
PN11 = [11, "K", 2007, "Air", 20.0, 20.0, 5.7, 0, 0, 0, 0, 16.5, 2.0]

# Deviation1 (PN04, PN06, PN10)
# Matched case-control data (e.g., https://www.sjsu.edu/faculty/gerstman/StatPrimer/case-control.pdf)
PN04.extend([np.exp(np.log(PN04[6]) + 1.96 * np.sqrt((1 / PN04[7]) + (1 / PN04[8]))),
             np.exp(np.log(PN04[6]) - 1.96 * np.sqrt((1 / PN04[7]) + (1 / PN04[8])))])
PN06.extend([np.exp(np.log(PN06[6]) + 1.96 * np.sqrt((1 / PN06[7]) + (1 / PN06[8]))),
             np.exp(np.log(PN06[6]) - 1.96 * np.sqrt((1 / PN06[7]) + (1 / PN06[8])))])
PN10.extend([np.exp(np.log(PN10[6]) + 1.96 * np.sqrt((1 / PN10[7]) + (1 / PN10[8]))),
             np.exp(np.log(PN10[6]) - 1.96 * np.sqrt((1 / PN10[7]) + (1 / PN10[8])))])

# PN04.extend([np.exp(np.log(PN04[6]) + 1.96*np.sqrt((1 / PN04[7]) + (1 / PN04[8]) + (1 / PN04[9]) + (1 / PN04[10]))),
#             np.exp(np.log(PN04[6]) - 1.96*np.sqrt((1 / PN04[7]) + (1 / PN04[8]) + (1 / PN04[9]) + (1 / PN04[10])))])
# PN06.extend([np.exp(np.log(PN06[6]) + 1.96*np.sqrt((1 / PN06[7]) + (1 / PN06[8]) + (1 / PN06[9]) + (1 / PN06[10]))),
#             np.exp(np.log(PN06[6]) - 1.96*np.sqrt((1 / PN06[7]) + (1 / PN06[8]) + (1 / PN06[9]) + (1 / PN06[10])))])
# PN10.extend([np.exp(np.log(PN10[6]) + 1.96*np.sqrt((1 / PN10[7]) + (1 / PN10[8]) + (1 / PN10[9]) + (1 / PN10[10]))),
#             np.exp(np.log(PN10[6]) - 1.96*np.sqrt((1 / PN10[7]) + (1 / PN10[8]) + (1 / PN10[9]) + (1 / PN10[10])))])

# Deviation2 (PN02~PN11)
PN02.extend([np.log(PN02[6]), np.log(PN02[11]), np.log(PN02[12])])
PN03.extend([np.log(PN03[6]), np.log(PN03[11]), np.log(PN03[12])])
PN04.extend([np.log(PN04[6]), np.log(PN04[11]), np.log(PN04[12])])
PN05.extend([np.log(PN05[6]), np.log(PN05[11]), np.log(PN05[12])])
PN06.extend([np.log(PN06[6]), np.log(PN06[11]), np.log(PN06[12])])
PN07.extend([np.log(PN07[6]), np.log(PN07[11]), np.log(PN07[12])])
PN08.extend([np.log(PN08[6]), np.log(PN08[11]), np.log(PN08[12])])
PN09.extend([np.log(PN09[6]), np.log(PN09[11]), np.log(PN09[12])])
PN10.extend([np.log(PN10[6]), np.log(PN10[11]), np.log(PN10[12])])
PN11.extend([np.log(PN11[6]), np.log(PN11[11]), np.log(PN11[12])])

# Data Frame (PN01~PN11)
OriginalDataSet = pd.DataFrame([PN01, PN02, PN03, PN04, PN05, PN06, PN07, PN08, PN09, PN10, PN11],
                               index=["PN01", "PN02", "PN03", "PN04", "PN05", "PN06", "PN07", "PN08", "PN09", "PN10",
                                      "PN11"],
                               columns=["PN", "Au", "Year", "Type", "Time", "TimeF", "Data", "TEY", "TEN", "CEY", "CEN",
                                        "CI(U)", "CI(L)", "LN(D)", "LN(U)", "LN(L)"]
                               )
# print(OriginalDataSet)

############################################################################################################
# Flag Data Set (FragDataSet) #
############################################################################################################
PN01_F = [1, "B", "black", "blue"]
PN02_F = [2, "A", "black", "red"]
PN03_F = [3, "A", "black", "red"]
PN04_F = [4, "C", "green", "limegreen"]
PN05_F = [5, "B", "black", "blue"]
PN06_F = [6, "C", "green", "limegreen"]
PN07_F = [7, "A", "black", "red"]
PN08_F = [8, "D", "orange", "orange"]
PN09_F = [9, "B", "black", "blue"]
PN10_F = [10, "C", "green", "limegreen"]
PN11_F = [11, "B", "black", "blue"]
# Data Frame
FragDataSet = pd.DataFrame([PN01_F, PN02_F, PN03_F, PN04_F, PN05_F, PN06_F, PN07_F, PN08_F, PN09_F, PN10_F, PN11_F],
                           index=["PN01", "PN02", "PN03", "PN04", "PN05", "PN06", "PN07", "PN08", "PN09", "PN10",
                                  "PN11"],
                           columns=["PN", "RG", "labelC1", "labelC2"]
                           )
# print(FragDataSet)

############################################################################################################
# Label Data Set (LabelDataSet) #
############################################################################################################
PN01_L = [1, -0.25, 0.10, "", ""]
PN02_L = [2, 1.95, 0.74, "K(2h)", "K(2h)"]
PN03_L = [3, 5.65, 1.87, "K(6h)", "K(6h)"]
PN04_L = [4, 5.95, 1.36, "C(6h)", ""]
PN05_L = [5, 9.55, 1.79, "K(10h)", "K(10h)"]
PN06_L = [6, 9.95, 1.27, "C(10h)", ""]
PN07_L = [7, 11.55, 4.20, "P(12h)", "P(12h)"]
PN08_L = [8, 12.10, 2.45, "M(12h)", ""]
PN09_L = [9, 13.75, 2.62, "K(14h)", "K(14h)"]
PN10_L = [10, 15.75, 1.89, "C(16h)", ""]
PN11_L = [11, 19.00, 2.90, "K(20h)", "K(20h)"]

# Data Frame
LabelDataSet = pd.DataFrame([PN01_L, PN02_L, PN03_L, PN04_L, PN05_L, PN06_L, PN07_L, PN08_L, PN09_L, PN10_L, PN11_L],
                            index=["PN01", "PN02", "PN03", "PN04", "PN05", "PN06", "PN07", "PN08", "PN09", "PN10",
                                   "PN11"],
                            columns=["PN", "Lx", "Ly", "Label", "Label2"]
                            )
# print(LabelDataSet)


Air = ["Air travel", 31, 179, 16, 194]
Long = ["Long-distance flights(Patients: 13 hours 30 minutes, Controls: 19 hours)", 11, 179, 4, 194]
Short = ["Short-distance flights(Patients: 2 hours, Controls: 1 hour 30 minutes)", 20, 179, 12, 194]

MartinelliDataSet = pd.DataFrame(
    [Air, Long, Short],
    index=["Air", "Long", "Short"],
    columns=["Travel Type", "TEY", "TEN", "CEY", "CEN"])

MartinelliDataSet["OR"] = (MartinelliDataSet["TEY"] / MartinelliDataSet["TEN"]) / (
        MartinelliDataSet["CEY"] / MartinelliDataSet["CEN"])
MartinelliDataSet["CIL"] = np.exp(np.log(MartinelliDataSet["OR"]) - 1.96 * np.sqrt(
    (1 / MartinelliDataSet["TEY"]) + (1 / MartinelliDataSet["TEN"]) + (1 / MartinelliDataSet["CEY"]) + (
            1 / MartinelliDataSet["CEN"])))
MartinelliDataSet["CIU"] = np.exp(np.log(MartinelliDataSet["OR"]) + 1.96 * np.sqrt(
    (1 / MartinelliDataSet["TEY"]) + (1 / MartinelliDataSet["TEN"]) + (1 / MartinelliDataSet["CEY"]) + (
            1 / MartinelliDataSet["CEN"])))

MartinelliDataSet["LN(OR)"] = np.log(MartinelliDataSet["OR"])
MartinelliDataSet["LN(L)"] = np.log(MartinelliDataSet["CIL"])
MartinelliDataSet["LN(U)"] = np.log(MartinelliDataSet["CIU"])

MartinelliDataSet["OR - Abs( LN(L) - LN(OR) )"] = \
    MartinelliDataSet["OR"] - np.abs(MartinelliDataSet["LN(L)"] - MartinelliDataSet["LN(OR)"])

MartinelliDataSet["OR + Abs( LN(L) - LN(OR) )"] = \
    MartinelliDataSet["OR"] + np.abs(MartinelliDataSet["LN(U)"] - MartinelliDataSet["LN(OR)"])

OC_Affected = (2.0 + 1.8 + 2.8 + 3.0 + 1.8) / 5
############################################################################################################
# Integrating Data Sets
############################################################################################################
AnalysisDataSet = pd.merge(pd.merge(OriginalDataSet, FragDataSet, on="PN", how="left"), LabelDataSet, on="PN",
                           how="left")
AnalysisDataSet.index = ["PN01", "PN02", "PN03", "PN04", "PN05", "PN06", "PN07", "PN08", "PN09", "PN10", "PN11"]

AnalysisDataSet = AnalysisDataSet.drop(["PN01"], axis=0)

AnalysisDataSet["OR - Abs( LN(L) - LN(OR) )"] = \
    AnalysisDataSet["Data"] - np.abs(AnalysisDataSet["LN(L)"] - AnalysisDataSet["LN(D)"])
AnalysisDataSet["OR + Abs( LN(U) - LN(OR) )"] = \
    AnalysisDataSet["Data"] + np.abs(AnalysisDataSet["LN(U)"] - AnalysisDataSet["LN(D)"])

AnalysisDataSet_A = AnalysisDataSet[AnalysisDataSet['RG'] == "A"]
AnalysisDataSet_B = AnalysisDataSet[AnalysisDataSet['RG'] == "B"]
AnalysisDataSet_C = AnalysisDataSet[AnalysisDataSet['RG'] == "C"]
AnalysisDataSet_D = AnalysisDataSet[AnalysisDataSet['RG'] == "D"]

print("")
print("########## Whole of Data (AnalysisDataSet) ##########")
print(AnalysisDataSet)
print("")
print("########## Risk Group A (AnalysisDataSet_A) ##########")
print(AnalysisDataSet_A)
print("")
print("########## Risk Group B (AnalysisDataSet_B) ##########")
print(AnalysisDataSet_B)
print("")
# print("########## Risk Group C (AnalysisDataSet_C) ##########")
# print(AnalysisDataSet_C)
# print("")
# print("########## Risk Group D (AnalysisDataSet_D) ##########")
# print(AnalysisDataSet_D)
# print("########## Simplified Data Set for Other Figure(s) ##########")
print("")
print("########## Martinelli Data Set (MartinelliDataSet) ##########")
print(MartinelliDataSet)
print("")
print("########## OC Affected Mean (OC_Affected) ##########")
print(OC_Affected)


########################################################################################################################
# Regression Analysis
########################################################################################################################

############################################################################################################
# Definition of Regression Function
############################################################################################################

def reg_func_l(parameter, x, y):
    alpha = parameter[0]
    beta = parameter[1]
    _Output_ = y - (alpha * x + beta)
    return _Output_


def reg_func_s(parameter, x, y):
    k1 = parameter[0]
    k2 = parameter[1]
    l = 1.0
    m = parameter[2]
    _Output_ = y - (k1 / (1 + np.exp(-l * (x - m))) + k2)
    return _Output_


def reg_func_ci(parameter, x, y, k1, k2, l, m):
    a = parameter[0]
    b = parameter[1]
    c = parameter[2]
    _Output_ = y - (k1 / (1 + np.exp(-l * (x - m))) + k2 + a * x ** 2 + b * x + c)
    return _Output_


############################################################################################################
# Linear Regression Analysis
############################################################################################################
# Line
x = AnalysisDataSet["Time"]
y = AnalysisDataSet["LN(D)"]
parameter_0 = [0.0, 0.0]
RegResult_Original_L = optimize.leastsq(reg_func_l, parameter_0, args=(x, y), full_output=True)
Result_L0 = RegResult_Original_L[0]
Result_L = [Result_L0[0], Result_L0[1]]
############################################################################################################
# Non-linear Regression Analysis on "Whole Data"
############################################################################################################
x = AnalysisDataSet["Time"]
y = AnalysisDataSet["LN(D)"]
parameter_0 = [0.0, 0.0, 0.0]
RegResult_Original_W = optimize.leastsq(reg_func_s, parameter_0, args=(x, y), full_output=True)
Result_W0 = RegResult_Original_W[0]
Result_W = [Result_W0[0], Result_W0[1], 1.0, Result_W0[2]]

############################################################################################################
# Non-linear Regression Analysis on "Risk Group A"
############################################################################################################
# S Curve
x = AnalysisDataSet_A["Time"]
y = AnalysisDataSet_A["Data"]
parameter_0 = [0.0, 0.0, 0.0]
RegResult_Original_AS = optimize.leastsq(reg_func_s, parameter_0, args=(x, y), full_output=True)
Result_AS0 = RegResult_Original_AS[0]
Result_AS = [Result_AS0[0], Result_AS0[1], 1.0, Result_AS0[2]]

# CI Upper
x = AnalysisDataSet_A["Time"]
y = AnalysisDataSet_A["OR - Abs( LN(L) - LN(OR) )"]
k1 = Result_AS[0]
k2 = Result_AS[1]
l = Result_AS[2]
m = Result_AS[3]
parameter_0 = [0.0, 0.0, 0.0]
RegResult_Original_AU = optimize.leastsq(reg_func_ci, parameter_0, args=(x, y, k1, k2, l, m), full_output=True)
Result_AU0 = RegResult_Original_AU[0]
Result_AU = [Result_AS[0], Result_AS[1], Result_AS[2], Result_AS[3], Result_AU0[0], Result_AU0[1], Result_AU0[2]]

# CI Lower
x = AnalysisDataSet_A["Time"]
y = AnalysisDataSet_A["OR + Abs( LN(U) - LN(OR) )"]
k1 = Result_AS[0]
k2 = Result_AS[1]
l = Result_AS[2]
m = Result_AS[3]
parameter_0 = [0.0, 0.0, 0.0]
RegResult_Original_AL = optimize.leastsq(reg_func_ci, parameter_0, args=(x, y, k1, k2, l, m), full_output=True)
Result_AL0 = RegResult_Original_AL[0]
Result_AL = [Result_AS[0], Result_AS[1], Result_AS[2], Result_AS[3], Result_AL0[0], Result_AL0[1], Result_AL0[2]]

############################################################################################################
# Non-linear Regression Analysis on "Risk Group  B"
############################################################################################################
# S Curve
x = AnalysisDataSet_B["Time"]
y = AnalysisDataSet_B["Data"]

parameter_0 = [0.0, 0.0, 0.0]
RegResult_Original_BS = optimize.leastsq(reg_func_s, parameter_0, args=(x, y), full_output=True)
Result_BS0 = RegResult_Original_BS[0]
Result_BS = [Result_BS0[0], Result_BS0[1], 1.0, Result_BS0[2]]

# CI Upper
x = AnalysisDataSet_B["Time"]
y = AnalysisDataSet_B["OR - Abs( LN(L) - LN(OR) )"]
k1 = Result_BS[0]
k2 = Result_BS[1]
l = Result_BS[2]
m = Result_BS[3]
parameter_0 = [0.004, -0.1, 1.6]
RegResult_Original_BU = optimize.leastsq(reg_func_ci, parameter_0, args=(x, y, k1, k2, l, m), full_output=True)
Result_BU0 = RegResult_Original_BU[0]
Result_BU = [Result_BS[0], Result_BS[1], Result_BS[2], Result_BS[3], Result_BU0[0], Result_BU0[1], Result_BU0[2]]

# CI Lower
x = AnalysisDataSet_B["Time"]
y = AnalysisDataSet_B["OR + Abs( LN(U) - LN(OR) )"]
k1 = Result_BS[0]
k2 = Result_BS[1]
l = Result_BS[2]
m = Result_BS[3]
parameter_0 = [0.0, 0.0, 0.0]
RegResult_Original_BL = optimize.leastsq(reg_func_ci, parameter_0, args=(x, y, k1, k2, l, m), full_output=True)
Result_BL0 = RegResult_Original_BL[0]
Result_BL = [Result_BS[0], Result_BS[1], Result_BS[2], Result_BS[3], Result_BL0[0], Result_BL0[1], Result_BL0[2]]

############################################################################################################
# Non-linear Regression Analysis on "Risk Group  C"
############################################################################################################
# S Curve
x = AnalysisDataSet_C["Time"]
y = AnalysisDataSet_C["Data"]

parameter_0 = [1.0, 1.5, 11.0]
RegResult_Original_CS = optimize.leastsq(reg_func_s, parameter_0, args=(x, y), full_output=True)
Result_CS0 = RegResult_Original_CS[0]
Result_CS = [Result_CS0[0], Result_CS0[1], 1.0, Result_CS0[2]]

parameter_0 = [0.0, 0.0]
RegResult_Original_CLine = optimize.leastsq(reg_func_l, parameter_0, args=(x, y), full_output=True)
Result_CLine0 = RegResult_Original_CLine[0]
Result_CLine = [Result_CLine0[0], Result_CLine0[1]]
############################################################################################################
# Non-linear Regression Analysis on "Risk Group  MartinelliDataSet"
############################################################################################################
x = [2.0, 12.0]
y0 = MartinelliDataSet["OR"]
y = [y0[2], y0[1]]
Result_Martinelli_Line = [(y[1] - y[0]) / (x[1] - x[0]), y[0] - ((y[1] - y[0]) / (x[1] - x[0])) * x[0]]

############################################################################################################
# Evaluation of Result on the Regression Analysis
############################################################################################################
print("")
print("########## Evaluation of Result on the Regression Analysis ##########")
print("Reg. Result on Whole Data (S Curve)", "Evaluation of Result:", RegResult_Original_W[-1],
      "(1, 2, 3 or 4, the solution was found.)")
print("Reg. Result on A (S Curve)         ", "Evaluation of Result:", RegResult_Original_AS[-1],
      "(1, 2, 3 or 4, the solution was found.)")
print("Reg. Result on A (CI Curve (U))    ", "Evaluation of Result:", RegResult_Original_AU[-1],
      "(1, 2, 3 or 4, the solution was found.)")
print("Reg. Result on A (CI Curve (L))    ", "Evaluation of Result:", RegResult_Original_AL[-1],
      "(1, 2, 3 or 4, the solution was found.)")
print("Reg. Result on B (S Curve)         ", "Evaluation of Result:", RegResult_Original_BS[-1],
      "(1, 2, 3 or 4, the solution was found.)")
print("Reg. Result on B (CI Curve (U))    ", "Evaluation of Result:", RegResult_Original_BU[-1],
      "(1, 2, 3 or 4, the solution was found.)")
print("Reg. Result on B (CI Curve (L))    ", "Evaluation of Result:", RegResult_Original_BL[-1],
      "(1, 2, 3 or 4, the solution was found.)")
print("Reg. Result on C (S Curve)         ", "Evaluation of Result:", RegResult_Original_CS[-1],
      "(1, 2, 3 or 4, the solution was found.)")
# print("Reg. Result on D (CI Curve (L))    ", "Evaluation of Result:", RegResult_Original_DS[-1],
#      "(1, 2, 3 or 4, the solution was found.)")
print("Reg. Result on C (Line)            ", "Evaluation of Result:", RegResult_Original_CLine[-1],
      "(1, 2, 3 or 4, the solution was found.)")

############################################################################################################
# Coefficient Matrix (Data Set)
############################################################################################################
CoefficientDataSet_L = pd.DataFrame([[Result_L[0], Result_L[1], ""], [143 / 2000, -1 / 10,
                                                                      "obtained from Chandra et al. 2009 Figure 3"]],
                                    index=["Reg. Result on Whole Data",
                                           "Reg. Result on Whole Data (Chandra's Line)"],
                                    columns=["alpha", "beta", "Notice"]
                                    )

CoefficientDataSet = pd.DataFrame([
    Result_W, Result_AS, Result_AU, Result_AL, Result_BS, Result_BU, Result_BL, Result_CS],
    index=["Reg. Result on Whole Data (S Curve)",
           "Reg. Result on A (S Curve)         ",
           "Reg. Result on A (CI Curve (U))    ",
           "Reg. Result on A (CI Curve (L))    ",
           "Reg. Result on B (S Curve)         ",
           "Reg. Result on B (CI Curve (U))    ",
           "Reg. Result on B (CI Curve (L))    ",
           "Reg. Result on C (S Curve)         "],
    columns=["K1", "K2", "L", "M (Mean)", "a", "b", "c"]
)

print("")
print("########## Result of Linear Regression Analysis (CoefficientDataSet_L) ##########")
print(CoefficientDataSet_L)

print("")
print("########## Result of Non-linear Regression Analysis (CoefficientDataSet) ##########")
print(CoefficientDataSet)

########################################################################################################################
# Values for Line and Curves
########################################################################################################################
############################################################################################################
# Regression Line on Outline
# X_Outline; Y_Outline_L, Y_Outline_U, Y_OutLine_L
############################################################################################################
X_Outline = np.linspace(np.min(AnalysisDataSet_A["Time"]) - 0.2, np.max(AnalysisDataSet_A["Time"]) + 0.0, 1000)
Y_Outline_Line = 0.288306 * X_Outline - 1.192637
Y_Outline_U = (0.288306 * X_Outline - 1.192637) + (0.018630 * X_Outline ** 2 - 0.212403 * X_Outline + 1.608162)
Y_OutLine_L = (0.288306 * X_Outline - 1.192637) + (-0.057579 * X_Outline ** 2 + 0.757596 * X_Outline - 2.971434)

############################################################################################################
# Regression Line on Whole Data
# X_Line_W; Y_Line_W
############################################################################################################
alpha = CoefficientDataSet_L.at["Reg. Result on Whole Data", "alpha"]
beta = CoefficientDataSet_L.at["Reg. Result on Whole Data", "beta"]
X_Line_W = np.linspace(0.0, 20.0, 1000)
Y_Line_W = alpha * X_Line_W + beta

############################################################################################################
# Regression Curve on Whole Data
# X_Curve_W; Y_Curve_W
############################################################################################################
k1 = CoefficientDataSet.at["Reg. Result on Whole Data (S Curve)", "K1"]
k2 = CoefficientDataSet.at["Reg. Result on Whole Data (S Curve)", "K2"]
l = CoefficientDataSet.at["Reg. Result on Whole Data (S Curve)", "L"]
m = CoefficientDataSet.at["Reg. Result on Whole Data (S Curve)", "M (Mean)"]
X_Curve_W = np.linspace(0.0, 20.0, 1000)
Y_Curve_W = k1 / (1 + np.exp(-l * (X_Curve_W - m))) + k2
Y_Curve_W_Diff = (k1 * l * np.exp(-l * (X_Curve_W - m))) / (np.exp(-l * (X_Curve_W - m)) + 1) ** 2

############################################################################################################
# Regression Line on Whole Data (Chandra's Line)
# X_Line_C; Y_Line_C
############################################################################################################
alpha = CoefficientDataSet_L.at["Reg. Result on Whole Data (Chandra's Line)", "alpha"]
beta = CoefficientDataSet_L.at["Reg. Result on Whole Data (Chandra's Line)", "beta"]
X_Line_C = np.linspace(0.0, 20.0, 1000)
Y_Line_C = alpha * X_Line_C + beta

############################################################################################################
# Regression Curves on Risk Group A
# X_A; Y_AS/Y_AS_Diff, Y_AU, Y_AL
############################################################################################################
X_A_S = np.linspace(np.min(AnalysisDataSet_A["Time"]) - 0.0, np.max(AnalysisDataSet_A["Time"]) + 0.0, 1000)
X_A_CI = np.linspace(np.min(AnalysisDataSet_A["Time"]) - 0.5, np.max(AnalysisDataSet_A["Time"]) + 0.5, 1000)

# Y_AS, Y_AS_Diff
k1 = CoefficientDataSet.at["Reg. Result on A (S Curve)         ", "K1"]
k2 = CoefficientDataSet.at["Reg. Result on A (S Curve)         ", "K2"]
l = CoefficientDataSet.at["Reg. Result on A (S Curve)         ", "L"]
m = CoefficientDataSet.at["Reg. Result on A (S Curve)         ", "M (Mean)"]
Y_AS = k1 / (1 + np.exp(-l * (X_A_S - m))) + k2
Y_AS_Diff = (k1 * l * np.exp(-l * (X_A_S - m))) / (np.exp(-l * (X_A_S - m)) + 1) ** 2

# Y_AU
k1 = CoefficientDataSet.at["Reg. Result on A (CI Curve (U))    ", "K1"]
k2 = CoefficientDataSet.at["Reg. Result on A (CI Curve (U))    ", "K2"]
l = CoefficientDataSet.at["Reg. Result on A (CI Curve (U))    ", "L"]
m = CoefficientDataSet.at["Reg. Result on A (CI Curve (U))    ", "M (Mean)"]
a = CoefficientDataSet.at["Reg. Result on A (CI Curve (U))    ", "a"]
b = CoefficientDataSet.at["Reg. Result on A (CI Curve (U))    ", "b"]
c = CoefficientDataSet.at["Reg. Result on A (CI Curve (U))    ", "c"]
Y_AU = k1 / (1 + np.exp(-l * (X_A_CI - m))) + k2 + a * X_A_CI ** 2 + b * X_A_CI + c

# Y_AL
k1 = CoefficientDataSet.at["Reg. Result on A (CI Curve (L))    ", "K1"]
k2 = CoefficientDataSet.at["Reg. Result on A (CI Curve (L))    ", "K2"]
l = CoefficientDataSet.at["Reg. Result on A (CI Curve (L))    ", "L"]
m = CoefficientDataSet.at["Reg. Result on A (CI Curve (L))    ", "M (Mean)"]
a = CoefficientDataSet.at["Reg. Result on A (CI Curve (L))    ", "a"]
b = CoefficientDataSet.at["Reg. Result on A (CI Curve (L))    ", "b"]
c = CoefficientDataSet.at["Reg. Result on A (CI Curve (L))    ", "c"]
Y_AL = k1 / (1 + np.exp(-l * (X_A_CI - m))) + k2 + a * X_A_CI ** 2 + b * X_A_CI + c

############################################################################################################
# Regression Curves on Risk Group B
# X_B; Y_BS/Y_BS_Diff, Y_BU, Y_BL
############################################################################################################
X_B_S = np.linspace(0.0, np.max(AnalysisDataSet_B["Time"]) + 0.0, 1000)
X_B_CI = np.linspace(-0.5, np.max(AnalysisDataSet_B["Time"]) + 0.5, 1000)

# Y_BS, Y_BS_Diff
k1 = CoefficientDataSet.at["Reg. Result on B (S Curve)         ", "K1"]
k2 = CoefficientDataSet.at["Reg. Result on B (S Curve)         ", "K2"]
l = CoefficientDataSet.at["Reg. Result on B (S Curve)         ", "L"]
m = CoefficientDataSet.at["Reg. Result on B (S Curve)         ", "M (Mean)"]
Y_BS = k1 / (1 + np.exp(-l * (X_B_S - m))) + k2
Y_BS_Diff = (k1 * l * np.exp(-l * (X_B_S - m))) / (np.exp(-l * (X_B_S - m)) + 1) ** 2

# Y_BU
k1 = CoefficientDataSet.at["Reg. Result on B (CI Curve (U))    ", "K1"]
k2 = CoefficientDataSet.at["Reg. Result on B (CI Curve (U))    ", "K2"]
l = CoefficientDataSet.at["Reg. Result on B (CI Curve (U))    ", "L"]
m = CoefficientDataSet.at["Reg. Result on B (CI Curve (U))    ", "M (Mean)"]
a = CoefficientDataSet.at["Reg. Result on B (CI Curve (U))    ", "a"]
b = CoefficientDataSet.at["Reg. Result on B (CI Curve (U))    ", "b"]
c = CoefficientDataSet.at["Reg. Result on B (CI Curve (U))    ", "c"]
Y_BU = k1 / (1 + np.exp(-l * (X_B_CI - m))) + k2 + a * X_B_CI ** 2 + b * X_B_CI + c

# Y_BL
k1 = CoefficientDataSet.at["Reg. Result on B (CI Curve (L))    ", "K1"]
k2 = CoefficientDataSet.at["Reg. Result on B (CI Curve (L))    ", "K2"]
l = CoefficientDataSet.at["Reg. Result on B (CI Curve (L))    ", "L"]
m = CoefficientDataSet.at["Reg. Result on B (CI Curve (L))    ", "M (Mean)"]
a = CoefficientDataSet.at["Reg. Result on B (CI Curve (L))    ", "a"]
b = CoefficientDataSet.at["Reg. Result on B (CI Curve (L))    ", "b"]
c = CoefficientDataSet.at["Reg. Result on B (CI Curve (L))    ", "c"]
Y_BL = k1 / (1 + np.exp(-l * (X_B_CI - m))) + k2 + a * X_B_CI ** 2 + b * X_B_CI + c

############################################################################################################
# Regression Curves on Risk Group C
# X_C; Y_CS
############################################################################################################
X_C_S = np.linspace(0.0, np.max(AnalysisDataSet_C["Time"]) + 0.0, 1000)

# Y_CS
k1 = CoefficientDataSet.at["Reg. Result on C (S Curve)         ", "K1"]
k2 = CoefficientDataSet.at["Reg. Result on C (S Curve)         ", "K2"]
l = CoefficientDataSet.at["Reg. Result on C (S Curve)         ", "L"]
m = CoefficientDataSet.at["Reg. Result on C (S Curve)         ", "M (Mean)"]
Y_CS = k1 / (1 + np.exp(-l * (X_C_S - m))) + k2

alpha = Result_CLine[0]
beta = Result_CLine[1]
X_GroupC_Line = np.linspace(np.min(AnalysisDataSet_C["Time"]) + 0.0, np.max(AnalysisDataSet_C["Time"]) + 0.0, 1000)
Y_GroupC_Line = alpha * X_GroupC_Line + beta
############################################################################################################
# Regression Curves on Risk Group Result_Martinelli
# X_D; Y_DS
############################################################################################################
alpha = Result_Martinelli_Line[0]
beta = Result_Martinelli_Line[1]
X_GroupMartinelli_Line = np.linspace(np.min([2.0, 12.0]) + 0.0, np.max([2.0, 12.0]) + 0.0, 1000)
Y_GroupMartinelli_Line = alpha * X_GroupMartinelli_Line + beta
#################################################################################################
# Risk Group A & Risk Group B (Distribution)
#################################################################################################
X_Line_W = np.linspace(0.0, 20.0, 1000)

# Y_AS_Diff
k1a = CoefficientDataSet.at["Reg. Result on A (S Curve)         ", "K1"]
k2a = CoefficientDataSet.at["Reg. Result on A (S Curve)         ", "K2"]
la = CoefficientDataSet.at["Reg. Result on A (S Curve)         ", "L"]
ma = CoefficientDataSet.at["Reg. Result on A (S Curve)         ", "M (Mean)"]

# Y_BS_Diff
k1b = CoefficientDataSet.at["Reg. Result on B (S Curve)         ", "K1"]
k2b = CoefficientDataSet.at["Reg. Result on B (S Curve)         ", "K2"]
lb = CoefficientDataSet.at["Reg. Result on B (S Curve)         ", "L"]
mb = CoefficientDataSet.at["Reg. Result on B (S Curve)         ", "M (Mean)"]

Y_Total_Diff = (k1a * la * np.exp(-la * (X_Line_W - ma))) / (np.exp(-la * (X_Line_W - ma)) + 1) ** 2 + \
               (k1b * l * np.exp(-lb * (X_Line_W - mb))) / (np.exp(-lb * (X_Line_W - mb)) + 1) ** 2

########################################################################################################################
# Figures
########################################################################################################################
# (a) "Result of Source Data Verification (SDV) & Data Review"
# (b) "Flow of Data Selection for Regression Analysis"
# (c) "Identify Latent Patterns of  High-risk Populations & Re-regression Analysis"
########################################################################################################################
Figure_object = plt.figure(1, figsize=(11.69, 8.27 * (3 / 4)), dpi=400, edgecolor="black", linewidth=0.5)
# plt.subplots_adjust(left=0.25, bottom=None, right=None, top=None, wspace=None, hspace=None)
plt.subplots_adjust(left=0.125, bottom=0.11, right=0.9, top=0.88, wspace=0.2, hspace=0.2)

ticks_x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
ticks_y = [-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
ticks_y2 = [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

Axes_obj_01 = plt.axes([0.0730777, 0.0899279, 0.414848, 0.853509], xticks=ticks_x, yticks=ticks_y)
Axes_obj_02 = plt.axes([0.570896, 0.0899279, 0.414848, 0.853509], xticks=ticks_x, yticks=ticks_y2)
# Axes_obj_03 = plt.axes([0.0730777-0.050, 0.0786875-0.070, 0.199689*1.40, 0.177484*1.40])
# Axes_obj_04 = plt.axes([0.3526420+0.025, 0.0675971-0.010, 0.599067, 0.199665+0.015])

plt.figtext(0.0300, 0.950, "a", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")
plt.figtext(0.5130 + 0.015, 0.950, "b", horizontalalignment='center', fontsize=13.0549, fontweight="bold",
            color="black")
# plt.figtext(0.0300, 0.245+0.050, "c", horizontalalignment='center', fontsize=13.0549, fontweight="bold",
# color="black") plt.figtext(0.3200+0.025, 0.245, "d", horizontalalignment='center', fontsize=13.0549,
# fontweight="bold", color="black") plt.figtext(0.3200+0.025+0.001+0.1, 0.245+0.050, "d",
# horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")

# plt.figtext(0.3200+0.01, 0.245, "d", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")

gs_master = matplotlib.gridspec.GridSpec(nrows=1, ncols=1)
gs_1 = matplotlib.gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=2, subplot_spec=gs_master[0])
# gs_2 = matplotlib.gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=2, subplot_spec=gs_master[1], width_ratios=[1, 3])
# Axes_obj_01 = Figure_object.add_subplot(gs_1[0], xticks=ticks_x, yticks=ticks_y)
# Axes_obj_02 = Figure_object.add_subplot(gs_1[1], xticks=ticks_x, yticks=ticks_y)
########################################################################################################################
############################################################################################################
# (a) "Result of Source Data Review", Source Documents: Level of Eligibility for Regression Analysis:
############################################################################################################
# Axes_obj_01 = Figure_object.add_subplot(gs_1[0], xticks=ticks_x, yticks=ticks_y)
# Axes_obj_01.tick_params(direction="in")
# Basic Information on This Figure #
Axes_obj_01.set_ylabel("\nLog, Relative Risk for Venous Thromboembolism")
Axes_obj_01.set_title("Source Data Review & Re-draw", size=11.4230, fontweight="normal")
Axes_obj_01.set_xlabel("Duration of Travel, hr")

# Basic Items #
# 外側に白の枠線を入れる
Axes_obj_01.hlines(y=-4.0, xmin=-1.0, xmax=21, color="white", linestyle="dotted", linewidth=0.4079)
Axes_obj_01.hlines(y=5.5, xmin=-1.0, xmax=21, color="white", linestyle="dotted", linewidth=0.4079)
Axes_obj_01.vlines(x=-1.0, ymin=-4.0, ymax=5.5, color="white", linestyle="dotted", linewidth=0.4079)
Axes_obj_01.vlines(x=21, ymin=-4.0, ymax=5.5, color="white", linestyle="solid", linewidth=0.4079)

for i in range(0, 11):
    Axes_obj_01.vlines(x=2.0 * i, ymin=-2.5, ymax=4.5, color="gray", linestyle="dotted", linewidth=0.5)

for i in range(0, 13):
    Axes_obj_01.hlines(y=-2.0 + 0.5 * i, xmin=-1.0, xmax=21, color="gray", linestyle="dotted", linewidth=0.5)

Axes_obj_01.hlines(y=0, xmin=-1.0, xmax=21.5, color="black", linestyle="solid", linewidth=0.5)
Axes_obj_01.text(18.20, 0.1, "Log(OR)=0", size=10, color="black")

#################################################################################################
# Display of Data (1): Scatter Plot(s) #
#################################################################################################
for i in range(0, len(AnalysisDataSet["TimeF"])):
    Axes_obj_01.scatter(AnalysisDataSet["TimeF"][i], AnalysisDataSet["LN(D)"][i],
                        color=AnalysisDataSet["labelC1"][i], zorder=10)
    Axes_obj_01.scatter(AnalysisDataSet["TimeF"][i], AnalysisDataSet["LN(U)"][i],
                        color=AnalysisDataSet["labelC1"][i], marker='_', zorder=10)
    Axes_obj_01.scatter(AnalysisDataSet["TimeF"][i], AnalysisDataSet["LN(L)"][i],
                        color=AnalysisDataSet["labelC1"][i], marker='_', zorder=10)
    Axes_obj_01.vlines(x=AnalysisDataSet["TimeF"][i],
                       ymin=AnalysisDataSet["LN(L)"][i], ymax=AnalysisDataSet["LN(U)"][i],
                       color=AnalysisDataSet["labelC1"][i], linestyle='solid', linewidth=1.0, zorder=10)
    Axes_obj_01.text(AnalysisDataSet["Lx"][i], AnalysisDataSet["Ly"][i],
                     AnalysisDataSet["Label"][i], size=8.0, color=AnalysisDataSet["labelC1"][i], zorder=10)

Axes_obj_01.scatter([0.0], [0.0], color="black")

#################################################################################################
# Martinelli Data & Oral Contraceptives #
#################################################################################################
Axes_obj_01.scatter(1.6, MartinelliDataSet["LN(OR)"][2], color="orange")
Axes_obj_01.scatter(1.6, MartinelliDataSet["LN(L)"][2], color="orange", marker='_')
Axes_obj_01.scatter(1.6, MartinelliDataSet["LN(U)"][2], color="orange", marker='_')
Axes_obj_01.vlines(x=1.6,
                   ymin=MartinelliDataSet["LN(L)"][2],
                   ymax=MartinelliDataSet["LN(U)"][2],
                   color="orange", linestyle='solid', linewidth=1.0)

#################################################################################################
# Display of Data (2): Plot(s) #
#################################################################################################
#################################################################################################
# Regression Line on Outline
#################################################################################################
Axes_obj_01.plot(X_Outline, Y_Outline_Line, color="red", linestyle="dashed", linewidth=0.4079)
Axes_obj_01.plot(X_Outline, Y_Outline_U, color="red", linestyle="dashed", linewidth=0.4079)
Axes_obj_01.plot(X_Outline, Y_OutLine_L, color="red", linestyle="dashed", linewidth=0.4079)

#################################################################################################
# Regression Line on Whole Data (Chandra's Line)
#################################################################################################
Axes_obj_01.plot(X_Line_C, Y_Line_C, color="black", linestyle="dashed", linewidth=1.0)

X = np.linspace(0.0, 20.0, 1000)
Y = 0.0853 * X_Line_C + 0.0044
Axes_obj_01.plot(X, Y, color="black", linestyle="solid", linewidth=1.0)
#################################################################################################
# Vertical Lines #
#################################################################################################
# Risk Group Outline
Axes_obj_01.vlines(x=np.min(X_Outline), ymin=Y_OutLine_L[0], ymax=Y_Outline_U[0], color="red",
                   linestyle="solid", linewidth=0.4079)
Axes_obj_01.vlines(x=np.max(X_Outline), ymin=Y_OutLine_L[-1], ymax=Y_Outline_U[-1], color="red",
                   linestyle="solid", linewidth=0.4079)

#################################################################################################
# Filling Aria #
#################################################################################################
# Latent Hyperbolic Pattern
Axes_obj_01.fill_between(X_Outline, Y_Outline_U, Y_OutLine_L, facecolor='red', alpha=0.05)

# Differential Curves
Line_fill = -4.0 * np.ones(1000)
Axes_obj_01.plot(X_A_S, Y_AS_Diff * 0.75 - 4.0, color="black", linestyle="solid", linewidth=0.5)
Axes_obj_01.fill_between(X_A_S, Y_AS_Diff * 0.75 - 4.0, Line_fill, facecolor="black", alpha=0.05)

#################################################################################################
# Annotations (1): Texts #
#################################################################################################
Axes_obj_01.text(6.85, -3.5, "?", size=11.4230, color="black")

#################################################################################################
# Annotations (2): Texts & Allows #
#################################################################################################
Axes_obj_01.annotate(
    text='\n'.join(["\"Any\" Types of Travel &", "Control is partner"]), xy=(9.5, -0.75 - 0.5),
    ha='center', va='center', size=10.0,
    bbox=dict(boxstyle='round', edgecolor='green', fc='white')
)
Axes_obj_01.hlines(y=-0.24, xmin=6.1, xmax=16.1, color='green', linestyle='solid', linewidth=1.0)
Axes_obj_01.vlines(x=6.1, ymin=-0.25 - 0.1, ymax=-0.25 + 0.1, color='green', linestyle='solid', linewidth=1.0)
Axes_obj_01.vlines(x=16.1, ymin=-0.25 - 0.1, ymax=-0.25 + 0.1, color='green', linestyle='solid', linewidth=1.0)
Axes_obj_01.vlines(x=11.0, ymin=-0.55 - 0.5, ymax=-0.25, color='green', linestyle='solid', linewidth=1.0)

#################################################################################################
Axes_obj_01.annotate(
    text='\n'.join(["Latent", "Hyperbolic", "Pattern ?"]),
    xy=(7.0 + 0.4, 2.25), xytext=(7.0 + 0.4, 3.5), ha='center', va='center', size=10.0,
    bbox=dict(boxstyle='round', edgecolor='red', fc='yellow'),
    arrowprops=dict(
        facecolor='black', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0, edgecolor='black', shrink=0.1)
)

#################################################################################################
Axes_obj_01.annotate(
    text='\n'.join(["Control: partner or friend"]),
    xy=(13, 2.75), xytext=(17.75, 4.25), ha='center', va='center', size=10.0,
    bbox=dict(boxstyle='round', edgecolor='darkorange', fc='white'),
    arrowprops=dict(
        facecolor='darkorange', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0, edgecolor='darkorange',
        shrink=0.1)
)

Axes_obj_01.annotate(
    text='\n'.join(["This Point's Value", "May Be Revised Downward"]),
    xy=(11.5, 4.25), xytext=(5.0, 5.0), ha='center', va='center', size=10.0,
    bbox=dict(boxstyle='round', edgecolor='black', fc='white'),
    arrowprops=dict(
        facecolor='black', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0, edgecolor='black', shrink=0.1)
)

Axes_obj_01.annotate(
    text='\n'.join(["Chandra et al. 2009", "Figure 3"]),
    xy=(18, 1.25), xytext=(18, -1.0), ha='center', va='center', size=10.0,
    bbox=dict(boxstyle='round', edgecolor='black', fc='white'),
    arrowprops=dict(
        facecolor='black', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0, edgecolor='black', shrink=0.1)
)

Axes_obj_01.annotate(
    text="Re-analysis",
    xy=(18, 1.6), xytext=(18, 3.4), ha='center', va='center', size=10.0,
    bbox=dict(boxstyle='round', edgecolor='black', fc='white'),
    arrowprops=dict(
        facecolor='black', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0, edgecolor='black', shrink=0.1)
)

Axes_obj_01.annotate(
    text='\n'.join(["Martinelli et al.", "Short flight: ", "Non-adjusted Value"]),
    xy=(1.6, 1.33), xytext=(1.6, 3.0), ha='center', va='center', size=10.0,
    bbox=dict(boxstyle='round', edgecolor='darkorange', fc='white'),
    arrowprops=dict(
        facecolor='darkorange', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0, edgecolor='darkorange',
        shrink=0.1)
)
#################################################################################################
# Annotations: Legend #
#################################################################################################
text1 = matplotlib.offsetbox.TextArea(Legend_K, textprops=dict(fontname="monospace", color="black", fontweight="bold"))
text2 = matplotlib.offsetbox.TextArea(Legend_P, textprops=dict(fontname="monospace", color="black", fontweight="bold"))
text3 = matplotlib.offsetbox.TextArea(Legend_C, textprops=dict(fontname="monospace", color="green", fontweight="bold"))
text4 = matplotlib.offsetbox.TextArea(Legend_M,
                                      textprops=dict(fontname="monospace", color="darkorange", fontweight="bold"))

texts_vbox = matplotlib.offsetbox.VPacker(children=[text1, text2, text3, text4],
                                          align="left", mode="fixed", pad=0.0, sep=0.0)
ann = matplotlib.offsetbox.AnnotationBbox(texts_vbox, box_alignment=(1, 0), xycoords="data",
                                          xy=(22 - 0.11 + 0.5, -4.25 - 0.15 + 0.02),
                                          bboxprops=dict(boxstyle='square', edgecolor='black', fc='white'),
                                          frameon=True)
Axes_obj_01.add_artist(ann)
############################################################################################################
# (b) "Estimated Distributions of Latent High-risk Populations"
############################################################################################################
# Axes_obj_02 = Figure_object.add_subplot(gs_1[1], xticks=ticks_x, yticks=ticks_y2)
# Basic information on this figure #
Axes_obj_02.set_ylabel("Relative Risk for Venous Thoromboembolism")
Axes_obj_02.set_title("Explore Latent Hyperbolic Patterns (OR + LN(CI))", size=11.4230, fontweight="normal")
Axes_obj_02.set_xlabel("Duration of Travel, hr")
Axes_obj_02.set_xlim(-2.0, 22.0)
Axes_obj_02.set_ylim(-4.55, 12.5)

for i in range(0, 11):
    Axes_obj_02.vlines(x=2.0 * i, ymin=-1.5, ymax=10.5, color="gray", linestyle="dotted", linewidth=0.5)

for i in range(0, 23):
    Axes_obj_02.hlines(y=-1.0 + 0.5 * i, xmin=-1.0, xmax=21, color="gray", linestyle="dotted", linewidth=0.5)

Axes_obj_02.hlines(y=1, xmin=-1.0, xmax=21, color="black", linestyle="solid", linewidth=0.5, zorder=1)
Axes_obj_02.text(18.20, 1.1, "OR=1", size=10, color="black")

Axes_obj_02.hlines(y=OC_Affected, xmin=-1.0, xmax=21, color="black", linestyle="solid", linewidth=0.5, zorder=1)

X = [1.95, 4.5, 5.95, 8.0, 9.95, 11.8, 12.10, 13.0, 15.75, 19.00]
Y = [1.958145 + 0.25, 3.242043 + 0.25, 2.569788 + 0.25, 3.097942 + 0.25, 2.378238 + 0.25, 9.842287 + 0.25,
     4.152680 + 0.25, 6.149990 + 0.25, 3.564874 + 0.25, 6.762894 + 0.25]

# Display of Data (1): Scatter Plot(s) #
for i in range(0, len(AnalysisDataSet["TimeF"])):
    Axes_obj_02.scatter(AnalysisDataSet["TimeF"][i], AnalysisDataSet["Data"][i],
                        color=AnalysisDataSet["labelC2"][i], zorder=10)
    Axes_obj_02.scatter(AnalysisDataSet["TimeF"][i], AnalysisDataSet["OR + Abs( LN(U) - LN(OR) )"][i],
                        color=AnalysisDataSet["labelC2"][i], marker='_', zorder=10)
    Axes_obj_02.scatter(AnalysisDataSet["TimeF"][i], AnalysisDataSet["OR - Abs( LN(L) - LN(OR) )"][i],
                        color=AnalysisDataSet["labelC2"][i], marker='_', zorder=10)
    Axes_obj_02.vlines(x=AnalysisDataSet["TimeF"][i],
                       ymin=AnalysisDataSet["OR - Abs( LN(L) - LN(OR) )"][i],
                       ymax=AnalysisDataSet["OR + Abs( LN(U) - LN(OR) )"][i],
                       color=AnalysisDataSet["labelC2"][i], linestyle='solid', linewidth=1.0, zorder=10)
    Axes_obj_02.text(X[i], Y[i],
                     AnalysisDataSet["Label2"][i], size=8.0, color=AnalysisDataSet["labelC2"][i], zorder=10)

#################################################################################################
# Regression Curves on Risk Group A
#################################################################################################
Axes_obj_02.plot(X_A_S, Y_AS, color="red", linestyle="solid", linewidth=1.0, zorder=10)
Axes_obj_02.plot(X_A_CI, Y_AU, color="red", linestyle="solid", linewidth=0.4079, zorder=10)
Axes_obj_02.plot(X_A_CI, Y_AL, color="red", linestyle="solid", linewidth=0.4079, zorder=10)
Axes_obj_02.plot(X_A_S, Y_AS_Diff - 3.5, color="red", linestyle="solid", linewidth=1.0, zorder=10)

#################################################################################################
# Regression Curves on Risk Group B
#################################################################################################
Axes_obj_02.plot(X_B_S, Y_BS, color="blue", linestyle="solid", linewidth=1.0, zorder=10)
Axes_obj_02.plot(X_B_CI, Y_BU, color="blue", linestyle="solid", linewidth=0.4079, zorder=10)
Axes_obj_02.plot(X_B_CI, Y_BL, color="blue", linestyle="solid", linewidth=0.4079, zorder=10)
Axes_obj_02.plot(X_B_S, Y_BS_Diff - 3.5, color="blue", linestyle="solid", linewidth=1.0, zorder=10)

#################################################################################################
# Regression Curves on Risk Group C
#################################################################################################
Axes_obj_02.plot(X_GroupC_Line, Y_GroupC_Line, color="limegreen", linestyle="dashed", linewidth=1.0, zorder=8)

#################################################################################################
# Regression Curves on Risk Group Martinelli
#################################################################################################
Axes_obj_02.plot(X_GroupMartinelli_Line, Y_GroupMartinelli_Line, color="orange", linestyle="dashed", linewidth=1.0,
                 zorder=8)

#################################################################################################
# Risk Group A & Risk Group B
#################################################################################################
Axes_obj_02.plot(X_Line_W, Y_Total_Diff - 3.5 + 0.025, color="black", linestyle="solid", linewidth=0.5)

#################################################################################################
# Vertical Lines #
#################################################################################################
# Risk Group A
Axes_obj_02.vlines(x=np.min(X_A_CI), ymin=Y_AL[0], ymax=Y_AU[0], color="red", linestyle="solid", linewidth=0.4079)
Axes_obj_02.vlines(x=np.max(X_A_CI), ymin=Y_AL[-1], ymax=Y_AU[-1], color="red", linestyle="solid", linewidth=0.4079)

# Risk Group B
Axes_obj_02.vlines(x=np.min(X_B_CI), ymin=Y_BL[0], ymax=Y_BU[0], color="blue", linestyle="solid", linewidth=0.4079)
Axes_obj_02.vlines(x=np.max(X_B_CI), ymin=Y_BL[-1], ymax=Y_BU[-1], color="blue", linestyle="solid", linewidth=0.4079)

#################################################################################################
# Filling Aria #
#################################################################################################
# S Curves: Risk Group A & Risk Group B
Axes_obj_02.fill_between(X_A_CI, Y_AU, Y_AL, facecolor='red', alpha=0.05)
Axes_obj_02.fill_between(X_B_CI, Y_BU, Y_BL, facecolor='blue', alpha=0.05)

# Differential Curves: Risk Group A & Risk Group B
Line_fill = -3.5 * np.ones(1000)
Axes_obj_02.fill_between(X_A_S, Y_AS_Diff - 3.5, Line_fill, facecolor="red", alpha=0.05)
Axes_obj_02.fill_between(X_B_S, Y_BS_Diff - 3.5, Line_fill, facecolor="blue", alpha=0.05)

#################################################################################################
# Martinelli Data & Oral Contraceptives #
#################################################################################################
Axes_obj_02.scatter(1.8, MartinelliDataSet["OR"][2], color="orange")
Axes_obj_02.scatter(1.8, MartinelliDataSet["OR - Abs( LN(L) - LN(OR) )"][2], color="orange", marker='_')
Axes_obj_02.scatter(1.8, MartinelliDataSet["OR + Abs( LN(L) - LN(OR) )"][2], color="orange", marker='_')
Axes_obj_02.vlines(x=1.8,
                   ymin=MartinelliDataSet["OR - Abs( LN(L) - LN(OR) )"][2],
                   ymax=MartinelliDataSet["OR + Abs( LN(L) - LN(OR) )"][2],
                   color="orange", linestyle='solid', linewidth=1.0)

#################################################################################################
# Annotations (1): Texts #
#################################################################################################
Axes_obj_02.annotate(
    text='\n'.join(["High-risk", "Group A"]),
    xy=(7.0 + 2.0, -3.25 + 1.0), xytext=(9.0 + 2.0, -2.0 + 1.0 - 0.25), ha='left', va='center', size=10.0,
    bbox=dict(boxstyle='round', edgecolor='red', fc='white'),
    arrowprops=dict(
        facecolor='red', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0, edgecolor='red', shrink=0.1)
)
#################################################################################################
Axes_obj_02.annotate(
    text='\n'.join(["High-risk", "Group B"]),
    xy=(7.0 + 4.0 + 2.0, -3.25 - 0.125 + 0.75), xytext=(9.0 + 4.0 + 2.0, -2.0 - 0.125 + 0.75 - 0.25), ha='left',
    va='center', size=10.0,
    bbox=dict(boxstyle='round', edgecolor='blue', fc='white'),
    arrowprops=dict(
        facecolor='blue', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0, edgecolor='blue', shrink=0.1)
)
#################################################################################################
Axes_obj_02.annotate(
    text='\n'.join(["Hyperbolic", "Pattern A"]),
    xy=(7, 2.5 + 5.5), xytext=(7.0, 3.5 + 5.5), ha='center', va='center', size=10.0,
    bbox=dict(boxstyle='round', edgecolor='red', fc='white'),
    arrowprops=dict(
        facecolor='red', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0, edgecolor='red', shrink=0.1)
)
#################################################################################################
Axes_obj_02.annotate(
    text='\n'.join(["Hyperbolic", "Pattern B"]),
    xy=(16.5, 2.65 + 4.5), xytext=(16.5, 3.65 + 4.5), ha='center', va='center', size=10.0,
    bbox=dict(boxstyle='round', edgecolor='blue', fc='white'),
    arrowprops=dict(
        facecolor='blue', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0, edgecolor='blue', shrink=0.1)
)
#################################################################################################
Axes_obj_02.annotate(
    text='\n'.join(["Mean of Data (\"C\" & \"M\")", "Control: partner"]),
    xy=(16.75, 2.28), xytext=(16.75, 0.125), ha='center', va='center', size=10.0,
    bbox=dict(boxstyle='round', edgecolor='black', fc='white'),
    arrowprops=dict(
        facecolor='black', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0, edgecolor='black', shrink=0.1)
)

# Axes_obj_02.annotate(
# text='\n'.join(["Mean of Data (\"C\" & \"M\")", "Affected by OC?"]),
#    xy=(16.75, 2.28), xytext=(16.75, 0.125), ha='center', va='center', size=10.0,
#    bbox=dict(boxstyle='round', edgecolor='black', fc='white'),
#    arrowprops=dict(
#        facecolor='black', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0, edgecolor='black', shrink=0.1)
# )

#################################################################################################
Inflection_point_A = CoefficientDataSet.at["Reg. Result on A (S Curve)         ", "M (Mean)"]
Inflection_point_B = CoefficientDataSet.at["Reg. Result on B (S Curve)         ", "M (Mean)"]

Axes_obj_02.vlines(x=Inflection_point_A, ymin=-3.5, ymax=4.5, color='red', linestyle='dashed', linewidth=0.5)
Axes_obj_02.vlines(x=Inflection_point_B, ymin=-3.5, ymax=4.5, color='blue', linestyle='dashed', linewidth=0.5)

s_Inflection_point_A_0 = Decimal(str(Inflection_point_A)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)
s_Inflection_point_B_0 = Decimal(str(Inflection_point_B)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)
s_Inflection_point_A = ''.join([str(s_Inflection_point_A_0), ' h'])
s_Inflection_point_B = ''.join([str(s_Inflection_point_B_0), ' h'])

Axes_obj_02.annotate(text=s_Inflection_point_A,
                     bbox=dict(boxstyle='round', edgecolor='red', fc='white'),
                     xy=(Inflection_point_A, -4.0), ha='center', va='center')

Axes_obj_02.annotate(text=s_Inflection_point_B,
                     bbox=dict(boxstyle='round', edgecolor='blue', fc='white'),
                     xy=(Inflection_point_B, -4.0), ha='center', va='center')

#################################################################################################
# Annotations (2): Texts & Allows #
#################################################################################################
Axes_obj_02.text(5.0, 11.5, "Time < 10 h", size=10, color="black", ha='center', va='center',
                 bbox=dict(boxstyle='round', edgecolor='black', fc='white'))
Axes_obj_02.text(15.0, 11.5, "Time > 10 h", size=10, color="black", ha='center', va='center',
                 bbox=dict(boxstyle='round', edgecolor='black', fc='white'))

Axes_obj_02.quiver(3.0, 11.5, -3.0, 0.0, scale_units='xy', angles='xy', scale=1, linestyle='solid', linewidth=0.4079)
Axes_obj_02.quiver(5.0, 11.5, 5.0, 0.0, scale_units='xy', angles='xy', scale=1, linestyle='solid', linewidth=0.4079)
Axes_obj_02.quiver(13.0, 11.5, -3.0, 0.0, scale_units='xy', angles='xy', scale=1, linestyle='solid', linewidth=0.4079)
Axes_obj_02.quiver(17.0, 11.5, 3.0, 0.0, scale_units='xy', angles='xy', scale=1, linestyle='solid', linewidth=0.4079)
Axes_obj_02.vlines(x=10.0, ymin=-3.5, ymax=12.0, color="black", linestyle='solid', linewidth=1.0)
Axes_obj_02.vlines(x=0.0, ymin=11.0, ymax=12.0, color="black", linestyle='solid', linewidth=1.0)
Axes_obj_02.vlines(x=20.0, ymin=11.0, ymax=12.0, color="black", linestyle='solid', linewidth=1.0)

########################################################################################################################
# "(c) Cannegieter et al. 2006 Figure 1":
########################################################################################################################
########################################################################################################################
# Making Data Sets
########################################################################################################################
AnalysisDataSet = pd.DataFrame({
    "Week": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "Day": [4 + (7 * 0), 4 + (7 * 1), 4 + (7 * 2), 4 + (7 * 3), 4 + (7 * 4), 4 + (7 * 5), 4 + (7 * 6), 4 + (7 * 7),
            4 + (7 * 8), 4 + (7 * 9), 4 + (7 * 10), 4 + (7 * 11)],
    "Range": [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    "Day Label": ["Day 4(1-7)   ", "Day 11(8-14) ", "Day 18(15-21)", "Day 25(22-28)", "Day 32(29-35)", "Day 39(36-42)",
                  "Day 46(43-49)", "Day 53(50-56)", "Day 60(57-63)", "Day 67(64-70)", "Day 74(71-77)", "Day 81(78-84)"],
    "Cases": [68, 38, 24, 30, 30, 18, 10, 16, 13, 10, 11, 6]},
    index=["Week 1", "Week 2", "Week 3", "Week 4", "Week 5", "Week 6", "Week 7", "Week 8", "Week 9", "Week 10",
           "Week 11", "Week 12"])
print(AnalysisDataSet)


########################################################################################################################
# Regression Analysis
########################################################################################################################
############################################################################################################
#    Definition of Regression Function
# n1 * np.exp(-l1 * (x-b)) + (n2 * np.exp(-l2 * (x-b))) * (np.sin(a * (x - b)))
# n1 * exp(-l1 * (x-b)) + n2 * exp(-l2 * (x-b)) * sin(a * (x-b))
# n1 * exp(-l1 * (x-b)) + (n2 * exp(-l2 * (x-b))) * (sin(a * (x - b)))
############################################################################################################

def reg_func(parameter, x, y):
    n1 = parameter[0]
    l1 = parameter[1]
    n2 = parameter[2]
    l2 = parameter[3]
    a = parameter[4]
    b = parameter[5]
    _Output_ = y - (n1 * np.exp(-l1 * (x - b)) + (n2 * np.exp(-l2 * (x - b))) * (np.sin(a * (x - b))))
    return _Output_


############################################################################################################
#    Non-linear Regression Analysis (Weeks)
############################################################################################################
x = AnalysisDataSet["Week"]
y = AnalysisDataSet["Cases"]
parameter_0 = [50, 0.1, 25, 0.1, 1.0, 0.0]
RegResult_Original_W = optimize.leastsq(reg_func, parameter_0, args=(x, y), full_output=True)
Result_W_0 = RegResult_Original_W[0]
Result_W = [Result_W_0[0], Result_W_0[1], Result_W_0[2], Result_W_0[3], Result_W_0[4], Result_W_0[5]]

print("")
print("########## Evaluation of Result on the Regression Analysis ##########")
print(["Reg. Analysis Result:", RegResult_Original_W[-1], "1, 2, 3 or 4, the solution was found."])
# print(Result_W)

############################################################################################################
#    Non-linear Regression Analysis (Days)
############################################################################################################
x = AnalysisDataSet["Day"]
y = AnalysisDataSet["Cases"]
parameter_0 = [50, 0.03, 25, 0.03, 0.1, 0.0]
RegResult_Original_D = optimize.leastsq(reg_func, parameter_0, args=(x, y), full_output=True)
Result_D_0 = RegResult_Original_D[0]
Result_D = [Result_D_0[0], Result_D_0[1], Result_D_0[2], Result_D_0[3], Result_D_0[4], Result_D_0[5]]
print(["Reg. Analysis Result:", RegResult_Original_W[-1], "1, 2, 3 or 4, the solution was found."])
# print(Result_D)

############################################################################################################
# Coefficient Matrix (Data Set)
############################################################################################################
CoefficientDataSet = pd.DataFrame([Result_W, Result_D],
                                  index=["Reg. Result (Week)", "Reg. Result (Day)"],
                                  columns=["n1", "l1", "n2", "l2", "a", "b"]
                                  )
print("")
print("########## Result of Non-linear Regression Analysis (CoefficientDataSet) ##########")
print(CoefficientDataSet)

CoefficientDataSet_round = [Decimal(str(CoefficientDataSet.iat[1, 0])).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP),
                            Decimal(str(CoefficientDataSet.iat[1, 1])).quantize(Decimal('0.0001'),
                                                                                rounding=ROUND_HALF_UP),
                            Decimal(str(CoefficientDataSet.iat[1, 2])).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP),
                            Decimal(str(CoefficientDataSet.iat[1, 3])).quantize(Decimal('0.0001'),
                                                                                rounding=ROUND_HALF_UP),
                            Decimal(str(CoefficientDataSet.iat[1, 4])).quantize(Decimal('0.001'),
                                                                                rounding=ROUND_HALF_UP),
                            Decimal(str(CoefficientDataSet.iat[1, 5])).quantize(Decimal('0.01'),
                                                                                rounding=ROUND_HALF_UP)]

print("")
print("########## Rounded Result of Non-linear Regression Analysis (CoefficientDataSet_round) ##########")
print(CoefficientDataSet_round)
########################################################################################################################
# Values for Curves
########################################################################################################################
# Composite Function:                                       CF
# Monotonic Decrease Function (Exponential　Function):      MF
# Periodic Function (Trigonometric Function):               PF
# Difference Function:                                      DF
# Envelope Function (Upper):                                EF_U
# Envelope Function (Lower):                                EF_L

########################################################################################################################
# Values for Curves (Week)
########################################################################################################################
n1 = CoefficientDataSet.at["Reg. Result (Week)", "n1"]
l1 = CoefficientDataSet.at["Reg. Result (Week)", "l1"]
n2 = CoefficientDataSet.at["Reg. Result (Week)", "n2"]
l2 = CoefficientDataSet.at["Reg. Result (Week)", "l2"]
a = CoefficientDataSet.at["Reg. Result (Week)", "a"]
b = CoefficientDataSet.at["Reg. Result (Week)", "b"]
X_Week = np.linspace(0.25, 18.5, 1000)
Y_Week = n1 * np.exp(-l1 * (X_Week - b)) + (n2 * np.exp(-l2 * (X_Week - b))) * (np.sin(a * (X_Week - b)))

########################################################################################################################
# Values for Curves (Day)
########################################################################################################################
n1 = CoefficientDataSet.at["Reg. Result (Day)", "n1"]
l1 = CoefficientDataSet.at["Reg. Result (Day)", "l1"]
n2 = CoefficientDataSet.at["Reg. Result (Day)", "n2"]
l2 = CoefficientDataSet.at["Reg. Result (Day)", "l2"]
a = CoefficientDataSet.at["Reg. Result (Day)", "a"]
b = CoefficientDataSet.at["Reg. Result (Day)", "b"]

X_Day = np.linspace(-50, 200, 1000)
Y_CF = n1 * np.exp(-l1 * (X_Day - b)) + (n2 * np.exp(-l2 * (X_Day - b))) * (np.sin(a * (X_Day - b)))
Y_MF = n1 * np.exp(-l1 * (X_Day - b)) + (0 * np.exp(-0 * (X_Day - b))) * (np.sin(0 * (X_Day - b)))
Y_PF = 0 * np.exp(-0 * (X_Day - b)) + (n2 * np.exp(-l2 * (X_Day - b))) * (np.sin(a * (X_Day - b)))
Y_DF = AnalysisDataSet["Cases"] - (n1 * np.exp(-l1 * (AnalysisDataSet["Day"] - b)))
Y_EF_U = 0 * np.exp(-0 * (X_Day - b)) + (n2 * np.exp(-l2 * (X_Day - b)))
Y_EF_L = 0 * np.exp(-0 * (X_Day - b)) - (n2 * np.exp(-l2 * (X_Day - b)))
Y_EF_U2 = n1 * np.exp(-l1 * (X_Day - b)) + (n2 * np.exp(-l2 * (X_Day - b)))
Y_EF_L2 = n1 * np.exp(-l1 * (X_Day - b)) - (n2 * np.exp(-l2 * (X_Day - b)))
Line_fill = 0.0 * np.ones(1000)

Peaks_1 = [(b + (np.pi / 2) / a),
           (b + (np.pi / 2) / a) + 2 * np.pi / a,
           (b + (np.pi / 2) / a) + 2 * np.pi / a + 2 * np.pi / a,
           (b + (np.pi / 2) / a) + 2 * np.pi / a + 2 * np.pi / a + 2 * np.pi / a]

Peaks_1_round = [Decimal(str(Peaks_1[0])).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP),
                 Decimal(str(Peaks_1[1])).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP),
                 Decimal(str(Peaks_1[2])).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP),
                 Decimal(str(Peaks_1[3])).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)]

Values = [n1 * np.exp(-l1 * (30 - b)) + (n2 * np.exp(-l2 * (30 - b))) * (np.sin(a * (30 - b))),
          n1 * np.exp(-l1 * (60 - b)) + (n2 * np.exp(-l2 * (60 - b))) * (np.sin(a * (60 - b))),
          n1 * np.exp(-l1 * (90 - b)) + (n2 * np.exp(-l2 * (90 - b))) * (np.sin(a * (90 - b))),
          n1 * np.exp(-l1 * (120 - b)) + (n2 * np.exp(-l2 * (120 - b))) * (np.sin(a * (120 - b))),
          n1 * np.exp(-l1 * (150 - b)) + (n2 * np.exp(-l2 * (150 - b))) * (np.sin(a * (150 - b))),
          n1 * np.exp(-l1 * (180 - b)) + (n2 * np.exp(-l2 * (180 - b))) * (np.sin(a * (180 - b)))]

Values_round = [Decimal(str(Values[0])).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP),
                Decimal(str(Values[1])).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP),
                Decimal(str(Values[2])).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP),
                Decimal(str(Values[3])).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP),
                Decimal(str(Values[4])).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP),
                Decimal(str(Values[5])).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)]

Period = 2 * np.pi / a
Period_round = Decimal(str(Period)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)

########################################################################################################################
# Values1
########################################################################################################################
print("")
print("########## Peaks (Peaks_1) #######################")
print(Peaks_1)
print("")
print("########## Round Peaks (Peaks_1_round) ###########")
print(Peaks_1_round)
print("")
print("########## Values (Values) #######################")
print(Values)
print("")
print("########## Round Values (Values_round) ###########")
print(Values_round)
print("")
print("########## Period (Period) #######################")
print(Period)
print("")
print("########## Round Period (Period_round) ###########")
print(Period_round)

########################################################################################################################
# Values2
########################################################################################################################

Unit_A1 = (n1 / l1) * np.exp(l1 * b)
Unit_B1 = (n2 / (l2 ** 2 + a ** 2)) * (l2 * np.sin(-a * b) + a * np.cos(-a * b))
S = Unit_A1 + Unit_B1
S1 = (n1 / l1) * np.exp(l1 * b) - (n2 / l2) * np.exp(l2 * b)
S2 = S - S1
Percent_S2 = (S2 / S) * 100

List_S = [S1, S2, S, Percent_S2]

s = 56
Unit_B2_1 = l2 * np.sin(a * (s - b)) + a * np.cos(a * (s - b))
Unit_B2_2 = l2 * np.sin(-a * b) + a * np.cos(-a * b)
Unit_B2 = (n2 / (l2 ** 2 + a ** 2)) * (Unit_B2_1 - Unit_B2_2)
Unit_A2 = -(n1 / l1) * (np.exp(-l1 * (s - b)) - np.exp(l1 * b))
S_w = Unit_A2 + Unit_B2
S1_w = -(n1 / l1) * (np.exp(-l1 * (s - b)) - np.exp(l1 * b)) + (n2 / l2) * (np.exp(-l2 * (s - b)) - np.exp(l2 * b))
S2_w = S_w - S1_w
Percent_S2_w = (S2_w / S_w) * 100

List_S_w = [S1_w, S2_w, S_w, Percent_S2_w]

IntegralDataSet = pd.DataFrame([List_S, List_S_w],
                               index=["Integral Value (Interval: 0 - Infinity)",
                                      "Integral Value (Interval: 0 - 8 Weeks)"],
                               columns=["S1", "S2", "S", "Percent of S2"])
print("")
print("########## Result of Integral Computation (IntegralDataSet) #######################")
print(IntegralDataSet)
########################################################################################################################
# (c)
########################################################################################################################
# im = Image.open("./achan.jpg") im.show()
# Axes_obj_03 = Figure_object.add_subplot(gs_2[0])
# Axes_obj_03.set_title("Cannegieter et al. 2006", size=11.4230, fontweight="normal")
# Axes_obj_03.yaxis.set_visible(False)
# Axes_obj_03.xaxis.set_visible(False)
# img = plt.imread("journal.pmed.0030307.g001.png")
# Axes_obj_03.imshow(img)

############################################################################################################
# "(4) Added Damped Wave":
############################################################################################################
# Axes_obj_04 = Figure_object.add_subplot(gs_2[1])
# Axes_obj_04 .tick_params(direction="in")
# Axes_obj_04.set_title("Non-recognized Pattern", size=11.4230, fontweight="normal")
# Axes_obj_04.set_xlabel('Day')
# Axes_obj_04.set_ylabel('Cases')
# Axes_obj_04.set_xlim(0.0, 190)
# Axes_obj_04.set_ylim(0.0, 85)
# Axes_obj_04.set_xticks([0, 30, 60, 90, 120,  150, 180])
# Axes_obj_04.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180])
# Axes_obj_04.set_yticks([0, 20, 40, 60, 80, 85])
# Axes_obj_04.set_yticklabels(["0", "20", "40", "60", "80", ""])
# Axes_obj_04.bar(AnalysisDataSet["Day"], AnalysisDataSet["Cases"], color='none', width=5.0, alpha=1.0
#                , edgecolor="black", linestyle="solid", linewidth=1.0, zorder=1)

# Axes_obj_04.plot(X_Day, Y_EF_U2, color='black', linestyle='dashed', linewidth=0.5)
# Axes_obj_04.plot(X_Day, Y_CF, color='red', linestyle='solid', linewidth=1.5)
# Axes_obj_04.plot(X_Day, Y_EF_L2, color='blue', linestyle='dashed', linewidth=1.0)
# Axes_obj_04.scatter(AnalysisDataSet["Day"], AnalysisDataSet["Cases"], color='black', s=15, zorder=10)
# Axes_obj_04.fill_between(X_Day, Y_CF, Y_EF_L2, facecolor='red', alpha=0.3)
# Axes_obj_04.fill_between(X_Day, Y_EF_L2, Line_fill, facecolor='blue', alpha=0.3)

# Annotations (1): Texts #

# y_1 = n1 * np.exp(-l1 * (35-b)) - (n2 * np.exp(-l2 * (35-b)))
# y_2 = n1 * np.exp(-l1 * (8.5-b)) + (n2 * np.exp(-l2 * (8.5-b))) * (np.sin(a * (8.5-b)))

# Axes_obj_04.annotate(
# text="Y = f(x)",
#    xy=(8.5, y_2), xytext=(8.5+5.0, y_2+22.5), ha='left', va='center', size=11.4230, color="red",
#    bbox=dict(boxstyle='round', edgecolor='white', fc='white'),
#    arrowprops=dict(
#        facecolor='red', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0, edgecolor='red', shrink=0.1)
# )

# Axes_obj_04.annotate(
# text="Y = g(x)",
#    xy=(33, y_1), xytext=(33+6.0, y_1+22.5), ha='left', va='center', size=11.4230, color="blue",
#    bbox=dict(boxstyle='round', edgecolor='white', fc='white'),
#    arrowprops=dict(
#        facecolor='blue', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0, edgecolor='blue', shrink=0.1)
# )

# Annotations (2): Texts & Allows #
#################################################################################################
# Axes_obj_04.annotate(text=Values_round[2], xy=(90, Values[2]+2.5), xytext=(90, Values[2]+22.5),
#                     ha='center', va='center', size=10.0,
#                     bbox=dict(boxstyle='round', edgecolor='black', fc='white'),
#                     arrowprops=dict(facecolor='black', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0,
#                                      edgecolor='black', shrink=0.1))

# Axes_obj_04.annotate(text=Values_round[3], xy=(120, Values[3]+2.5), xytext=(120, Values[3]+22.5),
#                     ha='center', va='center', size=10.0,
#                     bbox=dict(boxstyle='round', edgecolor='black', fc='white'),
#                     arrowprops=dict(facecolor='black', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0,
#                                      edgecolor='black', shrink=0.1))

# Axes_obj_04.annotate(text=Values_round[4], xy=(150, Values[4]+2.5), xytext=(150, Values[4]+22.5),
#                     ha='center', va='center', size=10.0,
#                     bbox=dict(boxstyle='round', edgecolor='black', fc='white'),
#                     arrowprops=dict(facecolor='black', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0,
#                                      edgecolor='black', shrink=0.1))

# Axes_obj_04.annotate(text=Values_round[5], xy=(180, Values[5]+2.5), xytext=(180, Values[5]+22.5),
#                     ha='center', va='center', size=10.0,
#                     bbox=dict(boxstyle='round', edgecolor='black', fc='white'),
#                     arrowprops=dict(facecolor='black', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0,
#                                      edgecolor='black', shrink=0.1))

############################################################################################################
# Equations
############################################################################################################
# \left( …… \right)
# \left{a\left(x-b\right) \right}

# Axes_obj_04.text(67.5, 85-15,
#                 r"$f(x) = n_{1}e^{-\lambda_{1}(x-b)}+n_{2}e^{-\lambda_{2}(x-b)}\sin{\{a\left(x-b\right)\}}$",
#                 horizontalalignment='left', fontsize=14.6848, color="red")
# Axes_obj_04.text(67.5, 65-15,
#                 r"$g(x) = n_{1}e^{-\lambda_{1}(x-b)}-n_{2}e^{-\lambda_{2}(x-b)}$",
#                 horizontalalignment='left', fontsize=14.6848, color="blue")

########################################################################################################################
# Figure_object.tight_layout()
my_path = os.path.abspath("Figures")
plt.savefig(os.path.join(my_path, "00_Kimoto_et_al_(2023)_[_Fig_03_].png"))

img = Image.open(os.path.join(my_path, "00_Kimoto_et_al_(2023)_[_Fig_03_].png"))
img_resize = img.resize(size=(2866, 1512))
my_path2 = os.path.abspath("Figures/Fixed_Version/B6")
img_resize.save(os.path.join(my_path2, "00_Kimoto_et_al_(2023)_[_Fig_03_]_B6.png"))

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

# plt.show()
