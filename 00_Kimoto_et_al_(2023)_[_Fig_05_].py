########################################################################################################################
# Fig. 5. Cyclic pattern of thrombosis onset appeared in a figure by Kelman et al.(00_Kimoto_et_al_(2023)_[_Fig_05_].py)
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
# Kelman CW, Kortt MA, Becker NG, Li Z, Mathews JD, Guest CS, Holman CD.
# Deep vein thrombosis and air travel: record linkage study.
# BMJ. 2003 Nov 8;327(7423):1072.
# doi: 10.1136/bmj.327.7423.1072.
# PMID: 14604926; PMCID: PMC261739.
# Setting: Western Australia.
# Participants: 5408 patients admitted to hospital with
# venous thromboembolism and matched with data for
# arrivals of international flights during 1981-99.
#
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
############################
import os
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option("display.width", 300)
import matplotlib.ticker as ticker
################################
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
print("")
print("###############################################################################################################")
print("### Kelman et al. Data 01#")
print("###############################################################################################################")
Point_001 = ["001", "01", 0, 3, 1, 3, 30, 4.285714286]
Point_002 = ["002", "01", 1, 3, 1, 9, 30, 4.285714286]
Point_003 = ["003", "01", 2, 3, 1, 7, 30, 4.285714286]
Point_004 = ["004", "01", 3, 3, 1, 2, 30, 4.285714286]
Point_005 = ["005", "01", 4, 3, 1, 6, 30, 4.285714286]
Point_006 = ["006", "01", 5, 3, 1, 2, 30, 4.285714286]
Point_007 = ["007", "01", 6, 3, 1, 1, 30, 4.285714286]
Point_008 = ["008", "02", 7, 10, 2, 2, 15, 2.142857143]
Point_009 = ["009", "02", 8, 10, 2, 3, 15, 2.142857143]
Point_010 = ["010", "02", 9, 10, 2, 1, 15, 2.142857143]
Point_011 = ["011", "02", 10, 10, 2, 1, 15, 2.142857143]
Point_012 = ["012", "02", 11, 10, 2, 3, 15, 2.142857143]
Point_013 = ["013", "02", 12, 10, 2, 2, 15, 2.142857143]
Point_014 = ["014", "02", 13, 10, 2, 3, 15, 2.142857143]
Point_015 = ["015", "03", 14, 17, 3, 1, 8, 1.142857143]
Point_016 = ["016", "03", 15, 17, 3, 1, 8, 1.142857143]
Point_017 = ["017", "03", 16, 17, 3, 0, 8, 1.142857143]
Point_018 = ["018", "03", 17, 17, 3, 5, 8, 1.142857143]
Point_019 = ["019", "03", 18, 17, 3, 1, 8, 1.142857143]
Point_020 = ["020", "03", 19, 17, 3, 0, 8, 1.142857143]
Point_021 = ["021", "03", 20, 17, 3, 0, 8, 1.142857143]
Point_022 = ["022", "04", 21, 24, 4, 1, 12, 1.714285714]
Point_023 = ["023", "04", 22, 24, 4, 2, 12, 1.714285714]
Point_024 = ["024", "04", 23, 24, 4, 3, 12, 1.714285714]
Point_025 = ["025", "04", 24, 24, 4, 2, 12, 1.714285714]
Point_026 = ["026", "04", 25, 24, 4, 2, 12, 1.714285714]
Point_027 = ["027", "04", 26, 24, 4, 1, 12, 1.714285714]
Point_028 = ["028", "04", 27, 24, 4, 1, 12, 1.714285714]
Point_029 = ["029", "05", 28, 31, 5, 2, 8, 1.142857143]
Point_030 = ["030", "05", 29, 31, 5, 0, 8, 1.142857143]
Point_031 = ["031", "05", 30, 31, 5, 2, 8, 1.142857143]
Point_032 = ["032", "05", 31, 31, 5, 2, 8, 1.142857143]
Point_033 = ["033", "05", 32, 31, 5, 2, 8, 1.142857143]
Point_034 = ["034", "05", 33, 31, 5, 0, 8, 1.142857143]
Point_035 = ["035", "05", 34, 31, 5, 0, 8, 1.142857143]
Point_036 = ["036", "06", 35, 38, 6, 1, 8, 1.142857143]
Point_037 = ["037", "06", 36, 38, 6, 1, 8, 1.142857143]
Point_038 = ["038", "06", 37, 38, 6, 0, 8, 1.142857143]
Point_039 = ["039", "06", 38, 38, 6, 3, 8, 1.142857143]
Point_040 = ["040", "06", 39, 38, 6, 2, 8, 1.142857143]
Point_041 = ["041", "06", 40, 38, 6, 1, 8, 1.142857143]
Point_042 = ["042", "06", 41, 38, 6, 0, 8, 1.142857143]
Point_043 = ["043", "07", 42, 45, 7, 1, 7, 1]
Point_044 = ["044", "07", 43, 45, 7, 3, 7, 1]
Point_045 = ["045", "07", 44, 45, 7, 0, 7, 1]
Point_046 = ["046", "07", 45, 45, 7, 1, 7, 1]
Point_047 = ["047", "07", 46, 45, 7, 1, 7, 1]
Point_048 = ["048", "07", 47, 45, 7, 0, 7, 1]
Point_049 = ["049", "07", 48, 45, 7, 1, 7, 1]
Point_050 = ["050", "08", 49, 52, 8, 1, 8, 1.142857143]
Point_051 = ["051", "08", 50, 52, 8, 1, 8, 1.142857143]
Point_052 = ["052", "08", 51, 52, 8, 1, 8, 1.142857143]
Point_053 = ["053", "08", 52, 52, 8, 0, 8, 1.142857143]
Point_054 = ["054", "08", 53, 52, 8, 0, 8, 1.142857143]
Point_055 = ["055", "08", 54, 52, 8, 2, 8, 1.142857143]
Point_056 = ["056", "08", 55, 52, 8, 3, 8, 1.142857143]
Point_057 = ["057", "09", 56, 59, 9, 2, 9, 1.285714286]
Point_058 = ["058", "09", 57, 59, 9, 1, 9, 1.285714286]
Point_059 = ["059", "09", 58, 59, 9, 0, 9, 1.285714286]
Point_060 = ["060", "09", 59, 59, 9, 0, 9, 1.285714286]
Point_061 = ["061", "09", 60, 59, 9, 2, 9, 1.285714286]
Point_062 = ["062", "09", 61, 59, 9, 2, 9, 1.285714286]
Point_063 = ["063", "09", 62, 59, 9, 2, 9, 1.285714286]
Point_064 = ["064", "10", 63, 66, 10, 1, 9, 1.285714286]
Point_065 = ["065", "10", 64, 66, 10, 0, 9, 1.285714286]
Point_066 = ["066", "10", 65, 66, 10, 2, 9, 1.285714286]
Point_067 = ["067", "10", 66, 66, 10, 2, 9, 1.285714286]
Point_068 = ["068", "10", 67, 66, 10, 0, 9, 1.285714286]
Point_069 = ["069", "10", 68, 66, 10, 3, 9, 1.285714286]
Point_070 = ["070", "10", 69, 66, 10, 1, 9, 1.285714286]
Point_071 = ["071", "11", 70, 73, 11, 1, 9, 1.285714286]
Point_072 = ["072", "11", 71, 73, 11, 0, 9, 1.285714286]
Point_073 = ["073", "11", 72, 73, 11, 0, 9, 1.285714286]
Point_074 = ["074", "11", 73, 73, 11, 1, 9, 1.285714286]
Point_075 = ["075", "11", 74, 73, 11, 1, 9, 1.285714286]
Point_076 = ["076", "11", 75, 73, 11, 3, 9, 1.285714286]
Point_077 = ["077", "11", 76, 73, 11, 3, 9, 1.285714286]
Point_078 = ["078", "12", 77, 80, 12, 0, 11, 1.571428571]
Point_079 = ["079", "12", 78, 80, 12, 1, 11, 1.571428571]
Point_080 = ["080", "12", 79, 80, 12, 3, 11, 1.571428571]
Point_081 = ["081", "12", 80, 80, 12, 2, 11, 1.571428571]
Point_082 = ["082", "12", 81, 80, 12, 3, 11, 1.571428571]
Point_083 = ["083", "12", 82, 80, 12, 0, 11, 1.571428571]
Point_084 = ["084", "12", 83, 80, 12, 2, 11, 1.571428571]
Point_085 = ["085", "13", 84, 87, 13, 1, 5, 0.714285714]
Point_086 = ["086", "13", 85, 87, 13, 0, 5, 0.714285714]
Point_087 = ["087", "13", 86, 87, 13, 2, 5, 0.714285714]
Point_088 = ["088", "13", 87, 87, 13, 0, 5, 0.714285714]
Point_089 = ["089", "13", 88, 87, 13, 2, 5, 0.714285714]
Point_090 = ["090", "13", 89, 87, 13, 0, 5, 0.714285714]
Point_091 = ["091", "13", 90, 87, 13, 0, 5, 0.714285714]
Point_092 = ["092", "14", 91, 94, 14, 2, 11, 1.571428571]
Point_093 = ["093", "14", 92, 94, 14, 1, 11, 1.571428571]
Point_094 = ["094", "14", 93, 94, 14, 1, 11, 1.571428571]
Point_095 = ["095", "14", 94, 94, 14, 1, 11, 1.571428571]
Point_096 = ["096", "14", 95, 94, 14, 3, 11, 1.571428571]
Point_097 = ["097", "14", 96, 94, 14, 0, 11, 1.571428571]
Point_098 = ["098", "14", 97, 94, 14, 3, 11, 1.571428571]
Point_099 = ["099", "15", 98, 98.5, 15, 2, 3, 1.5]
Point_100 = ["100", "15", 99, 98.5, 15, 1, 3, 1.5]

Data_Day = pd.DataFrame([Point_001, Point_002, Point_003, Point_004, Point_005, Point_006, Point_007, Point_008,
                         Point_009, Point_010, Point_011, Point_012, Point_013, Point_014, Point_015, Point_016,
                         Point_017, Point_018, Point_019, Point_020, Point_021, Point_022, Point_023, Point_024,
                         Point_025, Point_026, Point_027, Point_028, Point_029, Point_030, Point_031, Point_032,
                         Point_033, Point_034, Point_035, Point_036, Point_037, Point_038, Point_039, Point_040,
                         Point_041, Point_042, Point_043, Point_044, Point_045, Point_046, Point_047, Point_048,
                         Point_049, Point_050, Point_051, Point_052, Point_053, Point_054, Point_055, Point_056,
                         Point_057, Point_058, Point_059, Point_060, Point_061, Point_062, Point_063, Point_064,
                         Point_065, Point_066, Point_067, Point_068, Point_069, Point_070, Point_071, Point_072,
                         Point_073, Point_074, Point_075, Point_076, Point_077, Point_078, Point_079, Point_080,
                         Point_081, Point_082, Point_083, Point_084, Point_085, Point_086, Point_087, Point_088,
                         Point_089, Point_090, Point_091, Point_092, Point_093, Point_094, Point_095, Point_096,
                         Point_097, Point_098, Point_099, Point_100],
                        index=["Point_001", "Point_002", "Point_003", "Point_004", "Point_005", "Point_006",
                               "Point_007", "Point_008", "Point_009", "Point_010", "Point_011", "Point_012",
                               "Point_013", "Point_014", "Point_015", "Point_016", "Point_017", "Point_018",
                               "Point_019", "Point_020", "Point_021", "Point_022", "Point_023", "Point_024",
                               "Point_025", "Point_026", "Point_027", "Point_028", "Point_029", "Point_030",
                               "Point_031", "Point_032", "Point_033", "Point_034", "Point_035", "Point_036",
                               "Point_037", "Point_038", "Point_039", "Point_040", "Point_041", "Point_042",
                               "Point_043", "Point_044", "Point_045", "Point_046", "Point_047", "Point_048",
                               "Point_049", "Point_050", "Point_051", "Point_052", "Point_053", "Point_054",
                               "Point_055", "Point_056", "Point_057", "Point_058", "Point_059", "Point_060",
                               "Point_061", "Point_062", "Point_063", "Point_064", "Point_065", "Point_066",
                               "Point_067", "Point_068", "Point_069", "Point_070", "Point_071", "Point_072",
                               "Point_073", "Point_074", "Point_075", "Point_076", "Point_077", "Point_078",
                               "Point_079", "Point_080", "Point_081", "Point_082", "Point_083", "Point_084",
                               "Point_085", "Point_086", "Point_087", "Point_088", "Point_089", "Point_090",
                               "Point_091", "Point_092", "Point_093", "Point_094", "Point_095", "Point_096",
                               "Point_097", "Point_098", "Point_099", "Point_100"],
                        columns=["No.", "Category", "Day", "Day Mean", "Week", "Frequency", "Sum", "Mean"])
print(Data_Day)

print("")
print("###############################################################################################################")
print("### Kelman et al. Data 02#")
print("###############################################################################################################")

Week_01 = ["001", 3, 1, 4.285714286]
Week_02 = ["002", 10, 2, 2.142857143]
Week_03 = ["003", 17, 3, 1.142857143]
Week_04 = ["004", 24, 4, 1.714285714]
Week_05 = ["005", 31, 5, 1.142857143]
Week_06 = ["006", 38, 6, 1.142857143]
Week_07 = ["007", 45, 7, 1]
Week_08 = ["008", 52, 8, 1.142857143]
Week_09 = ["009", 59, 9, 1.285714286]
Week_10 = ["010", 66, 10, 1.285714286]
Week_11 = ["011", 73, 11, 1.285714286]
Week_12 = ["012", 80, 12, 1.571428571]
Week_13 = ["013", 87, 13, 0.714285714]
Week_14 = ["014", 94, 14, 1.571428571]
Week_15 = ["015", 98.5, 15, 1.5]

Data_Week = pd.DataFrame([Week_01, Week_02, Week_03, Week_04, Week_05, Week_06, Week_07, Week_08, Week_09, Week_10,
                          Week_11, Week_12, Week_13, Week_14, Week_15],
                        index=["Week_01", "Week_02", "Week_03", "Week_04", "Week_05", "Week_06", "Week_07", "Week_08",
                               "Week_09", "Week_10", "Week_11", "Week_12", "Week_13", "Week_14", "Week_15"],
                        columns=["No.", "Day_Mean", "Week", "Mean"])
print(Data_Week)

########################################################################################################################
# Regression Analysis
########################################################################################################################
############################################################################################################
#    Definition of Regression Function
# n1 * np.exp(-l1 * (x-b)) + (n2 * np.exp(-l2 * (x-b))) * (np.sin(a * (x - b)))
# n1 * exp(-l1 * (x-b)) + n2 * exp(-l2 * (x-b)) * sin(a * (x-b))
# n1 * exp(-l1 * (x-b)) + (n2 * exp(-l2 * (x-b))) * (sin(a * (x - b)))
############################################################################################################

def reg_func(parameter: ["n1", "l1", "n2", "l2", "a", "b", "c", "d"], x, y):
    n1 = parameter[0]
    l1 = parameter[1]
    n2 = parameter[2]
    l2 = parameter[3]
    a = parameter[4]
    b = parameter[5]
    c = parameter[6]
    d = parameter[7]

    _Output_ = y - ( n1 * np.exp(-l1 * (x-b)) + (n2 * np.exp(-l2 * (x-b))) * (np.sin(a * (x - b))) + c*x**2 + d )
    return _Output_

############################################################################################################
#    Non-linear Regression Analysis (Weeks)
############################################################################################################
#parameter_0 = [2.0,  0.075,  3.125,  0.1,  0.35,   1.25,  0.000045,  1.0]

x = Data_Week["Day_Mean"]
y = Data_Week["Mean"]
parameter_0 = [2.0,  0.075,  3.125,  0.1,  0.35,   1.25,  0.000045,  1.0]
RegResult_Original_W = optimize.leastsq(reg_func, parameter_0, args=(x, y), full_output=True)
Result_W_0 = RegResult_Original_W[0]
Result_W = [Result_W_0[0], Result_W_0[1], Result_W_0[2], Result_W_0[3], Result_W_0[4], Result_W_0[5], Result_W_0[6], Result_W_0[7]]

print("")
print("########## Evaluation of Result on the Regression Analysis ##########")
print(["Reg. Analysis Result:", RegResult_Original_W[-1], "1, 2, 3 or 4, the solution was found."])
#print(Result_W)

############################################################################################################
#    Non-linear Regression Analysis (Days)
############################################################################################################


x = Data_Day["Day"]
y = Data_Day["Frequency"]
parameter_0 = [2.0,  0.075,  3.125,  0.1,  0.35,   1.25,  0.000045,  1.0]
RegResult_Original_D = optimize.leastsq(reg_func, parameter_0, args=(x, y), full_output=True)
Result_D_0 = RegResult_Original_D[0]
Result_D = [Result_D_0[0], Result_D_0[1], Result_D_0[2], Result_D_0[3], Result_D_0[4], Result_D_0[5], Result_D_0[6], Result_D_0[7]]
print(["Reg. Analysis Result:", RegResult_Original_W[-1], "1, 2, 3 or 4, the solution was found."])
#print(Result_D)

############################################################################################################
# Coefficient Matrix (Data Set)
############################################################################################################
CoefficientDataSet = pd.DataFrame([Result_W, Result_D],
                                  index=["Reg. Result (Week)", "Reg. Result (Day)"],
                                  columns=["n1", "l1", "n2", "l2", "a", "b", "c", "d"]
                                  )
print("")
print("########## Result of Non-linear Regression Analysis (CoefficientDataSet) ##########")
print(CoefficientDataSet)

CoefficientDataSet_round = [Decimal(str(CoefficientDataSet.iat[1, 0])).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP),
                            Decimal(str(CoefficientDataSet.iat[1, 1])).quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP),
                            Decimal(str(CoefficientDataSet.iat[1, 2])).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP),
                            Decimal(str(CoefficientDataSet.iat[1, 3])).quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP),
                            Decimal(str(CoefficientDataSet.iat[1, 4])).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP),
                            Decimal(str(CoefficientDataSet.iat[1, 5])).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
                            Decimal(str(CoefficientDataSet.iat[1, 6])).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
                            Decimal(str(CoefficientDataSet.iat[1, 7])).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
                            ]

print("")
print("########## Rounded Result of Non-linear Regression Analysis (CoefficientDataSet_round) ##########")
print(CoefficientDataSet_round)

############################################################################################################
########################################################################################################################
# Values for Curves (Week)
########################################################################################################################
n1 = CoefficientDataSet.at["Reg. Result (Week)", "n1"]
l1 = CoefficientDataSet.at["Reg. Result (Week)", "l1"]
n2 = CoefficientDataSet.at["Reg. Result (Week)", "n2"]
l2 = CoefficientDataSet.at["Reg. Result (Week)", "l2"]
a = CoefficientDataSet.at["Reg. Result (Week)", "a"]
b = CoefficientDataSet.at["Reg. Result (Week)", "b"]
c = CoefficientDataSet.at["Reg. Result (Week)", "c"]
d = CoefficientDataSet.at["Reg. Result (Week)", "d"]


X_Week = np.linspace(0.0, 100, 1000)
Y_Week = n1 * np.exp(-l1 * (X_Week-b)) + (n2 * np.exp(-l2 * (X_Week-b))) * (np.sin(a * (X_Week-b))) + c*X_Week**2 + d
Y_Week2 = n1 * np.exp(-l1 * (X_Week-b)) + c*X_Week**2 + d
Y_Week3 = c*X_Week**2 + d
Y_Week4 = n1 * np.exp(-l1 * (X_Week-b)) - (n2 * np.exp(-l2 * (X_Week-b))) + c*X_Week**2 + d
Y_DF = Data_Week["Mean"] - (n1 * np.exp(-l1 * (Data_Week["Day_Mean"]-b)) + c*Data_Week["Day_Mean"]**2 + d)

Wave_U = (n2 * np.exp(-l2 * (X_Week-b)))
Wave_C = (n2 * np.exp(-l2 * (X_Week-b))) *(np.sin(a * (X_Week-b)))
Wave_L = -(n2 * np.exp(-l2 * (X_Week-b)))

Peaks_1 = [(b + (np.pi/2)/a),
           (b + (np.pi/2)/a) + 2*np.pi/a,
           (b + (np.pi/2)/a) + 2*np.pi/a + 2*np.pi/a,
           (b + (np.pi/2)/a) + 2*np.pi/a + 2*np.pi/a + 2*np.pi/a]

Peaks_1_round = [Decimal(str(Peaks_1[0])).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
                 Decimal(str(Peaks_1[1])).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
                 Decimal(str(Peaks_1[2])).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
                 Decimal(str(Peaks_1[3])).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)]

Values = [n1 * np.exp(-l1 * (30-b)) + (n2 * np.exp(-l2 * (30-b))) * (np.sin(a * (30-b))),
          n1 * np.exp(-l1 * (60-b)) + (n2 * np.exp(-l2 * (60-b))) * (np.sin(a * (60-b))),
          n1 * np.exp(-l1 * (90-b)) + (n2 * np.exp(-l2 * (90-b))) * (np.sin(a * (90-b))),
          n1 * np.exp(-l1 * (120-b)) + (n2 * np.exp(-l2 * (120-b))) * (np.sin(a * (120-b))),
          n1 * np.exp(-l1 * (150-b)) + (n2 * np.exp(-l2 * (150-b))) * (np.sin(a * (150-b))),
          n1 * np.exp(-l1 * (180-b)) + (n2 * np.exp(-l2 * (180-b))) * (np.sin(a * (180-b)))]

Values_round = [Decimal(str(Values[0])).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
                Decimal(str(Values[1])).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
                Decimal(str(Values[2])).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
                Decimal(str(Values[3])).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
                Decimal(str(Values[4])).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
                Decimal(str(Values[5])).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)]

Period = 2*np.pi/a
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

#n1 = CoefficientDataSet.at["Reg. Result (Day)", "n1"]
#l1 = CoefficientDataSet.at["Reg. Result (Day)", "l1"]
#n2 = CoefficientDataSet.at["Reg. Result (Day)", "n2"]
#l2 = CoefficientDataSet.at["Reg. Result (Day)", "l2"]
#a = CoefficientDataSet.at["Reg. Result (Day)", "a"]
#b = CoefficientDataSet.at["Reg. Result (Day)", "b"]
#c = CoefficientDataSet.at["Reg. Result (Day)", "c"]
#d = CoefficientDataSet.at["Reg. Result (Day)", "d"]

#X_Day = np.linspace(0.0, 100, 1000)
#Y_Day = n1 * np.exp(-l1 * (X_Day-b)) + (n2 * np.exp(-l2 * (X_Day-b))) * (np.sin(a * (X_Day-b))) + c*X_Day**2 + d
########################################################################################################################
# Figures
########################################################################################################################
Figure_object = plt.figure(1, figsize=(11.69, 8.27), dpi=400, edgecolor="black", linewidth=0.5)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)

plt.figtext(0.0190, 0.960, "a", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")
plt.figtext(0.5130, 0.960, "b", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")
plt.figtext(0.0190, 0.500, "c", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")
plt.figtext(0.5130, 0.500, "d", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")

gs_master = matplotlib.gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[1, 1])
gs_1 = matplotlib.gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=2, subplot_spec=gs_master[0])
gs_2 = matplotlib.gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=2, subplot_spec=gs_master[1])

#ticks_x = [0, 1, 2, 3, 4,  5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
#ticks_y = [-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]

Axes_obj_01 = Figure_object.add_subplot(gs_1[0])
Axes_obj_02 = Figure_object.add_subplot(gs_1[1])
Axes_obj_03 = Figure_object.add_subplot(gs_2[0])
Axes_obj_04 = Figure_object.add_subplot(gs_2[1])

############################################################################################################
# "(a) Kelman et al. 2003":
############################################################################################################
#im = Image.open("./achan.jpg") im.show()
Axes_obj_01.set_title("Kelman et al. 2003 Figure 1", size=11.4230, fontweight="normal")
Axes_obj_01.set_xlabel('Time (Days)')
Axes_obj_01.set_ylabel('Frequency (Cases)')
Axes_obj_01.tick_params(length=0.0)
Axes_obj_01.set_xticklabels([])
Axes_obj_01.set_yticklabels([])
#Axes_obj_01.yaxis.set_visible(False)
#Axes_obj_01.xaxis.set_visible(False)

#Axes_obj_01.spines['right'].set_visible(False)
#Axes_obj_01.spines['left'].set_visible(False)
#Axes_obj_01.spines['top'].set_visible(False)
#Axes_obj_01.spines['bottom'].set_visible(False)

img = plt.imread("31815kelc.f1.jpg")
Axes_obj_01.imshow(img)

############################################################################################################
# "(b) Appropriate 2D Bar Chart":
############################################################################################################
Axes_obj_02.set_title("Weekly Bar Chart & Curve Fitting", size=11.4230, fontweight="normal")
Axes_obj_02.set_xlabel('Time (Days)')
Axes_obj_02.set_ylabel('Frequency (Cases)')
Axes_obj_02.set_xticks([0, 10, 20, 30, 40,  50, 60, 70, 80, 90, 100])
Axes_obj_02.set_xlim(-7.5, 107.5)
Axes_obj_02.set_ylim(0.0, 10)

Axes_obj_02.bar(Data_Day["Day"]+0.5, Data_Day["Frequency"], facecolor='blue', width=0.7, alpha=0.3)
Axes_obj_02.bar(Data_Week["Day_Mean"][0:14], Data_Week["Mean"][0:14], facecolor='none', width=7.0, alpha=1.0,
                edgecolor='black', linewidth=1.0, linestyle='solid', hatch="")
Axes_obj_02.plot(X_Week, Y_Week2, color='red', linestyle='solid', linewidth=1.6318, zorder=8)
Axes_obj_02.scatter(Data_Week["Day_Mean"], Data_Week["Mean"], facecolor='red', edgecolor='black', linewidth=1.0, zorder=9)
#Axes_obj_02.plot(X_Week, Y_Week3, color='blue', linestyle='dashed', linewidth=1.0, zorder=10)

textstr='\n'.join(["Setting: Western Australia.",
                   "Participants: 5408 patients admitted to hospital with",
                   "venous thromboembolism and matched with data for",
                   "arrivals of international flights during 1981-99."])

# Axes_obj_02.text(10, 9, textstr, fontsize=10, verticalalignment='top',
#                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

Axes_obj_02.annotate(
 text='\n'.join(["Week 3", "Valley"]),
    xy=(16.5, 1.5), xytext=(16.5, 6.0), ha='center', va='center', size=10.0,
    bbox=dict(boxstyle='round', edgecolor='black', fc='white'),
    arrowprops=dict(
        facecolor='red', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0, edgecolor='red', shrink=0.1)
)

Axes_obj_02.annotate(
 text='\n'.join(["Week 1", "Peak"]),
    xy=(3.5, 8.0), xytext=(16.5, 8.0), ha='center', va='center', size=10.0,
    bbox=dict(boxstyle='round', edgecolor='black', fc='white'),
    arrowprops=dict(
        facecolor='red', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0, edgecolor='red', shrink=0.1)
)

Axes_obj_02.annotate(
    text='\n'.join(["Not Monotonic", "Decrease function?"]),
    xy=(40-2.5, 7.0), xytext=(55-2.5, 7.0), ha='center', va='center', size=10.0,
    bbox=dict(boxstyle='round', edgecolor='red', fc='white')
)
Axes_obj_02.hlines(y=6.0, xmin=25, xmax=30, color='red', linestyle='solid', linewidth=1.0)
Axes_obj_02.hlines(y=8.0, xmin=25, xmax=30, color='red', linestyle='solid', linewidth=1.0)
Axes_obj_02.hlines(y=7.0, xmin=30, xmax=35, color='red', linestyle='solid', linewidth=1.0)
Axes_obj_02.vlines(x=30, ymin=6.0, ymax=8.0, color='red', linestyle='solid', linewidth=1.0)


Axes_obj_02.annotate(
 text='\n'.join(["Valley?", "Few Days Before", "90 days"]),
    xy=(87, 2.5), xytext=(87, 4.5), ha='center', va='center', size=10.0,
    bbox=dict(boxstyle='round', edgecolor='black', fc='white'),
    arrowprops=dict(
        facecolor='red', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0, edgecolor='red', shrink=0.1)
)

############################################################################################################
# "(c) Decreasing Curve":
############################################################################################################
Axes_obj_03.set_title("Residual (Decreasing Wave)", size=11.4230, fontweight="normal")
Axes_obj_03.set_xlabel('Time (Days)')
Axes_obj_03.set_ylabel('Residual')
Axes_obj_03.set_xticks([0, 10, 20, 30, 40,  50, 60, 70, 80, 90, 100])
Axes_obj_03.set_xlim(-7.5, 107.5)
Axes_obj_03.set_ylim(-3.0, 3.0)

Axes_obj_03.hlines(y=0.0, xmin=0.0, xmax=101, color="gray", linestyle='solid', linewidth=0.5)
Axes_obj_03.vlines(x=0.0, ymin=-3.0, ymax=3.0, color="black", linestyle='dashed', linewidth=1.0)
Axes_obj_03.vlines(x=90.0, ymin=-3.0, ymax=3.0, color="black", linestyle='dashed', linewidth=1.0)

Axes_obj_03.scatter(Data_Week["Day_Mean"], Y_DF, color='red')

Axes_obj_03.plot(X_Week, Wave_U, color='red', linestyle='dotted', linewidth=1.0)
Axes_obj_03.plot(X_Week, Wave_C, color='red', linestyle='solid', linewidth=1.6318)
Axes_obj_03.plot(X_Week, Wave_L, color='red', linestyle='dotted', linewidth=1.0)

Axes_obj_03.vlines(x=Peaks_1[0], ymin=2.25, ymax=2.5, color="black", linestyle='solid', linewidth=1.5)
Axes_obj_03.vlines(x=Peaks_1[1], ymin=2.25, ymax=2.5, color="black", linestyle='solid', linewidth=1.5)
Axes_obj_03.vlines(x=Peaks_1[2], ymin=2.25, ymax=2.5, color="black", linestyle='solid', linewidth=1.5)
Axes_obj_03.vlines(x=Peaks_1[3], ymin=2.25, ymax=2.5, color="black", linestyle='solid', linewidth=1.5)

Axes_obj_03.text(Peaks_1[0]+1.0, 2.75, Peaks_1_round[0],
                 size=10, color="red", ha='center', va='center', fontweight="normal",
                 bbox=dict(boxstyle='round', edgecolor='red', fc='red', alpha=0.0))
Axes_obj_03.text(Peaks_1[1], 2.75, Peaks_1_round[1],
                 size=10, color="red", ha='center', va='center', fontweight="normal",
                 bbox=dict(boxstyle='round', edgecolor='red', fc='red', alpha=0.0))
Axes_obj_03.text(Peaks_1[2], 2.75, Peaks_1_round[2],
                 size=10, color="red", ha='center', va='center', fontweight="normal",
                 bbox=dict(boxstyle='round', edgecolor='red', fc='red', alpha=0.0))
Axes_obj_03.text(Peaks_1[3]-1.0, 2.75, Peaks_1_round[3],
                 size=10, color="red", ha='center', va='center', fontweight="normal",
                 bbox=dict(boxstyle='round', edgecolor='red', fc='red', alpha=0.0))

Axes_obj_03.vlines(x=Peaks_1[0]+(np.pi/2/a), ymin=-2.5, ymax=0.0, color="black", linestyle='solid', linewidth=0.5)
Axes_obj_03.vlines(x=Peaks_1[1]+(np.pi/2/a), ymin=-2.5, ymax=0.0, color="black", linestyle='solid', linewidth=0.5)

Axes_obj_03.quiver(Peaks_1[0]-(np.pi/2/a)+2*np.pi/a, -2.5, -(np.pi/a), 0.0,
                   scale_units='xy', angles='xy', scale=1, linestyle='solid', width=0.008)
Axes_obj_03.quiver(Peaks_1[0]-(np.pi/2/a)+2*np.pi/a, -2.5, (np.pi/a), 0.0,
                   scale_units='xy', angles='xy', scale=1, linestyle='solid', width=0.008)

Axes_obj_03.text(Peaks_1[0]-(np.pi/2/a)+2*np.pi/a, -2.0,
                 Period_round, size=11.4230, color="red", ha='center', va='center', fontweight="normal",
                 bbox=dict(boxstyle='round', edgecolor='red', fc='red', alpha=0.0))

Axes_obj_03.text(85, 2.25, '\n'.join([''.join([str(Period_round), ' days']), "cycle"]),
                 size=11.4230, color="black", ha='center', va='center',
                 bbox=dict(boxstyle='round', edgecolor='red', fc='white'))

Axes_obj_03.annotate(
 text='\n'.join(["Not Negligible", "Outliers"]),
    xy=(87, -0.75), xytext=(87, -2.0), ha='center', va='center', size=10.0,
    bbox=dict(boxstyle='round', edgecolor='black', fc='white'),
    arrowprops=dict(
        facecolor='red', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0, edgecolor='red', shrink=0.1)
)
############################################################################################################


############################################################################################################
# "(d) Damped Wave Curve":
############################################################################################################
Axes_obj_04.set_title("Curve Fitting (& Hypothesis based on OC)", size=11.4230, fontweight="normal")
Axes_obj_04.set_xlabel('Time (Days)')
Axes_obj_04.set_ylabel('Frequency (Cases)')
Axes_obj_04.set_xticks([0, 10, 20, 30, 40,  50, 60, 70, 80, 90, 100])
Axes_obj_04.set_xlim(-7.5, 107.5)
Axes_obj_04.set_ylim(0.0, 10)

Axes_obj_04.bar(Data_Day["Day"]+0.5, Data_Day["Frequency"], facecolor='blue', width=0.7, alpha=0.3)
Axes_obj_04.bar(Data_Week["Day_Mean"][0:14], Data_Week["Mean"][0:14], facecolor='none', width=7.0, alpha=1.0,
                edgecolor='black', linewidth=1.0, linestyle='solid', hatch="")
Axes_obj_04.plot(X_Week, Y_Week, color='red', linestyle='solid', linewidth=1.6318, zorder=9)
Axes_obj_04.scatter(Data_Week["Day_Mean"], Data_Week["Mean"], facecolor='red', edgecolor='black', linewidth=1.0, zorder=10)

Axes_obj_04.annotate(
 text='\n'.join(["Drug Withdrawal of", "Oral Contraceptives"]),
    xy=(87, 2.5), xytext=(87, 4.5), ha='center', va='center', size=10.0,
    bbox=dict(boxstyle='round', edgecolor='black', fc='white'),
    arrowprops=dict(
        facecolor='red', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0, edgecolor='red', shrink=0.1)
)

Axes_obj_04.annotate(
    text='\n'.join(["S"]),
    xy=(-3, 9.25), xytext=(-3, 9.25), ha='center', va='center', size=10.0,
    bbox=dict(boxstyle='round', edgecolor='red', fc='white')
)

Axes_obj_04.annotate(
    text='\n'.join(["W"]),
    xy=(87, 9.25), xytext=(87, 9.25), ha='center', va='center', size=10.0,
    bbox=dict(boxstyle='round', edgecolor='red', fc='white')
)

Axes_obj_04.annotate(
    text='\n'.join(["S"]),
    xy=(94, 9.25), xytext=(94, 9.25), ha='center', va='center', size=10.0,
    bbox=dict(boxstyle='round', edgecolor='red', fc='white')
)

Axes_obj_04.text(-4, 5, "Travel (Honeymoon?) & Start OC", color="red", size=10,
                 horizontalalignment="center", verticalalignment="center", rotation="vertical")

Axes_obj_04.quiver(-3.0, 9.25, 90-3.0, 0, scale_units='xy', angles='xy', scale=1, color="black", width=0.008)
Axes_obj_04.quiver(94, 9.25, 10, 0, scale_units='xy', angles='xy', scale=1, color="black", width=0.008)

Axes_obj_04.annotate(
    text='\n'.join(["Continuous OC Taking until before", " Scheduled Withdrawal for Bleed"]),
    xy=(40, 9.25), xytext=(40, 9.25), ha='center', va='center', size=10.0,
    bbox=dict(boxstyle='round', edgecolor='black', fc='white')
)

Axes_obj_04.vlines(x=0.0, ymin=0.0, ymax=10.0, color="black", linestyle='dashed', linewidth=1.0)
Axes_obj_04.vlines(x=90.0, ymin=0.0, ymax=10.0, color="black", linestyle='dashed', linewidth=1.0)


Axes_obj_04.annotate(
 text='\n'.join(["Polyphasic Oral", "Contraceptives", "Related Wave"]),
    xy=(20, 4.0), xytext=(40, 6.0), ha='center', va='center', size=10.0,
    bbox=dict(boxstyle='round', edgecolor='red', fc='white'),
    arrowprops=dict(
        facecolor='red', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0, edgecolor='red', shrink=0.1)
)
############################################################################################################
Figure_object.tight_layout()
my_path = os.path.abspath("Figures")
plt.savefig(os.path.join(my_path, "00_Kimoto_et_al_(2023)_[_Fig_05_].png"))

from PIL import Image # py -m pip install pillow
img = Image.open(os.path.join(my_path, "00_Kimoto_et_al_(2023)_[_Fig_05_].png"))
img_resize = img.resize(size=(2866, 2016)) #size in pixels, as a 2-tuple: (width, height)
my_path2 = os.path.abspath("Figures/Fixed_Version/B6")
img_resize.save(os.path.join(my_path2, "00_Kimoto_et_al_(2023)_[_Fig_05_]_B6.png"))

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


