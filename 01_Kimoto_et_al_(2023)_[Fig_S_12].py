########################################################################################################################
# Fig. S12. Bimodal distributions & tiled parabola in COVID-19 patients.
# (01_Kimoto_et_al_(2023)_[Fig_S_12].py)
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

########################################################################################################################
# Making Data Sets
########################################################################################################################
############################################################################################################
# () ########## Patient DataSet (Guo et al. 2020) (PatientDataSet) ##########
############################################################################################################
import os
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option("display.width", 300)
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1.inset_locator as mpl_inset
import matplotlib.patches as pat
from PIL import Image # py -m pip install pillow
import matplotlib.image as mpimg
# import cv2
#pip install opencv-contrib-python
#path_PyROOT = os.path.abspath("/home/root_src/bindings/pyroot_legacy/ROOT.py")
########################################################################################################################
# For Manuscripts to Scientific Jounals
########################################################################################################################
print(matplotlib.get_cachedir())
#/home/kimoto/.cache/matplotlib: fontlist-v310.json  fontlist-v330.json  tex.cache#
print(matplotlib.matplotlib_fname())
print(matplotlib.rcParams["font.family"])
print(matplotlib.rcParams["font.sans-serif"])
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rcParams['font.sans-serif'] = ["Arial"]
########################################################################################################################
#conda-forge / packages / root 6.24.2
#https://anaconda.org/conda-forge/root/
#To install this package with conda run one of the following:
#conda install -c conda-forge root
#conda install -c conda-forge/label/gcc7 root
#conda install -c conda-forge/label/broken root
#conda install -c conda-forge/label/cf202003 root
########################################################################################################################

########################################################################################################################
# Making Data Sets
########################################################################################################################
############################################################################################################
# () ########## Patient DataSet (Guo et al. 2020) (PatientDataSet) ##########
############################################################################################################
Point_01 = [0, 1, 0.003, 0.000, 0.003, -0.419, -0.109, 0.023, 0.133, -0.026, 0.217, 1.053, 0.003, 0.000, "green"]
Point_02 = [1, 1, 0.012, 0.109, -0.010, -0.310, -0.088, 0.129, 0.155, 0.080, 0.204, 1.056, -0.010, 0.003, "green"]
Point_03 = [2, 1, 0.012, 0.589, -0.106, 0.160, 0.006, 0.590, 0.248, 0.541, 0.111, 1.075, -0.103, 0.022, "green"]
Point_04 = [3, 1, 0.018, 0.145, -0.011, -0.273, -0.080, 0.165, 0.162, 0.116, 0.203, 1.056, -0.011, 0.003, "green"]
Point_05 = [4, 1, 0.027, 0.211, -0.016, -0.207, -0.067, 0.230, 0.175, 0.181, 0.199, 1.057, -0.015, 0.004, "green"]
Point_06 = [5, 1, 0.042, 0.451, -0.049, 0.031, -0.020, 0.463, 0.223, 0.414, 0.166, 1.063, -0.048, 0.010, "green"]
Point_07 = [6, 1, 0.042, 0.727, -0.104, 0.301, 0.034, 0.729, 0.277, 0.679, 0.112, 1.074, -0.102, 0.021, "green"]
Point_08 = [7, 1, 0.075, 0.276, 0.018, -0.134, -0.053, 0.302, 0.190, 0.253, 0.232, 1.050, 0.018, -0.003, "green"]
Point_09 = [8, 1, 0.075, 0.872, -0.101, 0.450, 0.064, 0.875, 0.306, 0.825, 0.116, 1.074, -0.098, 0.021, "green"]
Point_10 = [9, 1, 0.084, 0.182, 0.046, -0.225, -0.071, 0.213, 0.172, 0.164, 0.259, 1.045, 0.045, -0.009, "green"]
Point_11 = [10, 1, 0.096, 0.058, 0.082, -0.343, -0.094, 0.097, 0.148, 0.047, 0.295, 1.037, 0.081, -0.016, "green"]
Point_12 = [11, 1, 0.105, 1.076, -0.112, 0.656, 0.105, 1.076, 0.347, 1.027, 0.105, 1.076, -0.109, 0.023, "green"]
Point_13 = [12, 1, 0.117, 0.523, 0.010, 0.117, -0.003, 0.548, 0.240, 0.498, 0.224, 1.052, 0.010, -0.001, "green"]
Point_14 = [13, 1, 0.117, 0.654, -0.016, 0.245, 0.023, 0.673, 0.265, 0.624, 0.198, 1.057, -0.016, 0.004, "green"]
Point_15 = [14, 1, 0.161, 0.138, 0.131, -0.252, -0.076, 0.186, 0.166, 0.137, 0.342, 1.028, 0.128, -0.025, "green"]
Point_16 = [15, 1, 0.212, 0.552, 0.098, 0.164, 0.007, 0.594, 0.249, 0.545, 0.310, 1.034, 0.096, -0.019, "green"]
Point_17 = [16, 1, 0.224, 0.422, 0.136, 0.039, -0.018, 0.471, 0.224, 0.422, 0.347, 1.027, 0.133, -0.026, "green"]
Point_18 = [17, 2, 0.096, 1.737, -0.026, -0.897, -0.011, 1.745, 0.351, 1.720, 0.244, 3.853, 0.096, 1.737, "orange"]
Point_19 = [18, 2, 0.129, 2.813, -0.069, 0.179, 0.064, 2.818, 0.426, 2.792, 0.202, 3.856, 0.053, 1.740, "orange"]
Point_20 = [19, 2, 0.138, 3.860, -0.133, 1.223, 0.138, 3.860, 0.499, 3.835, 0.138, 3.860, -0.011, 1.745, "orange"]
Point_21 = [20, 2, 0.170, 2.399, 0.002, -0.232, 0.036, 2.409, 0.397, 2.383, 0.273, 3.851, 0.124, 1.735, "orange"]
Point_22 = [21, 2, 0.203, 2.922, -0.002, 0.293, 0.072, 2.932, 0.434, 2.906, 0.269, 3.851, 0.120, 1.736, "orange"]
Point_23 = [22, 2, 0.374, 2.050, 0.229, -0.566, 0.012, 2.076, 0.374, 2.050, 0.499, 3.835, 0.351, 1.720, "orange"]
Point_24 = [23, 3, 0.724, 1.737, -0.379, -0.169, 0.611, 1.814, 1.872, 0.957, 1.244, 2.502, 0.477, 1.374, "red"]
Point_25 = [24, 3, 0.736, 1.323, -0.136, -0.505, 0.422, 1.536, 1.683, 0.679, 1.445, 2.366, 0.677, 1.237, "red"]
Point_26 = [25, 3, 0.778, 1.883, -0.416, -0.018, 0.696, 1.939, 1.957, 1.081, 1.213, 2.523, 0.446, 1.395, "red"]
Point_27 = [26, 3, 0.948, 1.498, -0.058, -0.241, 0.570, 1.754, 1.831, 0.897, 1.509, 2.322, 0.741, 1.194, "red"]
Point_28 = [27, 3, 1.002, 1.832, -0.202, 0.066, 0.743, 2.008, 2.004, 1.151, 1.390, 2.403, 0.623, 1.275, "red"]
Point_29 = [28, 3, 1.002, 1.948, -0.267, 0.162, 0.797, 2.088, 2.058, 1.230, 1.336, 2.440, 0.569, 1.311, "red"]
Point_30 = [29, 3, 1.011, 1.396, 0.051, -0.290, 0.543, 1.714, 1.804, 0.857, 1.599, 2.261, 0.832, 1.132, "red"]
Point_31 = [30, 3, 1.014, 2.406, -0.515, 0.547, 1.014, 2.406, 2.275, 1.549, 1.131, 2.579, 0.364, 1.451, "red"]
Point_32 = [31, 3, 1.065, 0.974, 0.333, -0.608, 0.364, 1.451, 1.625, 0.593, 1.832, 2.102, 1.065, 0.974, "red"]
Point_33 = [32, 3, 1.235, 2.508, -0.389, 0.756, 1.131, 2.579, 2.392, 1.721, 1.235, 2.508, 0.468, 1.380, "red"]
Point_34 = [33, 4, 1.714, 0.800, 0.967, -0.388, 0.488, 1.633, 1.749, 0.776, 2.357, 1.745, 1.589, 0.617, "red"]
Point_35 = [34, 4, 2.353, 1.665, 1.010, 0.688, 1.093, 2.522, 2.353, 1.665, 2.392, 1.721, 1.625, 0.593, "red"]

PatientDataSet = pd.DataFrame(
    [Point_01, Point_02, Point_03, Point_04, Point_05, Point_06, Point_07, Point_08, Point_09, Point_10,
     Point_11, Point_12, Point_13, Point_14, Point_15, Point_16, Point_17, Point_18, Point_19, Point_20,
     Point_21, Point_22, Point_23, Point_24, Point_25, Point_26, Point_27, Point_28, Point_29, Point_30,
     Point_31, Point_32, Point_33, Point_34, Point_35],
   index=["Point_01", "Point_02", "Point_03", "Point_04", "Point_05",
          "Point_06", "Point_07", "Point_08", "Point_09", "Point_10",
          "Point_11", "Point_12", "Point_13", "Point_14", "Point_15",
          "Point_16", "Point_17", "Point_18", "Point_19", "Point_20",
          "Point_21", "Point_22", "Point_23", "Point_24", "Point_25",
          "Point_26", "Point_27", "Point_28", "Point_29", "Point_30",
          "Point_31", "Point_32", "Point_33", "Point_34", "Point_35"],
   columns=["No", "Frag1", "Plasma TnT", "Plasma NT-proBNP", "Axis1", "Axis2",
            "FlameLeftX", "FlameLeftY",
            "FlameRightX", "FlameRightY",
            "FlameUpperX", "FlameUpperY",
            "FlameLowewX", "FlameLowewY", "LabelC"]
)


PatientDataSet["Plasma TnT / Plasma NT-proBNP"] = PatientDataSet["Plasma TnT"] / PatientDataSet["Plasma NT-proBNP"]
PatientDataSet["Plasma NT-proBNP / Plasma TnT"] = PatientDataSet["Plasma NT-proBNP"] / PatientDataSet["Plasma TnT"]

print("")
print("########## Patient DataSet (Guo et al. 2020) (PatientDataSet) ##########")
print(PatientDataSet)

############################################################################################################
# () ########## Center of Gravity (CenterOfGravityDataSet) ##########
############################################################################################################
CenterOfGravity_G1 = ["y = 4.91827615894497x + 0", 0.08355686, 0.410955711, 4.918276159, 0, -0.203323272, 0.427944766]
CenterOfGravity_G2 = ["y = 14.2256220432896 + 0 ", 0.184908773, 2.630442321, 14.22562204, 0, -0.070295696, 2.643440612]
CenterOfGravity_G3 = ["y = 1.47059567791897 + 0 ", 1.131629299, 1.664169157, 1.470595678, 0, -0.679996559, 2.433673186]
CenterOfGravity_Guo = ["y = 1.29296159681312+ 0.123729020851572 ", np.NaN, np.NaN, 1.29296159681312, 0.123729020851572,
                        np.NaN, np.NaN]

CenterOfGravityDataSet = pd.DataFrame(
    [CenterOfGravity_G1, CenterOfGravity_G2, CenterOfGravity_G3, CenterOfGravity_Guo],
   index=["G1", "G2", "G3", "Guo et al."],
   columns=["Equation", "MeanX", "MeanY", "a1", "b1", "a2", "b2"]
)

print("")
print("########## Center of Gravity (CenterOfGravityDataSet) ##########")
print(CenterOfGravityDataSet)
############################################################################################################
# () ########## Axis DataSet (AxisDataSet) ##########
############################################################################################################
G1_AxisP1 = [0.0001, 0.0006, 0.214230409, 1.053644314, 0.2141, 1.0531]
G1_AxisP2 = [-0.0260, 0.4332, 0.216607852, 0.383903348, 0.2426, -0.0493]
G2_AxisP1 = [0.1220, 1.7356, 0.270700535, 3.850883492, 0.1487, 2.1153]
G2_AxisP2 = [0.0518, 2.6398, 0.413473399, 2.614375212, 0.3617, -0.0254]
G3_AxisP1 = [0.7896, 1.1612, 1.556732295, 2.289323785, 0.7672, 1.1282]
G3_AxisP2 = [0.7059, 1.9536, 1.966854363, 1.096218988, 1.2609, -0.8574]

AxisDataSet = pd.DataFrame(
    [G1_AxisP1, G1_AxisP2, G2_AxisP1, G2_AxisP2, G3_AxisP1, G3_AxisP2],
   index=["G1_P1", "G1_P2", "G2_P1", "G2_P2", "G3_P1", "G3_P2"],
   columns=["Sx", "Sy", "Ex", "Ey", "deltaX", "deltaY"]
)

print("")
print("########## Axis DataSet (AxisDataSet) ##########")
print(AxisDataSet)
############################################################################################################
# () ########## Frame DataSet (FrameDataSet) ##########
############################################################################################################
G1_S = [-0.109445807, 0.022861331, 0.133169794, -0.026468067]
G1_E = [0.104665801, 1.075921349, 0.347281402, 1.026591951]
G2_S = [-0.011134512, 1.74497963, 0.350571268, 1.719553271]
G2_E = [0.13755938, 3.860242742, 0.49926516, 3.834816382]
G3_S = [0.363899048, 1.450617736, 1.62480656, 0.593204967]
G3_E = [1.131049847, 2.578786385, 2.391957359, 1.721373616]

FrameDataSet = pd.DataFrame(
    [G1_S, G1_E, G2_S, G2_E, G3_S, G3_E],
   index=["G1_S", "G1_E", "G2_S", "G2_E", "G3_S", "G3_E"],
   columns=["Lx", "Ly", "Rx", "Ry"]
)

print("")
print("########## Frame DataSet (FrameDataSet) ##########")
print(FrameDataSet)

############################################################################################################
# () ########## Parabola (Rotated) (ParabolaRDataSet) ##########
############################################################################################################
G1_ParabolaR = [0.9937, -9.7746, -0.4863, 24.0372, -2.6270, 0.2690, 0.1000, 1.2500, 0.4900, 1.1500]
G2_ParabolaR = [0.1332, -3.7908, -0.5027, 26.9637, -7.1090, 2.2947, 0.1300, 4.1900, 0.7100, 4.1300]
G3_ParabolaR = [0.4849, -1.4263, -0.2786, 1.0487, -1.3686, 1.6372, 1.3300, 3.5100, 3.1500, 2.0400]
Sugawa_Parabola = [1.33, -2.67, -209.01, 1.33, -69.3, 8192, 43.8, 217.81, 98.01, 33.36]

ParabolaRDataSet = pd.DataFrame(
    [G1_ParabolaR, G2_ParabolaR, G3_ParabolaR, Sugawa_Parabola],
   index=["G1", "G2", "G3", "Sugawa"],
   columns=["y^2", "x*y", "y", "x^2", "x", "c", "Sx", "Sy", "Ex", "Ey"]
)

print("")
print("########## Parabola (Rotated) (ParabolaRDataSet) ##########")
print(ParabolaRDataSet)

G3_ParabolaR = [0.4849, -1.4263, -0.2786, 1.0487, -1.3686, 1.6372, 1.3300, 3.5100, 3.1500, 2.0400]
G3_ParabolaR_CRPH = [0.42126, -1.23900, -0.42364, 0.91103, -1.15538, 1.76697]
G3_ParabolaR_CRPL = [0.60196, -1.77048, -0.47171, 1.30183, -1.08467, 1.39727]

ParabolaRDataSet3 = pd.DataFrame(
    [G3_ParabolaR, G3_ParabolaR_CRPH, G3_ParabolaR_CRPL],
   index=["G3", "CRPH", "CRPL"],
   columns=["y^2", "x*y", "y", "x^2", "x", "c", "Sx", "Sy", "Ex", "Ey"]
)

print("")
print("########## (ParabolaDataSet3) ##########")
print(ParabolaRDataSet3)

img_01 = plt.imread("Guo_2020_Fig1B.png")
img_02 = plt.imread("Sugawa_2018_Fig1.png")


Patient_001 = [0.04, 2.01742]
Patient_002 = [0.08, 2.02749]
Patient_003 = [0.12, 2.04454]
Patient_004 = [0.16, 2.02009]
Patient_005 = [0.20, 2.03445]
Patient_006 = [0.24, 2.04749]
Patient_007 = [0.28, 2.03641]
Patient_008 = [0.32, 1.99047]
Patient_009 = [0.36, 1.98671]
Patient_010 = [0.40, 2.06711]
Patient_011 = [0.44, 2.09772]
Patient_012 = [0.48, 2.00539]
Patient_013 = [0.52, 1.86985]
Patient_014 = [0.56, 2.01679]
Patient_015 = [0.60, 2.12183]
Patient_016 = [0.64, 1.94453]
Patient_017 = [0.68, 1.86497]
Patient_018 = [0.72, 1.87444]
Patient_019 = [0.76, 2.09483]
Patient_020 = [0.80, 1.91073]
Patient_021 = [0.84, 1.69244]
Patient_022 = [0.88, 1.70968]
Patient_023 = [0.92, 1.81376]
Patient_024 = [0.96, 2.04219]
Patient_025 = [1.00, 1.69059]
Patient_026 = [1.04, 1.75939]
Patient_027 = [1.08, 1.95321]
Patient_028 = [1.12, 1.84170]
Patient_029 = [1.16, 1.58169]
Patient_030 = [1.20, 1.75585]
Patient_031 = [1.24, 1.73497]
Patient_032 = [1.28, 1.47847]
Patient_033 = [1.32, 1.91150]
Patient_034 = [1.36, 1.44962]
Patient_035 = [1.40, 1.85063]
Patient_036 = [1.44, 1.32298]
Patient_037 = [1.48, 1.28437]
Patient_038 = [1.52, 1.74621]
Patient_039 = [1.56, 1.27232]
Patient_040 = [1.60, 1.32763]
Patient_041 = [1.64, 1.57500]
Patient_042 = [1.68, 1.56609]
Patient_043 = [1.72, 1.59330]
Patient_044 = [1.76, 1.76707]
Patient_045 = [1.80, 1.11264]
Patient_046 = [1.84, 1.06188]
Patient_047 = [1.88, 1.40139]
Patient_048 = [1.92, 0.94084]
Patient_049 = [1.96, 1.07560]
Patient_050 = [2.00, 1.38507]
Patient_051 = [2.04, 1.33408]
Patient_052 = [2.08, 1.54375]
Patient_053 = [2.12, 1.60257]
Patient_054 = [2.16, 1.02029]
Patient_055 = [2.20, 0.98612]
Patient_056 = [2.24, 1.79094]
Patient_057 = [2.28, 0.97916]
Patient_058 = [2.32, 0.81212]
Patient_059 = [2.36, 1.14840]
Patient_060 = [2.40, 1.51680]
Patient_061 = [2.44, 1.70236]
Patient_062 = [2.48, 1.38945]
Patient_063 = [2.52, 1.69072]
Patient_064 = [2.56, 1.75042]
Patient_065 = [2.60, 1.67571]
Patient_066 = [2.64, 1.38623]
Patient_067 = [2.68, 1.81503]
Patient_068 = [2.72, 1.80665]
Patient_069 = [2.76, 1.41073]
Patient_070 = [2.80, 2.31750]
Patient_071 = [2.84, 2.24852]
Patient_072 = [2.88, 1.75950]
Patient_073 = [2.92, 2.81818]
Patient_074 = [2.96, 1.93654]
Patient_075 = [3.00, 2.36998]
Patient_076 = [3.04, 2.09870]
Patient_077 = [3.08, 2.19539]
Patient_078 = [3.12, 2.44747]
Patient_079 = [3.16, 2.57255]
Patient_080 = [3.20, 3.24637]
Patient_081 = [3.24, 2.78881]
Patient_082 = [3.28, 3.51638]
Patient_083 = [3.32, 2.68107]
Patient_084 = [3.36, 2.87259]
Patient_085 = [3.40, 4.37490]
Patient_086 = [3.44, 3.13393]
Patient_087 = [3.48, 3.92099]
Patient_088 = [3.52, 4.12230]
Patient_089 = [3.56, 3.78584]
Patient_090 = [3.60, 4.96660]
Patient_091 = [3.64, 5.48880]
Patient_092 = [3.68, 5.70712]
Patient_093 = [3.72, 4.84752]
Patient_094 = [3.76, 5.14090]
Patient_095 = [3.80, 6.31637]
Patient_096 = [3.84, 5.53198]
Patient_097 = [3.88, 6.89257]
Patient_098 = [3.92, 7.86387]
Patient_099 = [3.96, 7.98237]
Patient_100 = [4.00, 8.64705]

TestData = pd.DataFrame(
    [Patient_001, Patient_002, Patient_003, Patient_004, Patient_005, Patient_006,
     Patient_007, Patient_008, Patient_009, Patient_010, Patient_011, Patient_012,
     Patient_013, Patient_014, Patient_015, Patient_016, Patient_017, Patient_018,
     Patient_019, Patient_020, Patient_021, Patient_022, Patient_023, Patient_024,
     Patient_025, Patient_026, Patient_027, Patient_028, Patient_029, Patient_030,
     Patient_031, Patient_032, Patient_033, Patient_034, Patient_035, Patient_036,
     Patient_037, Patient_038, Patient_039, Patient_040, Patient_041, Patient_042,
     Patient_043, Patient_044, Patient_045, Patient_046, Patient_047, Patient_048,
     Patient_049, Patient_050, Patient_051, Patient_052, Patient_053, Patient_054,
     Patient_055, Patient_056, Patient_057, Patient_058, Patient_059, Patient_060,
     Patient_061, Patient_062, Patient_063, Patient_064, Patient_065, Patient_066,
     Patient_067, Patient_068, Patient_069, Patient_070, Patient_071, Patient_072,
     Patient_073, Patient_074, Patient_075, Patient_076, Patient_077, Patient_078,
     Patient_079, Patient_080, Patient_081, Patient_082, Patient_083, Patient_084,
     Patient_085, Patient_086, Patient_087, Patient_088, Patient_089, Patient_090,
     Patient_091, Patient_092, Patient_093, Patient_094, Patient_095, Patient_096,
     Patient_097, Patient_098, Patient_099, Patient_100],
    index=["Patient_001", "Patient_002", "Patient_003", "Patient_004", "Patient_005", "Patient_006",
           "Patient_007", "Patient_008", "Patient_009", "Patient_010", "Patient_011", "Patient_012",
           "Patient_013", "Patient_014", "Patient_015", "Patient_016", "Patient_017", "Patient_018",
           "Patient_019", "Patient_020", "Patient_021", "Patient_022", "Patient_023", "Patient_024",
           "Patient_025", "Patient_026", "Patient_027", "Patient_028", "Patient_029", "Patient_030",
           "Patient_031", "Patient_032", "Patient_033", "Patient_034", "Patient_035", "Patient_036",
           "Patient_037", "Patient_038", "Patient_039", "Patient_040", "Patient_041", "Patient_042",
           "Patient_043", "Patient_044", "Patient_045", "Patient_046", "Patient_047", "Patient_048",
           "Patient_049", "Patient_050", "Patient_051", "Patient_052", "Patient_053", "Patient_054",
           "Patient_055", "Patient_056", "Patient_057", "Patient_058", "Patient_059", "Patient_060",
           "Patient_061", "Patient_062", "Patient_063", "Patient_064", "Patient_065", "Patient_066",
           "Patient_067", "Patient_068", "Patient_069", "Patient_070", "Patient_071", "Patient_072",
           "Patient_073", "Patient_074", "Patient_075", "Patient_076", "Patient_077", "Patient_078",
           "Patient_079", "Patient_080", "Patient_081", "Patient_082", "Patient_083", "Patient_084",
           "Patient_085", "Patient_086", "Patient_087", "Patient_088", "Patient_089", "Patient_090",
           "Patient_091", "Patient_092", "Patient_093", "Patient_094", "Patient_095", "Patient_096",
           "Patient_097", "Patient_098", "Patient_099", "Patient_100"],
    columns=["Value_X", "Value_Y"]
)

print("")
print("###############################################################################################################")
print("########## Test Data (TestData) ##########")
print(TestData)

print("")
print("###############################################################################################################")
print("########## Linior Regression Analysis Results by R (Coefficients_Line) ##########")
L_0 = ["(Intercept)                                     ",   "0.7280",     "0.2535",   "2.871",  "0.00501", "**" ]
L_1 = ["poly(TestData$Value_X, degree = 1, raw = TRUE)  ",   "0.8313",     "0.1090",   "7.629", "1.56e-11", "***"]
Coefficients_Line = pd.DataFrame(
    [L_0, L_1],
    index=["L_0", "L_1"],
    columns=["Coefficients", "Estimate", "Std. Error", "t value", "Pr(>|t|)", "Significance"])
print(Coefficients_Line)

print("")
print("###############################################################################################################")
print("########## Polynomial Regression Analysis Results by R (Coefficients_Curve) ##########")
C_0 = ["(Intercept)                                     ",  "2.14594",    "0.17485",  "12.273",  "< 2e-16", "***"]
C_1 = ["poly(TestData$Value_X, degree = 4, raw = TRUE) 1", "-0.54052",    "0.59336", " -0.911", "0.364629", "   "]
C_2 = ["poly(TestData$Value_X, degree = 4, raw = TRUE) 2",  "0.49655",    "0.59306",   "0.837", "0.404539", "   "]
C_3 = ["poly(TestData$Value_X, degree = 4, raw = TRUE) 3", "-0.42523",    "0.22004",  "-1.933", "0.056275", ".  "]
C_4 = ["poly(TestData$Value_X, degree = 4, raw = TRUE) 4",  "0.10680",    "0.02702",   "3.952", "0.000149", "***"]
Coefficients_Curve = pd.DataFrame(
    [C_0, C_1, C_2, C_3, C_4],
    index=["C_0", "C_1", "C_2", "C_3", "C_4"],
    columns=["Coefficients", "Estimate", "Std. Error", "t value", "Pr(>|t|)", "Significance"])
print(Coefficients_Curve)

########################################################################################################################
# Figures
########################################################################################################################
Figure_object = plt.figure(1, figsize=(11.69, 8.27), dpi=400, edgecolor="black", linewidth=0.5)
plt.subplots_adjust(left=0.125, bottom=0.11, right=0.9, top=0.88, wspace=0.2, hspace=0.2)

#plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)
#gs_master = matplotlib.gridspec.GridSpec(nrows=2, ncols=3)

plt.figtext(0.0300, 0.9700, "a", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")
plt.figtext(0.3333+0.02, 0.9700, "b", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")
plt.figtext(0.6666+0.02, 0.9700, "c", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")

plt.figtext(0.0300, 0.4750, "d", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")
plt.figtext(0.3333+0.02, 0.4750, "e", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")
plt.figtext(0.6666+0.02, 0.4750, "f", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")

Axes_Outer_A = Figure_object.add_subplot(2, 3, 1)
Axes_Outer_B = Figure_object.add_subplot(2, 3, 2)
Axes_Outer_C = Figure_object.add_subplot(2, 3, 3)
Axes_Outer_D = Figure_object.add_subplot(2, 3, 4)
Axes_Outer_E = Figure_object.add_subplot(2, 3, 5)
Axes_Outer_F = Figure_object.add_subplot(2, 3, 6)

Axes_Outer_A.set_position([0.0128315, 0.513352, 0.316225, 0.423053])
Axes_Outer_B.set_position([0.341888, 0.513352, 0.316225, 0.423053])
Axes_Outer_C.set_position([0.670944, 0.513352, 0.316225, 0.423053])
Axes_Outer_D.set_position([0.0128315, 0.0181378, 0.316225, 0.423053])
Axes_Outer_E.set_position([0.341888, 0.0181378, 0.316225, 0.423053])
Axes_Outer_F.set_position([0.670944, 0.0181378, 0.316225, 0.423053])

#Axes_Outer_A = Figure_object.add_subplot(gs_master[0])
#Axes_Outer_B = Figure_object.add_subplot(gs_master[1])
#Axes_Outer_C = Figure_object.add_subplot(gs_master[2])
#Axes_Outer_D = Figure_object.add_subplot(gs_master[3])
#Axes_Outer_E = Figure_object.add_subplot(gs_master[4])
#Axes_Outer_F = Figure_object.add_subplot(gs_master[5])

########################################################################################################################
# Figure (a)
########################################################################################################################
Axes_Outer_A.set_title("Guo et al. JAMA Cardiol. 2020\nCOVID-19 (Wuhan, China)", size=11.4230, fontweight="normal")
Axes_Outer_A.tick_params(length=0.0)
Axes_Outer_A.set_xticklabels([])
Axes_Outer_A.set_yticklabels([])
#----------------------------------------------------------------------------------------------------------------------#
Axes_obj_01 = mpl_inset.inset_axes(Axes_Outer_A, width="175.0%", height="175.0%",
                                   bbox_to_anchor=(0.2500, 0.400, 0.5, 0.5),
                                   bbox_transform=Axes_Outer_A.transAxes, loc="center", borderpad=1)
Axes_obj_01.tick_params(length=0.0); Axes_obj_01.set_xticklabels([]); Axes_obj_01.set_yticklabels([])
Axes_obj_01.imshow(plt.imread("Guo_2020_Fig1B.png"))
Axes_obj_01.annotate(
    text='\n'.join(["Total patients: 187"]),
    xy=(280, 50), xytext=(280, 50), ha='center', va='center', size=14.5, color="red",
    bbox=dict(boxstyle='round', edgecolor='red', fc='white'), fontweight="normal", zorder=10,
)
#----------------------------------------------------------------------------------------------------------------------#
Axes_obj_02 = mpl_inset.inset_axes(Axes_Outer_A, width="80.00%", height="45.00%",
                                   bbox_to_anchor=(0.0+0.025, -0.125, 0.5, 0.5),
                                   bbox_transform=Axes_Outer_A.transAxes, loc="center", borderpad=1)
Axes_obj_02.set_title("Over 100 Points?", fontweight="normal")
Axes_obj_02.tick_params(length=0.0); Axes_obj_02.set_xticklabels([]); Axes_obj_02.set_yticklabels([])

img = Image.open("Guo_2020_Fig1B_origin.png")
img_resize = img.resize(size=(125, 70))#size in pixels, as a 2-tuple: (width, height)
img_resize.save("Guo_2020_Fig1B_origin_resized.png")
Axes_obj_02.imshow(plt.imread("Guo_2020_Fig1B_origin_resized.png"), interpolation='none', aspect='auto', cmap="gray")
#----------------------------------------------------------------------------------------------------------------------#
Axes_obj_03 = mpl_inset.inset_axes(Axes_Outer_A, width="80.00%", height="45.00%",
                                   bbox_to_anchor=(0.5-0.025, -0.125, 0.5, 0.5),
                                   bbox_transform=Axes_Outer_A.transAxes, loc="center", borderpad=1)
Axes_obj_03.set_title("Significance??", fontweight="normal")
Axes_obj_03.tick_params(length=0.0); Axes_obj_03.set_xticklabels([]); Axes_obj_03.set_yticklabels([])
Axes_obj_03.imshow(plt.imread("Guo_2020_Fig1B_P_value_cut.png"), interpolation='none', aspect='auto')

########################################################################################################################
# Figure (b)
########################################################################################################################
Axes_Outer_B.set_title("Marginal Dist. for Guo et al.\nCOVID-19 (same as on the left)", size=11.4230, fontweight="normal")
Axes_Outer_B.tick_params(length=0.0)
Axes_Outer_B.set_xticklabels([])
Axes_Outer_B.set_yticklabels([])
#----------------------------------------------------------------------------------------------------------------------#
Axes_obj_04 = mpl_inset.inset_axes(Axes_Outer_B, width="130.0%", height="125.0%",
                                   bbox_to_anchor=(0.4000, 0.4100, 0.5, 0.5),
                                   bbox_transform=Axes_Outer_B.transAxes, loc="center", borderpad=1)
Axes_obj_04.set_xticklabels([]); Axes_obj_04.set_yticklabels([])
Axes_obj_04.set_xlim(-0.125, 2.625)
Axes_obj_04.set_ylim(-0.125, 4.125)
Axes_obj_04.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
Axes_obj_04.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))

for i in range(0, 9):
    Axes_obj_04.hlines(y=0.0+0.5*i, xmin=0.0, xmax=2.5, color="gray", linestyle="dotted", linewidth=0.5)
for i in range(0, 6):
    Axes_obj_04.vlines(x=0.0+0.5*i, ymin=0.0, ymax=4.0, color="gray", linestyle="dotted", linewidth=0.5)

Axes_obj_04.scatter(PatientDataSet["Plasma TnT"], PatientDataSet["Plasma NT-proBNP"],
                    color="black", marker="s", s=20, zorder=9)

x = np.arange(0, max(PatientDataSet["Plasma TnT"]), 1.0*(10**(-3)))
y = CenterOfGravityDataSet.at["Guo et al.", "a1"]*x + CenterOfGravityDataSet.at["Guo et al.", "b1"]

Axes_obj_04.plot(x, y, color="black", linestyle="solid", linewidth=1.5)

Axes_obj_04.vlines(x=0.5, ymin=-0.125, ymax=4.125, color="red", linestyle='dashed', linewidth=1.5)
Axes_obj_04.hlines(y=1.125, xmin=-0.125, xmax=0.5, color="red", linestyle='dashed', linewidth=1.5)

x = np.arange(0.55, 2.0, 1.0*(10**(-3)))
y = CenterOfGravityDataSet.at["Guo et al.", "a1"]*x + 1.45

xu = np.arange(0.55, 2.0, 1.0*(10**(-3)))
yu = CenterOfGravityDataSet.at["Guo et al.", "a1"]*x + 1.30
#Axes_obj_05.plot(xu, yu, color="blue", linestyle="dashed", linewidth=1.5)
xu1 = min(xu); xu2 = max(xu); yu1 = min(yu); yu2 = max(yu)

xl = np.arange(1.05, 2.5, 1.0*(10**(-3)))
yl = CenterOfGravityDataSet.at["Guo et al.", "a1"]*x - 0.85
#Axes_obj_05.plot(xl, yl, color="blue", linestyle="dashed", linewidth=1.5)
xl1 = min(xl); xl2 = max(xl); yl1 = min(yl); yl2 = max(yl)

B_1 = ParabolaRDataSet3.at["G3", "y^2"]
B_2 = ParabolaRDataSet3.at["G3", "x*y"]
B_3 = ParabolaRDataSet3.at["G3", "y"]
B_4 = ParabolaRDataSet3.at["G3", "x^2"]
B_5 = ParabolaRDataSet3.at["G3", "x"]
B_6 = ParabolaRDataSet3.at["G3", "c"]
x1 = np.arange(0.898070, 1.025, 1.0*(10**(-3)))
x2 = np.arange(0.898070, 2.10, 1.0*(10**(-3)))

y1 = (np.sqrt((B_2**2-4*B_1*B_4)*x1**2+(2*B_2*B_3-4*B_1*B_5)*x1-4*B_1*B_6+B_3**2)-B_2*x1-B_3)/(2*B_1)
y2 = -(np.sqrt((B_2**2-4*B_1*B_4)*x2**2+(2*B_2*B_3-4*B_1*B_5)*x2-4*B_1*B_6+B_3**2)+B_2*x2+B_3)/(2*B_1)
Axes_obj_04.plot(x1, y1, color="blue", linestyle="solid", linewidth=5.0, zorder=7)
Axes_obj_04.plot(x2, y2, color="blue", linestyle="solid", linewidth=5.0, zorder=7)

Axes_obj_04.annotate(
    text='\n'.join(["cf. Budnik et al.", "    Int J Cardiol.", "    2016 15; 219"]),
    xy=(1.675, 3.25), xytext=(1.675, 3.25), ha='center', va='center', size=11.0, color="blue",
    bbox=dict(boxstyle='round', edgecolor='blue', fc='white'), fontweight="normal"
)

Axes_obj_04.annotate(
    text='\n'.join(["A(1)"]),
    xy=(0.2, 0.5), xytext=(0.2, 0.5), ha='center', va='center', size=10, color="red",
    bbox=dict(boxstyle='round', edgecolor='red', fc='white'), fontweight="normal", zorder=10,
)

Axes_obj_04.annotate(
    text='\n'.join(["A(2)"]),
    xy=(0.2, 2.0), xytext=(0.2, 2.0), ha='center', va='center', size=10, color="red",
    bbox=dict(boxstyle='round', edgecolor='red', fc='white'), fontweight="normal", zorder=10,
)

Axes_obj_04.fill_between(np.linspace(-0.125, 0.5, 1000), -0.125*np.ones(1000), 4.125*np.ones(1000),
                         facecolor='red', alpha=0.1)

x1 = np.arange(29/50, 1.3836, 1.0*(10**(-3)))
x2 = np.arange(29/50, 62/25, 1.0*(10**(-3)))
x3 = np.arange(-(6*np.sqrt(152886985058)-5890213)/2909614, 1.3836, 1.0*(10**(-3)))
x4 = np.arange(-(6*np.sqrt(152886985058)-5890213)/2909614, 2.44602, 1.0*(10**(-3)))
x5 = np.arange(2.44602, 62/25, 1.0*(10**(-3)))

y1 = (2**(3/2)*np.sqrt(-1250*x1**2+3825*x1-1798)+167)/100
y2 = -(2**(3/2)*np.sqrt(-1250*x2**2+3825*x2-1798)-167)/100
y3 = (np.sqrt(-5819228*x3**2+23560852*x3-20064983)+1322*x3+2437)/2088
y4 = -(np.sqrt(-5819228*x4**2+23560852*x4-20064983)-1322*x4-2437)/2088
y5 = (2**(3/2)*np.sqrt(-1250*x5**2+3825*x5-1798)+167)/100

Axes_obj_04.plot(x1, y1, color="blue", linestyle="dashed", linewidth=1.5, zorder=7)
Axes_obj_04.plot(x2, y2, color="blue", linestyle="dashed", linewidth=1.5, zorder=7)
Axes_obj_04.plot(x3, y3, color="blue", linestyle="dashed", linewidth=1.5, zorder=7)
Axes_obj_04.plot(x4, y4, color="blue", linestyle="dashed", linewidth=1.5, zorder=7)
Axes_obj_04.plot(x5, y5, color="blue", linestyle="dashed", linewidth=1.5, zorder=7)

A = 29/50
B = -(6*np.sqrt(152886985058)-5890213)/2909614
X = np.linspace(A, B, 1000)
Y1 = (2**(3/2)*np.sqrt(-1250*X**2+3825*X-1798)+167)/100
Y2 = -(2**(3/2)*np.sqrt(-1250*X**2+3825*X-1798)-167)/100
Axes_obj_04.fill_between(X, Y2, Y1, facecolor='blue', alpha=0.1)

A = -(6*np.sqrt(152886985058)-5890213)/2909614
B = 1.3836
X = np.linspace(A, B, 1000)
Y1 = (2**(3/2)*np.sqrt(-1250*X**2+3825*X-1798)+167)/100
Y2 = (np.sqrt(-5819228*X**2+23560852*X-20064983)+1322*X+2437)/2088
Axes_obj_04.fill_between(X, Y2, Y1, facecolor='blue', alpha=0.1)

A = -(6*np.sqrt(152886985058)-5890213)/2909614
B = 2.44602
X = np.linspace(A, B, 1000)
Y1 = -(np.sqrt(-5819228*X**2+23560852*X-20064983)-1322*X-2437)/2088
Y2 = -(2**(3/2)*np.sqrt(-1250*X**2+3825*X-1798)-167)/100
Axes_obj_04.fill_between(X, Y2, Y1, facecolor='blue', alpha=0.1)

A = 2.44602
B = 62/25
X = np.linspace(A, B, 1000)
Y1 = (2**(3/2)*np.sqrt(-1250*X**2+3825*X-1798)+167)/100
Y2 = -(2**(3/2)*np.sqrt(-1250*X**2+3825*X-1798)-167)/100
Axes_obj_04.fill_between(X, Y2, Y1, facecolor='blue', alpha=0.1)

#----------------------------------------------------------------------------------------------------------------------#
Axes_obj_05 = mpl_inset.inset_axes(Axes_Outer_B, width="130.0%", height="40.00%",
                                   bbox_to_anchor=(0.4000, -0.025, 0.5, 0.5),
                                   bbox_transform=Axes_Outer_B.transAxes, loc="center", borderpad=1)
Axes_obj_05.set_xlabel("Plasma Troponin T, ng/mL", fontweight="normal", size=10)
Axes_obj_05.set_ylabel("Freq.")
Axes_obj_05.set_xlim(-0.125, 2.625)
Axes_obj_05.set_yticklabels([]);
Axes_obj_05.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
Axes_obj_05.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
Axes_obj_05.set_xticklabels([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
Axes_obj_05.hist(PatientDataSet["Plasma TnT"],
                 range=(-0.125, 2.625), bins=12, edgecolor="black", color="gray", alpha=0.4, orientation="vertical")

from scipy.stats import gaussian_kde
kde_model = gaussian_kde(PatientDataSet["Plasma TnT"])
x = np.linspace(-0.125, 2.625, num=100)
y = kde_model(x)
Axes_obj_05.plot(x, y*12.5, color="blue")

for i in range(0, len(PatientDataSet["Plasma TnT"])):
    Axes_obj_05.scatter(PatientDataSet["Plasma TnT"][i], 0.15, facecolor="none", color="black", marker="s", s=20, zorder=9)

Axes_obj_05.text(0.125, 7.0, "A", color="red", size=20,
                 horizontalalignment="center", verticalalignment="center", rotation="horizontal", fontweight="normal")

Axes_obj_05.text(0.90, 6.75, "B", color="blue", size=20,
                 horizontalalignment="center", verticalalignment="center", rotation="horizontal", fontweight="normal")

#----------------------------------------------------------------------------------------------------------------------#
Axes_obj_06 = mpl_inset.inset_axes(Axes_Outer_B, width="35.00%", height="125.0%",
                                   bbox_to_anchor=(-0.035, 0.4100, 0.5, 0.5),
                                   bbox_transform=Axes_Outer_B.transAxes, loc="center", borderpad=1)
Axes_obj_06.set_xticklabels([]);
Axes_obj_06.set_ylim(-0.125, 4.125)
Axes_obj_06.set_xlabel("Freq.")
Axes_obj_06.set_ylabel("NT-proBNP, ×10 μg/mL", fontweight="normal", size=10)
Axes_obj_06.set_yticks = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
Axes_obj_06.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))

Axes_obj_06.hist(PatientDataSet["Plasma NT-proBNP"],
                 range=(-0.125, 4.125), bins=12, edgecolor="black", color="gray", alpha=0.4, orientation="horizontal")
Axes_obj_06.invert_xaxis()

from scipy.stats import gaussian_kde
kde_model = gaussian_kde(PatientDataSet["Plasma NT-proBNP"])
x = np.linspace(-0.125, 4.125, num=100)
y = kde_model(x)
Axes_obj_06.plot(y*15, x, color="blue")

for i in range(0, len(PatientDataSet["Plasma NT-proBNP"])):
    Axes_obj_06.scatter(0.15, PatientDataSet["Plasma NT-proBNP"][i],
                        facecolor="none", color="black", marker="s", s=20, zorder=9)

########################################################################################################################
# Figure (C)
########################################################################################################################
Axes_Outer_C.set_title("Wang, Y et al. Front Cardiovasc Med. 2020\nCOVID-19 (Wuhan, China)", size=11.4230, fontweight="normal")
Axes_Outer_C.tick_params(length=0.0)
Axes_Outer_C.set_xticklabels([])
Axes_Outer_C.set_yticklabels([])
Axes_Outer_C.text(0.95, 0.5, "Red items: annotated by Kimoto et al. 2021",
                 size=11.4230, color="red", horizontalalignment="center", verticalalignment="center", rotation=90)
#----------------------------------------------------------------------------------------------------------------------#
Axes_obj_07 = mpl_inset.inset_axes(Axes_Outer_C, width="190.0%", height="190.0%",
                                   bbox_to_anchor=(0.25, 0.25, 0.5, 0.5),
                                   bbox_transform=Axes_Outer_C.transAxes, loc="center", borderpad=1,
                                   axes_kwargs={"facecolor": "lightgreen"})

Axes_obj_07.tick_params(length=0.0); Axes_obj_07.set_xticklabels([]); Axes_obj_07.set_yticklabels([])
Axes_obj_07.spines["right"].set_visible(False); Axes_obj_07.spines["left"].set_visible(False)
Axes_obj_07.spines["top"].set_visible(False); Axes_obj_07.spines["bottom"].set_visible(False)
Axes_obj_07.imshow(plt.imread("Wang_Y_et_al_Front_Cardiovasc_Med_2020_25_7_147_Troponin_BNP_Lymphocyte_02.png"))

########################################################################################################################
# Figure (d)
########################################################################################################################
Axes_Outer_D.set_title("Caro-Codón et al. Eur J Heart Fail. 2020\nCOVID-19 (Madrid, Spain)", size=11.4230, fontweight="normal")
Axes_Outer_D.tick_params(length=0.0)
Axes_Outer_D.set_xticklabels([])
Axes_Outer_D.set_yticklabels([])

#----------------------------------------------------------------------------------------------------------------------#
Axes_obj_08 = mpl_inset.inset_axes(Axes_Outer_D, width="175.0%", height="175.0%",
                                   bbox_to_anchor=(0.300, 0.275, 0.5, 0.5),
                                   bbox_transform=Axes_Outer_D.transAxes, loc="center", borderpad=1)
Axes_obj_08.tick_params(length=0.0); Axes_obj_08.set_xticklabels([]); Axes_obj_08.set_yticklabels([])
Axes_obj_08.set_xlabel("LN(Troponin I), ng/L", fontweight="normal", size=10)
Axes_obj_08.set_ylabel("Percentage of patients (%)", fontweight="normal", size=10)
Axes_obj_08.imshow(plt.imread("Juan_Caro-Codón_et_al_2021_European_Journal_of_Heart_Failure_Figure_1B.png"))
Axes_obj_08.text(325, 125, "Blue items: annotated by\nKimoto et al. 2021",
                 size=11.4230, color="blue", horizontalalignment="center", verticalalignment="center",
                  rotation=90)
########################################################################################################################
# Figure (e)
########################################################################################################################
Axes_Outer_E.set_title("Demir et al. Am J Cardiol. 2021\nCOVID-19 (London, United Kingdom)", size=11.4230, fontweight="normal")
Axes_Outer_E.tick_params(length=0.0)
Axes_Outer_E.set_xticklabels([])
Axes_Outer_E.set_yticklabels([])
Axes_Outer_E.text(0.90, 0.5, "Red items (arrows, letters, and solid line \nfor comparing the distributions): \nannotated by Kimoto et al. 2021",
                 size=11.4230, color="red", horizontalalignment="center", verticalalignment="center", rotation=90)
#----------------------------------------------------------------------------------------------------------------------#
Axes_obj_09 = mpl_inset.inset_axes(Axes_Outer_E, width="190.0%", height="190.0%",
                                   bbox_to_anchor=(0.25, 0.25, 0.5, 0.5),
                                   bbox_transform=Axes_Outer_E.transAxes, loc="center", borderpad=1,
                                   axes_kwargs={"facecolor": "lightgreen"})

Axes_obj_09.tick_params(length=0.0); Axes_obj_09.set_xticklabels([]); Axes_obj_09.set_yticklabels([])
Axes_obj_09.spines["right"].set_visible(False); Axes_obj_09.spines["left"].set_visible(False)
Axes_obj_09.spines["top"].set_visible(False); Axes_obj_09.spines["bottom"].set_visible(False)
Axes_obj_09.imshow(plt.imread("Demir_et_al_Am_J_Cardiol_2021_15_147_129_136_Figure_2_merged.jpg"))

########################################################################################################################
# Figure (f)
########################################################################################################################
Axes_Outer_F.set_title("--------------- EXAMPLE ---------------\nVirtual Example for P-value (N=100)",
                       size=11.4230, fontweight="normal")
Axes_Outer_F.tick_params(length=0.0)
Axes_Outer_F.set_xticklabels([])
Axes_Outer_F.set_yticklabels([])
#----------------------------------------------------------------------------------------------------------------------#
Axes_obj_10 = mpl_inset.inset_axes(Axes_Outer_F, width="165.5%", height="165.5%",
                                   bbox_to_anchor=(0.32, 0.3, 0.5, 0.5),
                                   bbox_transform=Axes_Outer_F.transAxes, loc="center", borderpad=1)
Axes_obj_10.set_xlabel("X: Explanatory Variable", fontweight="normal", size=12.5)
Axes_obj_10.set_ylabel("Y: Objective Variable", fontweight="normal", size=12.5)
Axes_obj_10.set_xlim(-0.25, 12.125)
Axes_obj_10.set_ylim(-0.25, 10.125)
Axes_obj_10.set_xticks = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
Axes_obj_10.set_yticks = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
Axes_obj_10.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
Axes_obj_10.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))

Axes_obj_10.scatter(TestData["Value_X"], TestData["Value_Y"], facecolor="none", color="black", marker="s", s=20, zorder=1)

x = np.arange(0.0, 5.25, 1.0*(10**(-3)))
y = float(Coefficients_Line.at["L_1", "Estimate"])*x + float(Coefficients_Line.at["L_0", "Estimate"])
Axes_obj_10.plot(x, y, color="red", linestyle="solid", linewidth=1.5)

x = np.arange(0.0, 4.125, 1.0*(10**(-3)))
y = float(Coefficients_Curve.at["C_4", "Estimate"])*x**4 +\
    float(Coefficients_Curve.at["C_3", "Estimate"])*x**3 +\
    float(Coefficients_Curve.at["C_2", "Estimate"])*x**2 +\
    float(Coefficients_Curve.at["C_1", "Estimate"])*x**1 +\
    float(Coefficients_Curve.at["C_0", "Estimate"])*x**0

Axes_obj_10.plot(x, y, color="blue", linestyle="solid", linewidth=1.5)

x = np.arange(0.0, 4.125, 1.0*(10**(-3)))
y = float(Coefficients_Curve.at["C_4", "Estimate"])*x**4 +\
    float(Coefficients_Curve.at["C_0", "Estimate"])*x**0
Axes_obj_10.plot(x, y, color="blue", linestyle="dashed", linewidth=1.0)

Axes_obj_10.text(7.0, 9.35, "Misuse", horizontalalignment='left', fontsize=15.0, fontweight="normal", color="red")
Axes_obj_10.text(6.0, 8.25, r"$y = \beta_{0} + \beta_{1}x$", horizontalalignment='left', fontsize=15.0, fontweight="normal", color="red")
Axes_obj_10.text(6.0, 7.25, r"$\beta_{1}:  P < .001$", horizontalalignment='left', fontsize=15.0, fontweight="normal", color="red")
Axes_obj_10.text(6.0, 6.3-0.25, r"$Linear$", horizontalalignment='left', fontsize=15.0, fontweight="normal", color="red")
Axes_obj_10.text(6.0, 5.4-0.25, r"$relationship??$", horizontalalignment='left', fontsize=15.0, fontweight="normal", color="red")

Axes_obj_10.text(4.0, 3.00, r"$y = \beta_{0} + \beta_{1}x + \beta_{2}x^2$", horizontalalignment='left', fontsize=15.0, fontweight="normal", color="blue")
Axes_obj_10.text(6.3, 2.00, r"$ + \beta_{3}x^3 + \beta_{4}x^4$", horizontalalignment='left', fontsize=15.0, fontweight="normal", color="blue")
Axes_obj_10.text(4.75, 0.8, r"$\beta_{4}, \beta_{0}:  P < .001$", horizontalalignment='left', fontsize=15.0, fontweight="normal", color="blue")
Axes_obj_10.text(4.125, 0.0, r"$(important$", horizontalalignment='left', fontsize=15.0, fontweight="normal", color="blue")
Axes_obj_10.text(9.125, 0.0, r"$items)$", horizontalalignment='left', fontsize=15.0, fontweight="normal", color="blue")

Axes_obj_10.text(0.25, 9.25, r"$Only$", horizontalalignment='left', fontsize=15.0, fontweight="normal", color="blue")
Axes_obj_10.text(0.0, 8.25, r"$\beta_{4} & \beta_{0}$", horizontalalignment='left', fontsize=15.0, fontweight="normal", color="blue")

Axes_obj_10.quiver(1.2, 7.6, 1.4, -0.2,
                   scale_units='xy', angles='xy', scale=1, linestyle='solid', linewidth=0.55, width=0.016, color="blue")

Axes_obj_10.annotate(
    text='\n'.join(["                         ", "                    ", "                    ", "                    ", "                    "]),
    xy=(8.85-0.1, 7.0), xytext=(8.85-0.1, 7.0), ha='center', va='center', size=15.0, color="red",
    bbox=dict(boxstyle='round', edgecolor='red', fc='none'), fontweight="normal", zorder=10,
)
########################################################################################################################

#Figure_object.tight_layout()
my_path = os.path.abspath("Figures")
plt.savefig(os.path.join(my_path, "01_Kimoto_et_al_(2023)_[Fig_S_12].png"))

from PIL import Image # py -m pip install pillow
img = Image.open(os.path.join(my_path, "01_Kimoto_et_al_(2023)_[Fig_S_12].png"))
img_resize = img.resize(size=(2866, 2016))
my_path2 = os.path.abspath("Figures/Fixed_Version/B6")
img_resize.save(os.path.join(my_path2, "01_Kimoto_et_al_(2023)_[Fig_S_12]_B6.png"))

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