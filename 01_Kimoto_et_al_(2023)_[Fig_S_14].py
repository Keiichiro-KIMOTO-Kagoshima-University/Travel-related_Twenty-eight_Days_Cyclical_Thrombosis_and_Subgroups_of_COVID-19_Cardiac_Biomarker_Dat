########################################################################################################################
# Fig. S14. Three subgroup patterns appeared in a figure reported by Guo et al.
# (01_Kimoto_et_al_(2023)_[Fig_S_14].py)
########################################################################################################################
# Title of the manuscript (research):
# Travel-related Twenty-eight Days Cyclical Thrombosis and Subgroups of COVID-19 Cardiac Biomarker Data:
# Novel Review Strategy and Meta-analysis Method
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
################################

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

print("")
print("########## Patient DataSet (Guo et al. 2020) (PatientDataSet) ##########")
print(PatientDataSet)
############################################################################################################
# () ########## Patient DataSet (Sugawa et al. 2018) (PatientDataSet_S) ##########
############################################################################################################
Point_01 = [0, 46.91427384]
Point_02 = [0, 28.3994448]
Point_03 = [0.207770912, 78.21390465]
Point_04 = [0.207770912, 39.86091571]
Point_05 = [1.453951477, 98.05112032]
Point_06 = [1.453951477, 32.36660466]
Point_07 = [2.077041759, 84.82630988]
Point_08 = [2.492583583, 39.4199628]
Point_09 = [3.323444777, 42.50616107]
Point_10 = [3.738764148, 29.28087851]
Point_11 = [4.154305971, 32.80755757]
Point_12 = [4.569625342, 66.31100872]
Point_13 = [4.985167165, 174.7566261]
Point_14 = [6.854438013, 33.6894634]
Point_15 = [7.685299207, 28.3994448]
Point_16 = [10.17810524, 34.57089711]
Point_17 = [12.67068883, 59.69860349]
Point_18 = [14.1246403, 137.285543]
Point_19 = [19.10980747, 192.3900214]
Point_20 = [21.80993951, 103.7820918]
Point_21 = [29.70323208, 36.77518955]
Point_22 = [36.97321192, 27.07705817]
Point_23 = [46.11268505, 35.01185002]
Point_24 = [106.3502329, 58.81669767]

PatientDataSet_S = pd.DataFrame(
    [Point_01, Point_02, Point_03, Point_04, Point_05, Point_06, Point_07, Point_08, Point_09, Point_10,
     Point_11, Point_12, Point_13, Point_14, Point_15, Point_16, Point_17, Point_18, Point_19, Point_20,
     Point_21, Point_22, Point_23, Point_24],
   index=["Point_01", "Point_02", "Point_03", "Point_04", "Point_05",
          "Point_06", "Point_07", "Point_08", "Point_09", "Point_10",
          "Point_11", "Point_12", "Point_13", "Point_14", "Point_15",
          "Point_16", "Point_17", "Point_18", "Point_19", "Point_20",
          "Point_21", "Point_22", "Point_23", "Point_24"],
   columns=["X(BNP)", "Y(TnT)"]
)

print("")
print("########## Patient DataSet (Sugawa et al. 2018) (PatientDataSet_S) ##########")
print(PatientDataSet_S)
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
# () ########## Parabola (ParabolaDataSet) ##########
############################################################################################################
G1_ParabolaU = ["", np.NaN, 2/3, 16.6873, 0.0725, 0.0000]
G1_ParabolaR = ["y = 25.031x^2 - 2.4775x - 0.1503", 0.5753, 1, 25.031, 0.0495, -0.2116]
G1_ParabolaL = ["", np.NaN, 3/2, 37.5465, 0.0265, -0.4232]
G2_ParabolaU = ["", np.NaN, 2/3, 18.0647, 0.1732, 0.0000]
G2_ParabolaR = ["y = 27.097x^2 - 7.0563x - 0.3422", 0.6395, 1, 27.097, 0.1302, -0.8016]
G2_ParabolaL = ["", np.NaN, 3/2, 40.6455, 0.0872, -1.6032]
G3_ParabolaU = ["", np.NaN, 2/3, 1.0225, 0.4649, 0.0000]
G3_ParabolaR = ["y = 1.5337x^2 - 0.9751x - 0.3752", 0.4931, 1, 1.5337, 0.31789, -0.5302]
G3_ParabolaL = ["", np.NaN, 3/2, 2.3006, 0.1709, -1.0604]

ParabolaDataSet = pd.DataFrame(
    [G1_ParabolaU, G1_ParabolaR, G1_ParabolaL,
     G2_ParabolaU, G2_ParabolaR, G2_ParabolaL,
     G3_ParabolaU, G3_ParabolaR, G3_ParabolaL],
   index=["G1_U", "G1_R", "G1_L", "G2_U", "G2_R", "G2_L", "G3_U", "G3_R", "G3_L"],
   columns=["Equation", "R^2", "Coef.", "a", "Px", "Py"]
)

print("")
print("########## Parabola (ParabolaDataSet) ##########")
print(ParabolaDataSet)
############################################################################################################
# () ########## Solution (SolutionDataSet) ##########
############################################################################################################
SolutionDataG1 = [-0.0425, 0.0000, 0.1414, 0.0000, 0.0725, 0.0000, -0.16, 0.93, 0.14, 0.08]
SolutionDataG2 = [-0.0418, 0.0000, 0.3022, 0.0000, 0.1732, 0.0000, -0.22, 2.2, 0.32, 0.68]
SolutionDataG3 = [-0.2701, 0.0000, 0.9058, 0.0000, 0.4649, 0.0000, -1.04, 2.32, 0.91, 0.2]

SolutionDataSet = pd.DataFrame(
    [SolutionDataG1, SolutionDataG2, SolutionDataG3],
   index=["G1", "G2", "G3"],
   columns=["So1x", "So1y", "So2x", "So2y", "So14x", "So14y", "Lx", "Ly", "Rx", "Ry"]
)

print("")
print("########## Solution (SolutionDataSet) ##########")
print(SolutionDataSet)
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

############################################################################################################
# () ########## Vector (Rotated Axis to the Circle) (VectorDataSet) ##########
############################################################################################################
V1 = [0.00, 0.00, 0.00, 4.38, 0.00, 4.38, 0.00, 4.50]
V2 = [0.28, 3.99, 0.31, 4.36, 0.03, 0.37, 0.32, 4.49]
V3 = [0.80, 3.92, 0.87, 4.29, 0.07, 0.37, 0.90, 4.41]
V4 = [2.25, 3.31, 2.46, 3.62, 0.21, 0.31, 2.53, 3.72]
V5 = [0.00, 0.00, 4.38, 0.00, 4.38, 0.00, 4.50, 0.00]

VectorDataSet = pd.DataFrame(
    [V1, V2, V3, V4, V5],
   index=["V1", "V2", "V3", "V4", "V5"],
   columns=["Sx", "Sy", "Ex", "Ey", "deltaX", "deltaY", "CircleX", "CircleY"]
)

print("")
print("########## Vector (Rotated Axis to the Circle) (VectorDataSet) ##########")
print(VectorDataSet)

img_01 = plt.imread("Guo_2020_Fig1B.png")
img_02 = plt.imread("Sugawa_2018_Fig1.png")
########################################################################################################################
# Figures
########################################################################################################################
Figure_object = plt.figure(1, figsize=(11.69, 8.27), dpi=400, edgecolor="black", linewidth=0.5)
#plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
plt.subplots_adjust(left=0.125 ,bottom=0.11 ,right=0.9 ,top=0.88 ,wspace=0.2 ,hspace=0.2)

plt.figtext(0.0440, 0.9700, "a", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")
#plt.figtext(0.3750, 0.9700, "b", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")

plt.figtext(0.4100, 0.9250, "b", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")
plt.figtext(0.6700, 0.9250, "c", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")

plt.figtext(0.0440, 0.6000, "d", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")
plt.figtext(0.5000, 0.6000, "e", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")
plt.figtext(0.7500, 0.6000, "f", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")
plt.figtext(0.5000, 0.3000, "g", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")
plt.figtext(0.7560, 0.3000, "h", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")

gs_master = matplotlib.gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[1.0, 1.6])
gs_1 = matplotlib.gridspec.GridSpecFromSubplotSpec(
    nrows=1, ncols=2, subplot_spec=gs_master[0],  width_ratios=[1.0, 2.0])


gs_2 = matplotlib.gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=2, subplot_spec=gs_master[1], width_ratios=[1.0, 1.0])
gs_2_1 = matplotlib.gridspec.GridSpecFromSubplotSpec(nrows=2, ncols=1, subplot_spec=gs_2[1])
gs_2_1_1 = matplotlib.gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=2, subplot_spec=gs_2_1[0])
gs_2_1_2 = matplotlib.gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=2, subplot_spec=gs_2_1[1])

#Axes_obj_01 = Figure_object.add_subplot(gs_1[0], xticks=[], yticks=[])
#Axes_Outer = Figure_object.add_subplot(gs_1[1])
#Axes_obj_02 = mpl_inset.inset_axes(Axes_Outer, width="72.5%", height="72.5%", bbox_to_anchor=(0.0, 0.0, 0.5, 1.0), bbox_transform=Axes_Outer.transAxes, loc="center", borderpad=1)
#Axes_obj_03 = mpl_inset.inset_axes(Axes_Outer, width="60.0%", height="60.0%", bbox_to_anchor=(0.5, 0.0-0.025, 0.5, 1.0+0.11), bbox_transform=Axes_Outer.transAxes, loc="center", borderpad=1)
#Axes_obj_04 = Figure_object.add_subplot(gs_2[0], xticks=[0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5], yticks=[0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
#Axes_obj_05 = Figure_object.add_subplot(gs_2_1_1[0], xticks=[0, 1.0, 2.0, 3.0, 4.0], yticks=[0, 1.0, 2.0, 3.0, 4.0])
#Axes_obj_06 = Figure_object.add_subplot(gs_2_1_1[1], xticks=[-0.2, -0.1, 0.0, 0.1, 0.2], yticks=[-0.5, 0.0, 0.5, 1.0])
#Axes_obj_07 = Figure_object.add_subplot(gs_2_1_2[0], xticks=[-0.4, -0.2, -0.0, 0.2, 0.4, 0.6], yticks=[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
#Axes_obj_08 = Figure_object.add_subplot(gs_2_1_2[1], xticks=[-1.0, 0.0, 1.0, 2.0], yticks=[-1.0, 0.0, 1.0, 2.0])

Axes_obj_01 = plt.axes([0.0657384+0.014, 0.644572, 0.288834, 0.298807], xticks=[], yticks=[])
Axes_Outer = plt.axes([0.408069-0.033+0.014, 0.631884, 0.577669, 0.324182])
Axes_obj_02 = mpl_inset.inset_axes(Axes_Outer, width="72.5%", height="72.5%", bbox_to_anchor=(0.0, 0.0, 0.5, 1.0), bbox_transform=Axes_Outer.transAxes, loc="center", borderpad=1)
Axes_obj_03 = mpl_inset.inset_axes(Axes_Outer, width="60.0%", height="60.0%", bbox_to_anchor=(0.5, 0.0-0.025, 0.5, 1.0+0.11), bbox_transform=Axes_Outer.transAxes, loc="center", borderpad=1)
Axes_obj_04 = plt.axes([0.0657384+0.014, 0.0692597, 0.366944, 0.518691], xticks=[0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5], yticks=[0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
Axes_obj_05 = plt.axes([0.552487-0.033*1.5+0.014, 0.341454+0.01, 0.196933, 0.246497*(9.5/10)], xticks=[0, 1.0, 2.0, 3.0, 4.0], yticks=[0, 1.0, 2.0, 3.0, 4.0])
Axes_obj_06 = plt.axes([0.788806-0.033+0.014, 0.341454+0.01, 0.196933, 0.246497*(9.5/10)], xticks=[-0.2, -0.1, 0.0, 0.1, 0.2], yticks=[-0.5, 0.0, 0.5, 1.0])
Axes_obj_07 = plt.axes([0.552487-0.033*1.5+0.014, 0.0692597*(2/3), 0.196933, 0.246497*(9.5/10)], xticks=[-0.4, -0.2, -0.0, 0.2, 0.4, 0.6], yticks=[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
Axes_obj_08 = plt.axes([0.788806-0.033+0.014, 0.0692597*(2/3), 0.196933, 0.246497*(9.5/10)], xticks=[-1.0, 0.0, 1.0, 2.0], yticks=[-1.0, 0.0, 1.0, 2.0])

############################################################################################################
# (a) Guo et al. 2020
############################################################################################################
Axes_obj_01.set_title("Guo et al. 2020 (Fig. 1B)", size=11.4230, fontweight="normal")
#Axes_obj_01.yaxis.set_visible(False)
#Axes_obj_01.xaxis.set_visible(False)
Axes_obj_01.tick_params(length=0.0)
Axes_obj_01.set_xticklabels([])
Axes_obj_01.set_yticklabels([])
#Axes_obj_01.set_ylabel("Plasma NT-proBNP, μg/mL")
#Axes_obj_01.set_xlabel("Plasma TnT, ng/mL")
Axes_obj_01.imshow(img_01)

############################################################################################################
# (b) & (c)
############################################################################################################
Axes_Outer.set_title("Parabola in the Other Research (Sugawa et al. 2018 Figure 1)", size=11.4230,  fontweight="normal")
Axes_Outer.tick_params(length=0.0)
Axes_Outer.set_xticklabels([])
Axes_Outer.set_yticklabels([])

############################################################################################################
# (b) Sugawa et al. 2018
############################################################################################################
Axes_obj_02.set_title("Sugawa et al. 2018", size=11.4230,  fontweight="normal")
#Axes_obj_02.yaxis.set_visible(False)
#Axes_obj_02.xaxis.set_visible(False)
Axes_obj_02.tick_params(length=0.0)
Axes_obj_02.set_xticklabels([])
Axes_obj_02.set_yticklabels([])
Axes_obj_02.set_ylabel("Cardiac Tropnin I (pg/mL)")
Axes_obj_02.set_xlabel("BNP (pg/mL)")
Axes_obj_02.imshow(img_02)
############################################################################################################
# (c) Add parabola Sugawa et al. 2018
############################################################################################################
Axes_obj_03.set_title("Parabola in b", size=11.4230,  fontweight="normal")
Axes_obj_03.set_ylabel("Cardiac Troponin I (pg/mL)")
Axes_obj_03.set_xlabel("BNP (pg/mL)")
Axes_obj_03.set_xlim(-5.0,  145)
Axes_obj_03.set_ylim(-5.0, 225)
Axes_obj_03.set_xticks([0, 40, 80, 120])
Axes_obj_03.set_yticks([0, 50, 100, 150, 200, 250])
Axes_obj_03.set_xticklabels=[0, 40, 80, 120]
Axes_obj_03.set_yticklabels=[0, 50, 100, 150, 200, 250]
#Axes_obj_03.set_xticks([0, 20, 40, 60, 80, 100, 120, 140])
#Axes_obj_03.set_yticks([0, 50, 100, 150, 200, 250])
#Axes_obj_03.set_xticklabels=[0, 20, 40, 60, 80, 100, 120, 140]
#Axes_obj_03.set_yticklabels=[0, 50, 100, 150, 200, 250]

Axes_obj_03.hlines(y=26.2, xmin=0.0, xmax=140, color="red", linestyle="dashed", linewidth=1.0)

Axes_obj_03.plot([0.0, 125.460040327514], [0.090*0.0+2.386, 0.090*125.460040327514+2.386],
                 color="black", linestyle="solid", linewidth=1.0)

Axes_obj_03.scatter(PatientDataSet_S["X(BNP)"], PatientDataSet_S["Y(TnT)"],
                    color="blue", marker="o", s=12, zorder=10)

B_1 = ParabolaRDataSet.at["Sugawa", "y^2"]
B_2 = ParabolaRDataSet.at["Sugawa", "x*y"]
B_3 = ParabolaRDataSet.at["Sugawa", "y"]
B_4 = ParabolaRDataSet.at["Sugawa", "x^2"]
B_5 = ParabolaRDataSet.at["Sugawa", "x"]
B_6 = ParabolaRDataSet.at["Sugawa", "c"]
x1 = np.arange(0, ParabolaRDataSet.at["Sugawa", "Sx"], 1.0*(10**(-2)))
x2 = np.arange(0, ParabolaRDataSet.at["Sugawa", "Ex"], 1.0*(10**(-2)))
y1 = (np.sqrt((B_2**2-4*B_1*B_4)*x1**2+(2*B_2*B_3-4*B_1*B_5)*x1-4*B_1*B_6+B_3**2)-B_2*x1-B_3)/(2*B_1)
y2 = -(np.sqrt((B_2**2-4*B_1*B_4)*x2**2+(2*B_2*B_3-4*B_1*B_5)*x2-4*B_1*B_6+B_3**2)+B_2*x2+B_3)/(2*B_1)
Axes_obj_03.plot(x1, y1, color="red", linestyle="dashed", linewidth=1.0)
Axes_obj_03.plot(x2, y2, color="red", linestyle="dashed", linewidth=1.0)

Axes_obj_03.quiver(0.0, 26.2, max(PatientDataSet_S["X(BNP)"]), 132.55000000000004,
                   scale_units='xy', angles='xy', scale=1, color="red", width=0.008)

############################################################################################################
# (d) Re-analysis
############################################################################################################
Axes_obj_04.set_title("Re-analysis Guo et al. 2020 (Fig. 1B)", size=11.4230, fontweight="normal")
Axes_obj_04.set_ylabel("Plasma NT-proBNP,  ×10 μg/mL", fontweight="normal")
Axes_obj_04.set_xlabel("Plasma Troponin T,  ng/mL", fontweight="normal")
Axes_obj_04.set_xlim(-0.25, 4.75)
Axes_obj_04.set_ylim(-0.25, 4.75)
Axes_obj_04.set_xticks = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
Axes_obj_04.set_yticks = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
Axes_obj_04.set_aspect(aspect=1)

for i in range(0, 10):
    Axes_obj_04.vlines(x=0.0+0.5*i, ymin=0.0, ymax=4.5, color="gray", linestyle="dotted", linewidth=0.5)
    Axes_obj_04.hlines(y=0.0+0.5*i, xmin=0.0, xmax=4.5, color="gray", linestyle="dotted", linewidth=0.5)

x = np.arange(0, 4.38, 1.0*(10**(-3)))
y = CenterOfGravityDataSet.at["G1", "a1"]*x + CenterOfGravityDataSet.at["G1", "b1"]
Axes_obj_04.plot(x, y, color="black", linestyle="solid", linewidth=0.5, zorder=0)

x = np.arange(0, 4.38, 1.0*(10**(-3)))
y = CenterOfGravityDataSet.at["G2", "a1"]*x + CenterOfGravityDataSet.at["G2", "b1"]
Axes_obj_04.plot(x, y, color="black", linestyle="solid", linewidth=0.5, zorder=0)

x = np.arange(0, 4.38, 1.0*(10**(-3)))
y = CenterOfGravityDataSet.at["G3", "a1"]*x + CenterOfGravityDataSet.at["G3", "b1"]
Axes_obj_04.plot(x, y, color="black", linestyle="solid", linewidth=0.5, zorder=0)

x = np.arange(0, max(PatientDataSet["Plasma TnT"]), 1.0*(10**(-3)))
y = CenterOfGravityDataSet.at["Guo et al.", "a1"]*x + CenterOfGravityDataSet.at["Guo et al.", "b1"]
Axes_obj_04.plot(x, y, color="black", linestyle="solid", linewidth=1.5)

Axes_obj_04.quiver(
    VectorDataSet.at["V1", "Sx"], VectorDataSet.at["V1", "Sy"],
    VectorDataSet.at["V1", "deltaX"], VectorDataSet.at["V1", "deltaY"],
    scale_units='xy', angles='xy', scale=1, color="black", width=0.005)
Axes_obj_04.quiver(
    VectorDataSet.at["V2", "Sx"], VectorDataSet.at["V2", "Sy"],
    VectorDataSet.at["V2", "deltaX"], VectorDataSet.at["V2", "deltaY"],
    scale_units='xy', angles='xy', scale=1, color="black", width=0.005)
Axes_obj_04.quiver(
    VectorDataSet.at["V3", "Sx"], VectorDataSet.at["V3", "Sy"],
    VectorDataSet.at["V3", "deltaX"], VectorDataSet.at["V3", "deltaY"],
    scale_units='xy', angles='xy', scale=1, color="black", width=0.005)
Axes_obj_04.quiver(
    VectorDataSet.at["V4", "Sx"], VectorDataSet.at["V4", "Sy"],
    VectorDataSet.at["V4", "deltaX"], VectorDataSet.at["V4", "deltaY"],
    scale_units='xy', angles='xy', scale=1, color="black", width=0.005)
Axes_obj_04.quiver(
    VectorDataSet.at["V5", "Sx"], VectorDataSet.at["V5", "Sy"],
    VectorDataSet.at["V5", "deltaX"], VectorDataSet.at["V5", "deltaY"],
    scale_units='xy', angles='xy', scale=1, color="black", width=0.005)

Axes_obj_04.plot(np.arange(0, 4.38, 1.0*(10**(-5))), np.sqrt(4.38**2-(np.arange(0, 4.38, 1.0*(10**(-5))))**2),
                 color="black", linestyle="solid", linewidth=0.5)

Axes_obj_04.text(-0.125, -0.125,
                 "O", color="black", size=11.4230, horizontalalignment="center", verticalalignment="center")
Axes_obj_04.text(VectorDataSet.at["V1", "CircleX"]-0.125, VectorDataSet.at["V1", "CircleY"],
                 "Y", color="black", size=11.4230, horizontalalignment="center", verticalalignment="center")
Axes_obj_04.text(VectorDataSet.at["V5", "CircleX"], VectorDataSet.at["V5", "CircleY"],
                 "X", color="black", size=11.4230, horizontalalignment="center", verticalalignment="center")

Axes_obj_04.annotate(text="G2 y-axis", xy=(VectorDataSet.at["V2", "CircleX"]+0.15, VectorDataSet.at["V2", "CircleY"]),
                     ha='center', va='center', size=10,
                     bbox=dict(boxstyle='square', edgecolor='red', fc='white'))

Axes_obj_04.annotate(text="G1 y-axis", xy=(VectorDataSet.at["V3", "CircleX"]+0.5, VectorDataSet.at["V3", "CircleY"]),
                     ha='center', va='center', size=10,
                     bbox=dict(boxstyle='square', edgecolor='red', fc='white'))

Axes_obj_04.annotate(text="G3 y-axis", xy=(VectorDataSet.at["V4", "CircleX"]+0.3, VectorDataSet.at["V4", "CircleY"]+0.05),
                     ha='center', va='center', size=10,
                     bbox=dict(boxstyle='square', edgecolor='red', fc='white'))
Axes_obj_04.annotate(text="Guo et al.\nRegression Line", xy=(3.0, 2.8),
                     ha='center', va='center', size=10,
                     bbox=dict(boxstyle='square', edgecolor='red', fc='white'))

Axes_obj_04.quiver(
    AxisDataSet.at["G1_P1", "Sx"], AxisDataSet.at["G1_P1", "Sy"],
    AxisDataSet.at["G1_P1", "deltaX"], AxisDataSet.at["G1_P1", "deltaY"],
    scale_units='xy', angles='xy', scale=1, color="red", width=0.004)
Axes_obj_04.quiver(
    AxisDataSet.at["G1_P2", "Sx"], AxisDataSet.at["G1_P2", "Sy"],
    AxisDataSet.at["G1_P2", "deltaX"], AxisDataSet.at["G1_P2", "deltaY"],
    scale_units='xy', angles='xy', scale=1, color="red", width=0.004)

Axes_obj_04.quiver(
    AxisDataSet.at["G2_P1", "Sx"], AxisDataSet.at["G2_P1", "Sy"],
    AxisDataSet.at["G2_P1", "deltaX"], AxisDataSet.at["G2_P1", "deltaY"],
    scale_units='xy', angles='xy', scale=1, color="red", width=0.004)
Axes_obj_04.quiver(
    AxisDataSet.at["G2_P2", "Sx"], AxisDataSet.at["G2_P2", "Sy"],
    AxisDataSet.at["G2_P2", "deltaX"], AxisDataSet.at["G2_P2", "deltaY"],
    scale_units='xy', angles='xy', scale=1, color="red", width=0.004)

Axes_obj_04.quiver(
    AxisDataSet.at["G3_P1", "Sx"], AxisDataSet.at["G3_P1", "Sy"],
    AxisDataSet.at["G3_P1", "deltaX"], AxisDataSet.at["G3_P1", "deltaY"],
    scale_units='xy', angles='xy', scale=1, color="red", width=0.004)
Axes_obj_04.quiver(
    AxisDataSet.at["G3_P2", "Sx"], AxisDataSet.at["G3_P2", "Sy"],
    AxisDataSet.at["G3_P2", "deltaX"], AxisDataSet.at["G3_P2", "deltaY"],
    scale_units='xy', angles='xy', scale=1, color="red", width=0.004)

Axes_obj_04.plot(
    PatientDataSet[PatientDataSet["Frag1"] == 1]["FlameLeftX"],
    PatientDataSet[PatientDataSet["Frag1"] == 1]["FlameLeftY"],
    color="red", linestyle="solid", linewidth=0.5)
Axes_obj_04.plot(
    PatientDataSet[PatientDataSet["Frag1"] == 1]["FlameRightX"],
    PatientDataSet[PatientDataSet["Frag1"] == 1]["FlameRightY"],
    color="red", linestyle="solid", linewidth=0.5)
Axes_obj_04.plot(
    PatientDataSet[PatientDataSet["Frag1"] == 1]["FlameUpperX"],
    PatientDataSet[PatientDataSet["Frag1"] == 1]["FlameUpperY"],
    color="red", linestyle="solid", linewidth=0.5)
Axes_obj_04.plot(
    PatientDataSet[PatientDataSet["Frag1"] == 1]["FlameLowewX"],
    PatientDataSet[PatientDataSet["Frag1"] == 1]["FlameLowewY"],
    color="red", linestyle="solid", linewidth=0.5)

Axes_obj_04.plot(
    PatientDataSet[PatientDataSet["Frag1"] == 2]["FlameLeftX"],
    PatientDataSet[PatientDataSet["Frag1"] == 2]["FlameLeftY"],
    color="red", linestyle="solid", linewidth=0.5)
Axes_obj_04.plot(
    PatientDataSet[PatientDataSet["Frag1"] == 2]["FlameRightX"],
    PatientDataSet[PatientDataSet["Frag1"] == 2]["FlameRightY"],
    color="red", linestyle="solid", linewidth=0.5)
Axes_obj_04.plot(
    PatientDataSet[PatientDataSet["Frag1"] == 2]["FlameUpperX"],
    PatientDataSet[PatientDataSet["Frag1"] == 2]["FlameUpperY"],
    color="red", linestyle="solid", linewidth=0.5)
Axes_obj_04.plot(
    PatientDataSet[PatientDataSet["Frag1"] == 2]["FlameLowewX"],
    PatientDataSet[PatientDataSet["Frag1"] == 2]["FlameLowewY"],
    color="red", linestyle="solid", linewidth=0.5)

Axes_obj_04.plot(
    PatientDataSet[(PatientDataSet["Frag1"] == 3) | (PatientDataSet["Frag1"] == 4)]["FlameLeftX"],
    PatientDataSet[(PatientDataSet["Frag1"] == 3) | (PatientDataSet["Frag1"] == 4)]["FlameLeftY"],
    color="red", linestyle="solid", linewidth=0.5)
Axes_obj_04.plot(
    PatientDataSet[(PatientDataSet["Frag1"] == 3) | (PatientDataSet["Frag1"] == 4)]["FlameRightX"],
    PatientDataSet[(PatientDataSet["Frag1"] == 3) | (PatientDataSet["Frag1"] == 4)]["FlameRightY"],
    color="red", linestyle="solid", linewidth=0.5)
Axes_obj_04.plot(
    PatientDataSet[(PatientDataSet["Frag1"] == 3) | (PatientDataSet["Frag1"] == 4)]["FlameUpperX"],
    PatientDataSet[(PatientDataSet["Frag1"] == 3) | (PatientDataSet["Frag1"] == 4)]["FlameUpperY"],
    color="red", linestyle="solid", linewidth=0.5)
Axes_obj_04.plot(
    PatientDataSet[(PatientDataSet["Frag1"] == 3) | (PatientDataSet["Frag1"] == 4)]["FlameLowewX"],
    PatientDataSet[(PatientDataSet["Frag1"] == 3) | (PatientDataSet["Frag1"] == 4)]["FlameLowewY"],
    color="red", linestyle="solid", linewidth=0.5)

TempData1 = PatientDataSet[PatientDataSet["Frag1"] == 1]
TempData2 = PatientDataSet[PatientDataSet["Frag1"] == 2]
TempData3 = PatientDataSet[(PatientDataSet["Frag1"] == 3) | (PatientDataSet["Frag1"] == 4)]

#for i in range(0, len(TempData1)):
#    Axes_obj_04.plot([CenterOfGravityDataSet.at["G1", "MeanX"], TempData1.iat[i, 2]],
#                     [CenterOfGravityDataSet.at["G1", "MeanY"], TempData1.iat[i, 3]],
#                     color="red", linestyle="dashed", linewidth=0.5)
#for i in range(0, len(TempData2)):
#    Axes_obj_04.plot([CenterOfGravityDataSet.at["G2", "MeanX"], TempData2.iat[i, 2]],
#                     [CenterOfGravityDataSet.at["G2", "MeanY"], TempData2.iat[i, 3]],
#                     color="red", linestyle="dashed", linewidth=0.5)
#for i in range(0, len(TempData3)):
#    Axes_obj_04.plot([CenterOfGravityDataSet.at["G3", "MeanX"], TempData3.iat[i, 2]],
#                     [CenterOfGravityDataSet.at["G3", "MeanY"], TempData3.iat[i, 3]],
#                     color="red", linestyle="dashed", linewidth=0.5)

Axes_obj_04.scatter(
    [CenterOfGravityDataSet.at["G1", "MeanX"],
     CenterOfGravityDataSet.at["G2", "MeanX"],
     CenterOfGravityDataSet.at["G3", "MeanX"]],
    [CenterOfGravityDataSet.at["G1", "MeanY"],
     CenterOfGravityDataSet.at["G2", "MeanY"],
     CenterOfGravityDataSet.at["G3", "MeanY"]],
        color="red", marker="o", s=18, zorder=9)

Axes_obj_04.scatter(PatientDataSet["Plasma TnT"], PatientDataSet["Plasma NT-proBNP"],
                    color="black", marker="s", s=20, zorder=9)

##################################################
B_1 = ParabolaRDataSet.at["G1", "y^2"]
B_2 = ParabolaRDataSet.at["G1", "x*y"]
B_3 = ParabolaRDataSet.at["G1", "y"]
B_4 = ParabolaRDataSet.at["G1", "x^2"]
B_5 = ParabolaRDataSet.at["G1", "x"]
B_6 = ParabolaRDataSet.at["G1", "c"]
x1 = np.arange(0.041750, ParabolaRDataSet.at["G1", "Sx"], 1.0*(10**(-3)))
x2 = np.arange(0.041750, ParabolaRDataSet.at["G1", "Ex"], 1.0*(10**(-3)))
y1 = (np.sqrt((B_2**2-4*B_1*B_4)*x1**2+(2*B_2*B_3-4*B_1*B_5)*x1-4*B_1*B_6+B_3**2)-B_2*x1-B_3)/(2*B_1)
y2 = -(np.sqrt((B_2**2-4*B_1*B_4)*x2**2+(2*B_2*B_3-4*B_1*B_5)*x2-4*B_1*B_6+B_3**2)+B_2*x2+B_3)/(2*B_1)
Axes_obj_04.plot(x1, y1, color="dodgerblue", linestyle="solid", linewidth=1.6318, zorder=10)
Axes_obj_04.plot(x2, y2, color="dodgerblue", linestyle="solid", linewidth=1.6318, zorder=10)

B_1 = ParabolaRDataSet.at["G2", "y^2"]
B_2 = ParabolaRDataSet.at["G2", "x*y"]
B_3 = ParabolaRDataSet.at["G2", "y"]
B_4 = ParabolaRDataSet.at["G2", "x^2"]
B_5 = ParabolaRDataSet.at["G2", "x"]
B_6 = ParabolaRDataSet.at["G2", "c"]
x1 = np.arange(0.127630, ParabolaRDataSet.at["G2", "Sx"], 1.0*(10**(-3)))
x2 = np.arange(0.127630, ParabolaRDataSet.at["G2", "Ex"], 1.0*(10**(-3)))
y1 = (np.sqrt((B_2**2-4*B_1*B_4)*x1**2+(2*B_2*B_3-4*B_1*B_5)*x1-4*B_1*B_6+B_3**2)-B_2*x1-B_3)/(2*B_1)
y2 = -(np.sqrt((B_2**2-4*B_1*B_4)*x2**2+(2*B_2*B_3-4*B_1*B_5)*x2-4*B_1*B_6+B_3**2)+B_2*x2+B_3)/(2*B_1)
Axes_obj_04.plot(x1, y1, color="dodgerblue", linestyle="solid", linewidth=1.6318)
Axes_obj_04.plot(x2, y2, color="dodgerblue", linestyle="solid", linewidth=1.6318)

B_1 = ParabolaRDataSet.at["G3", "y^2"]
B_2 = ParabolaRDataSet.at["G3", "x*y"]
B_3 = ParabolaRDataSet.at["G3", "y"]
B_4 = ParabolaRDataSet.at["G3", "x^2"]
B_5 = ParabolaRDataSet.at["G3", "x"]
B_6 = ParabolaRDataSet.at["G3", "c"]
x1 = np.arange(0.898070, 1.15, 1.0*(10**(-3)))
x2 = np.arange(0.898070, 2.80, 1.0*(10**(-3)))
y1 = (np.sqrt((B_2**2-4*B_1*B_4)*x1**2+(2*B_2*B_3-4*B_1*B_5)*x1-4*B_1*B_6+B_3**2)-B_2*x1-B_3)/(2*B_1)
y2 = -(np.sqrt((B_2**2-4*B_1*B_4)*x2**2+(2*B_2*B_3-4*B_1*B_5)*x2-4*B_1*B_6+B_3**2)+B_2*x2+B_3)/(2*B_1)
Axes_obj_04.plot(x1, y1, color="dodgerblue", linestyle="solid", linewidth=1.6318)
Axes_obj_04.plot(x2, y2, color="dodgerblue", linestyle="solid", linewidth=1.6318)

############################################################################################################
# (e) Transpose Axis
############################################################################################################
Axes_obj_05.set_title("Transpose Axis", size=11.4230,  fontweight="normal")
#Axes_obj_05.set_xlabel("NT-proBNP, ×10 μg/mL")
Axes_obj_05.set_ylabel("Plasma Troponin T, ng/mL")
Axes_obj_05.set_xlim(-0.25, 4.75)
Axes_obj_05.set_ylim(-0.25, 4.75)
#Axes_obj_05.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
Axes_obj_05.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
Axes_obj_05.set_xticklabels([0.0, 1.0, 2.0, 3.0, 4.0])
#Axes_obj_05.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
Axes_obj_05.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
Axes_obj_05.set_yticklabels([0.0, 1.0, 2.0, 3.0, 4.0])

for i in range(0, 10):
    Axes_obj_05.vlines(x=0.0+0.5*i, ymin=0.0, ymax=4.5, color="gray", linestyle="dotted", linewidth=0.5)
    Axes_obj_05.hlines(y=0.0+0.5*i, xmin=0.0, xmax=4.5, color="gray", linestyle="dotted", linewidth=0.5)

Axes_obj_05.scatter(PatientDataSet["Plasma NT-proBNP"], PatientDataSet["Plasma TnT"],
                    color="black", marker="s", s=12, zorder=9)

B_1 = ParabolaRDataSet.at["G1", "y^2"]
B_2 = ParabolaRDataSet.at["G1", "x*y"]
B_3 = ParabolaRDataSet.at["G1", "y"]
B_4 = ParabolaRDataSet.at["G1", "x^2"]
B_5 = ParabolaRDataSet.at["G1", "x"]
B_6 = ParabolaRDataSet.at["G1", "c"]
x1 = np.arange(0.041750, ParabolaRDataSet.at["G1", "Sx"], 1.0*(10**(-3)))
x2 = np.arange(0.041750, ParabolaRDataSet.at["G1", "Ex"], 1.0*(10**(-3)))
y1 = (np.sqrt((B_2**2-4*B_1*B_4)*x1**2+(2*B_2*B_3-4*B_1*B_5)*x1-4*B_1*B_6+B_3**2)-B_2*x1-B_3)/(2*B_1)
y2 = -(np.sqrt((B_2**2-4*B_1*B_4)*x2**2+(2*B_2*B_3-4*B_1*B_5)*x2-4*B_1*B_6+B_3**2)+B_2*x2+B_3)/(2*B_1)
Axes_obj_05.plot(y1, x1, color="dodgerblue", linestyle="solid", linewidth=1.6318, zorder=9)
Axes_obj_05.plot(y2, x2, color="dodgerblue", linestyle="solid", linewidth=1.6318, zorder=9)

B_1 = ParabolaRDataSet.at["G2", "y^2"]
B_2 = ParabolaRDataSet.at["G2", "x*y"]
B_3 = ParabolaRDataSet.at["G2", "y"]
B_4 = ParabolaRDataSet.at["G2", "x^2"]
B_5 = ParabolaRDataSet.at["G2", "x"]
B_6 = ParabolaRDataSet.at["G2", "c"]
x1 = np.arange(0.127630, ParabolaRDataSet.at["G2", "Sx"], 1.0*(10**(-3)))
x2 = np.arange(0.127630, ParabolaRDataSet.at["G2", "Ex"], 1.0*(10**(-3)))
y1 = (np.sqrt((B_2**2-4*B_1*B_4)*x1**2+(2*B_2*B_3-4*B_1*B_5)*x1-4*B_1*B_6+B_3**2)-B_2*x1-B_3)/(2*B_1)
y2 = -(np.sqrt((B_2**2-4*B_1*B_4)*x2**2+(2*B_2*B_3-4*B_1*B_5)*x2-4*B_1*B_6+B_3**2)+B_2*x2+B_3)/(2*B_1)
Axes_obj_05.plot(y1, x1, color="dodgerblue", linestyle="solid", linewidth=1.6318)
Axes_obj_05.plot(y2, x2, color="dodgerblue", linestyle="solid", linewidth=1.6318)

B_1 = ParabolaRDataSet.at["G3", "y^2"]
B_2 = ParabolaRDataSet.at["G3", "x*y"]
B_3 = ParabolaRDataSet.at["G3", "y"]
B_4 = ParabolaRDataSet.at["G3", "x^2"]
B_5 = ParabolaRDataSet.at["G3", "x"]
B_6 = ParabolaRDataSet.at["G3", "c"]
x1 = np.arange(0.898070, 1.15, 1.0*(10**(-3)))
x2 = np.arange(0.898070, 2.80, 1.0*(10**(-3)))
y1 = (np.sqrt((B_2**2-4*B_1*B_4)*x1**2+(2*B_2*B_3-4*B_1*B_5)*x1-4*B_1*B_6+B_3**2)-B_2*x1-B_3)/(2*B_1)
y2 = -(np.sqrt((B_2**2-4*B_1*B_4)*x2**2+(2*B_2*B_3-4*B_1*B_5)*x2-4*B_1*B_6+B_3**2)+B_2*x2+B_3)/(2*B_1)
Axes_obj_05.plot(y1, x1, color="dodgerblue", linestyle="solid", linewidth=1.6318)
Axes_obj_05.plot(y2, x2, color="dodgerblue", linestyle="solid", linewidth=1.6318)

############################################################################################################
# (f) Group 1
############################################################################################################
Axes_obj_06.set_title("Group 1", size=11.4230,  fontweight="normal")
Axes_obj_06.set_ylabel("")
Axes_obj_06.set_xlabel("")
Axes_obj_06.set_xlim(-0.275, 0.25)
Axes_obj_06.set_ylim(-0.5, 1.1)
Axes_obj_06.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
#Axes_obj_06.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
Axes_obj_06.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
Axes_obj_06.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))

Axes_obj_06.vlines(x=0.0, ymin=-0.5, ymax=1.1, color="black", linestyle="solid", linewidth=0.5)
Axes_obj_06.hlines(y=0.0, xmin=-0.75, xmax=0.85, color="black", linestyle="solid", linewidth=0.5)

Axes_obj_06.scatter(PatientDataSet[PatientDataSet["Frag1"] == 1].Axis1,
                    PatientDataSet[PatientDataSet["Frag1"] == 1].Axis2,
                    color="black", marker="s", s=12, zorder=10)

a = ParabolaDataSet.at["G1_U", "a"]
k = ParabolaDataSet.at["G1_U", "Coef."]
Px = ParabolaDataSet.at["G1_U", "Px"]
Py = ParabolaDataSet.at["G1_U", "Py"]
x = np.arange(SolutionDataSet.at["G1", "Lx"]-0.075, SolutionDataSet.at["G1", "Rx"]+0.05, 1.0*(10**(-3)))
y = a*k*(x - Px)**2 + Py
Axes_obj_06.plot(x, y, color="red", linestyle="dashed", linewidth=1.0)

a = ParabolaDataSet.at["G1_R", "a"]
k = ParabolaDataSet.at["G1_U", "Coef."]
Px = ParabolaDataSet.at["G1_R", "Px"]
Py = ParabolaDataSet.at["G1_R", "Py"]
x = np.arange(SolutionDataSet.at["G1", "Lx"], SolutionDataSet.at["G1", "Rx"], 1.0*(10**(-3)))
y = a*k*(x - Px)**2 + Py
Axes_obj_06.plot(x, y, color="dodgerblue", linestyle="solid", linewidth=1.6318)

a = ParabolaDataSet.at["G1_L", "a"]
k = ParabolaDataSet.at["G1_U", "Coef."]
Px = ParabolaDataSet.at["G1_L", "Px"]
Py = ParabolaDataSet.at["G1_L", "Py"]
x = np.arange(SolutionDataSet.at["G1", "Lx"]-0.075, SolutionDataSet.at["G1", "Rx"]+0.05, 1.0*(10**(-3)))
y = a*k*(x - Px)**2 + Py
Axes_obj_06.plot(x, y, color="red", linestyle="dashed", linewidth=1.0)

Axes_obj_06.plot(
    [ParabolaDataSet.at["G1_L", "Px"], ParabolaDataSet.at["G1_U", "Px"]],
    [ParabolaDataSet.at["G1_L", "Py"], ParabolaDataSet.at["G1_U", "Py"]],
    color="red", linestyle="solid", linewidth=1.0)

############################################################################################################
# (g) Group 2
############################################################################################################
Axes_obj_07.set_title("Group 2", size=11.4230, fontweight="normal")
Axes_obj_07.set_ylabel("")
Axes_obj_07.set_xlabel("")
Axes_obj_07.set_xlim(-0.5, 0.625)
Axes_obj_07.set_ylim(-2.0, 3.75)
Axes_obj_07.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
#Axes_obj_07.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
#Axes_obj_07.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
Axes_obj_07.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
Axes_obj_07.set_yticklabels([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])

Axes_obj_07.vlines(x=0.0, ymin=-2.0, ymax=3.75, color="black", linestyle="solid", linewidth=0.5)
Axes_obj_07.hlines(y=0.0, xmin=-0.5, xmax=0.625, color="black", linestyle="solid", linewidth=0.5)

Axes_obj_07.scatter(PatientDataSet[PatientDataSet["Frag1"] == 2].Axis1,
                    PatientDataSet[PatientDataSet["Frag1"] == 2].Axis2,
                    color="black", marker="s", s=12, zorder=10)

a = ParabolaDataSet.at["G2_U", "a"]
k = ParabolaDataSet.at["G3_U", "Coef."]
Px = ParabolaDataSet.at["G2_U", "Px"]
Py = ParabolaDataSet.at["G2_U", "Py"]
x = np.arange(SolutionDataSet.at["G2", "Lx"]-0.15, SolutionDataSet.at["G2", "Rx"]+0.1, 1.0*(10**(-3)))
y = a*k*(x - Px)**2 + Py
Axes_obj_07.plot(x, y, color="red", linestyle="dashed", linewidth=1.0)

a = ParabolaDataSet.at["G2_R", "a"]
k = ParabolaDataSet.at["G3_U", "Coef."]
Px = ParabolaDataSet.at["G2_R", "Px"]
Py = ParabolaDataSet.at["G2_R", "Py"]
x = np.arange(SolutionDataSet.at["G2", "Lx"], SolutionDataSet.at["G2", "Rx"], 1.0*(10**(-3)))
y = a*k*(x - Px)**2 + Py
Axes_obj_07.plot(x, y, color="dodgerblue", linestyle="solid", linewidth=1.6318)

a = ParabolaDataSet.at["G2_L", "a"]
k = ParabolaDataSet.at["G3_U", "Coef."]
Px = ParabolaDataSet.at["G2_L", "Px"]
Py = ParabolaDataSet.at["G2_L", "Py"]
x = np.arange(SolutionDataSet.at["G2", "Lx"]-0.15, SolutionDataSet.at["G2", "Rx"]+0.1, 1.0*(10**(-3)))
y = a*k*(x - Px)**2 + Py
Axes_obj_07.plot(x, y, color="red", linestyle="dashed", linewidth=1.0)

Axes_obj_07.plot(
    [ParabolaDataSet.at["G2_L", "Px"], ParabolaDataSet.at["G2_U", "Px"]],
    [ParabolaDataSet.at["G2_L", "Py"], ParabolaDataSet.at["G2_U", "Py"]],
    color="red", linestyle="solid", linewidth=1.0)

############################################################################################################
# (h) Group 3
############################################################################################################
Axes_obj_08.set_title("Group 3",  size=11.4230, fontweight="normal")
Axes_obj_08.set_ylabel("")
Axes_obj_08.set_xlabel("")
Axes_obj_08.set_xlim(-1.8, 2.2)
Axes_obj_08.set_ylim(-1.25, 2.75)
#Axes_obj_08.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
Axes_obj_08.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
Axes_obj_08.set_xticklabels([-1.0, 0.0, 1.0, 2.0])
#Axes_obj_08.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
Axes_obj_08.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
Axes_obj_08.set_yticklabels([-1.0, 0.0, 1.0, 2.0])

Axes_obj_08.vlines(x=0.0, ymin=-1.25, ymax=2.75, color="black", linestyle="solid", linewidth=0.5)
Axes_obj_08.hlines(y=0.0, xmin=-1.8, xmax=2.2, color="black", linestyle="solid", linewidth=0.5)
Axes_obj_08.scatter(PatientDataSet[(PatientDataSet["Frag1"] == 3) | (PatientDataSet["Frag1"] == 4)].Axis1,
                    PatientDataSet[(PatientDataSet["Frag1"] == 3) | (PatientDataSet["Frag1"] == 4)].Axis2,
                    color="black", marker="s", s=12, zorder=10)

a = ParabolaDataSet.at["G3_U", "a"]
k = ParabolaDataSet.at["G3_U", "Coef."]
Px = ParabolaDataSet.at["G3_U", "Px"]
Py = ParabolaDataSet.at["G3_U", "Py"]
x = np.arange(SolutionDataSet.at["G3", "Lx"]-0.4, SolutionDataSet.at["G3", "Rx"]+0.4, 1.0*(10**(-3)))
y = a*k*(x - Px)**2 + Py
Axes_obj_08.plot(x, y, color="red", linestyle="dashed", linewidth=1.0)

a = ParabolaDataSet.at["G3_R", "a"]
k = ParabolaDataSet.at["G3_U", "Coef."]
Px = ParabolaDataSet.at["G3_R", "Px"]
Py = ParabolaDataSet.at["G3_R", "Py"]
x = np.arange(SolutionDataSet.at["G3", "Lx"], SolutionDataSet.at["G3", "Rx"], 1.0*(10**(-3)))
y = a*k*(x - Px)**2 + Py
Axes_obj_08.plot(x, y, color="dodgerblue", linestyle="solid", linewidth=1.6318)

a = ParabolaDataSet.at["G3_L", "a"]
k = ParabolaDataSet.at["G3_U", "Coef."]
Px = ParabolaDataSet.at["G3_L", "Px"]
Py = ParabolaDataSet.at["G3_L", "Py"]
x = np.arange(SolutionDataSet.at["G3", "Lx"]-0.4, SolutionDataSet.at["G3", "Rx"]+0.4, 1.0*(10**(-3)))
y = a*k*(x - Px)**2 + Py
Axes_obj_08.plot(x, y, color="red", linestyle="dashed", linewidth=1.0)

Axes_obj_08.plot(
    [ParabolaDataSet.at["G3_L", "Px"], ParabolaDataSet.at["G3_U", "Px"]],
    [ParabolaDataSet.at["G3_L", "Py"], ParabolaDataSet.at["G3_U", "Py"]],
    color="red", linestyle="solid", linewidth=1.0)

############################################################################################################

#Figure_object.tight_layout()
my_path = os.path.abspath("Figures")
plt.savefig(os.path.join(my_path, "01_Kimoto_et_al_(2023)_[Fig_S_14].png"))

from PIL import Image # py -m pip install pillow
img = Image.open(os.path.join(my_path, "01_Kimoto_et_al_(2023)_[Fig_S_14].png"))
img_resize = img.resize(size=(2866, 2016))
my_path2 = os.path.abspath("Figures/Fixed_Version/B6")
img_resize.save(os.path.join(my_path2, "01_Kimoto_et_al_(2023)_[Fig_S_14]_B6.png"))

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