########################################################################################################################
# Fig. 4. Unrecognized cyclic pattern of thrombosis onset after travel. (00_Kimoto_et_al_(2023)_[_Fig_04_].py)
########################################################################################################################
# Title of the manuscript (research):
# Novel Findings in Travel-related Thrombosis and COVID-19 Related Cardiovascular Data:
# Data Mining Method for Previously Published Data
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
# Contents
########################################################################################################################
# List of Import
# Making Data Sets
# Regression Analysis
#    Definition of Regression Function
#    Non-linear Regression Analysis (Weeks)
#    Non-linear Regression Analysis (Days)
# Coefficient Matrix (Data Set)
# Values for Curves

# Composite Function:                                       CF
# Monotonic Decrease Function (Exponential　Function):      MF
# Periodic Function (Trigonometric Function):               PF
# Difference Function:                                      DF
# Envelope Function (Upper):                                EF_U
# Envelope Function (Lower):                                EF_L

# Values for Curves (Week)
# Values for Curves (Day)
# Values1
# Values2
# Figures
#    "(a) Cannegieter et al. 2006 Figure 1"
#    "(b) Appropriate 2D Bar Chart"
#    "(c) Decreasing Curve"
#    "(d) Damped Wave Curve"
#    "(e) Added Damped Wave"
#    "(f) Added Damped Wave"
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
import matplotlib.ticker as ticker
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
AnalysisDataSet = pd.DataFrame({
 "Week": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
 "Day": [4+(7*0), 4+(7*1), 4+(7*2), 4+(7*3), 4+(7*4), 4+(7*5), 4+(7*6), 4+(7*7), 4+(7*8), 4+(7*9), 4+(7*10), 4+(7*11)],
 "Range": [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
 "Day Label": ["Day 4(1-7)   ", "Day 11(8-14) ", "Day 18(15-21)", "Day 25(22-28)", "Day 32(29-35)", "Day 39(36-42)",
               "Day 46(43-49)", "Day 53(50-56)", "Day 60(57-63)", "Day 67(64-70)", "Day 74(71-77)", "Day 81(78-84)"],
 "Cases": [68, 38, 24, 30, 30, 18, 10, 16, 13, 10, 11, 6]}
    , index=["Week 1", "Week 2", "Week 3", "Week 4", "Week 5", "Week 6", "Week 7", "Week 8", "Week 9", "Week 10",
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

def reg_func(parameter: ["n1", "l1", "n2", "l2", "a", "b"], x, y):
    n1 = parameter[0]
    l1 = parameter[1]
    n2 = parameter[2]
    l2 = parameter[3]
    a = parameter[4]
    b = parameter[5]
    _Output_ = y - ( n1 * np.exp(-l1 * (x-b)) + (n2 * np.exp(-l2 * (x-b))) * (np.sin(a * (x - b))) )
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
#print(Result_W)

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
#print(Result_D)

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
                            Decimal(str(CoefficientDataSet.iat[1, 1])).quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP),
                            Decimal(str(CoefficientDataSet.iat[1, 2])).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP),
                            Decimal(str(CoefficientDataSet.iat[1, 3])).quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP),
                            Decimal(str(CoefficientDataSet.iat[1, 4])).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP),
                            Decimal(str(CoefficientDataSet.iat[1, 5])).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)]

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
Y_Week = n1 * np.exp(-l1 * (X_Week-b)) + (n2 * np.exp(-l2 * (X_Week-b))) * (np.sin(a * (X_Week-b)))

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
Y_CF = n1 * np.exp(-l1 * (X_Day-b)) + (n2 * np.exp(-l2 * (X_Day-b))) * (np.sin(a * (X_Day-b)))
Y_MF = n1 * np.exp(-l1 * (X_Day-b)) + (0 * np.exp(-0 * (X_Day-b))) * (np.sin(0 * (X_Day-b)))
Y_PF = 0 * np.exp(-0 * (X_Day-b)) + (n2 * np.exp(-l2 * (X_Day-b))) * (np.sin(a * (X_Day-b)))
Y_DF = AnalysisDataSet["Cases"] - (n1 * np.exp(-l1 * (AnalysisDataSet["Day"]-b)))
Y_EF_U = 0 * np.exp(-0 * (X_Day-b)) + (n2 * np.exp(-l2 * (X_Day-b)))
Y_EF_L = 0 * np.exp(-0 * (X_Day-b)) - (n2 * np.exp(-l2 * (X_Day-b)))
Y_EF_U2 = n1 * np.exp(-l1 * (X_Day-b)) + (n2 * np.exp(-l2 * (X_Day-b)))
Y_EF_L2 = n1 * np.exp(-l1 * (X_Day-b)) - (n2 * np.exp(-l2 * (X_Day-b)))
Line_fill = 0.0*np.ones(1000)

Peaks_1 = [(b + (np.pi/2)/a),
           (b + (np.pi/2)/a) + 2*np.pi/a,
           (b + (np.pi/2)/a) + 2*np.pi/a + 2*np.pi/a,
           (b + (np.pi/2)/a) + 2*np.pi/a + 2*np.pi/a + 2*np.pi/a]

Peaks_1_round = [Decimal(str(Peaks_1[0])).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP),
                 Decimal(str(Peaks_1[1])).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP),
                 Decimal(str(Peaks_1[2])).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP),
                 Decimal(str(Peaks_1[3])).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)]

Values = [n1 * np.exp(-l1 * (30-b)) + (n2 * np.exp(-l2 * (30-b))) * (np.sin(a * (30-b))),
          n1 * np.exp(-l1 * (60-b)) + (n2 * np.exp(-l2 * (60-b))) * (np.sin(a * (60-b))),
          n1 * np.exp(-l1 * (90-b)) + (n2 * np.exp(-l2 * (90-b))) * (np.sin(a * (90-b))),
          n1 * np.exp(-l1 * (120-b)) + (n2 * np.exp(-l2 * (120-b))) * (np.sin(a * (120-b))),
          n1 * np.exp(-l1 * (150-b)) + (n2 * np.exp(-l2 * (150-b))) * (np.sin(a * (150-b))),
          n1 * np.exp(-l1 * (180-b)) + (n2 * np.exp(-l2 * (180-b))) * (np.sin(a * (180-b)))]

Values_round = [Decimal(str(Values[0])).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP),
                Decimal(str(Values[1])).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP),
                Decimal(str(Values[2])).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP),
                Decimal(str(Values[3])).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP),
                Decimal(str(Values[4])).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP),
                Decimal(str(Values[5])).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)]

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
# Values2
########################################################################################################################

Unit_A1 = (n1/l1)*np.exp(l1*b)
Unit_B1 = (n2/(l2**2 + a**2))*(l2*np.sin(-a*b)+a*np.cos(-a*b))
S = Unit_A1 + Unit_B1
S1 = (n1/l1)*np.exp(l1*b)-(n2/l2)*np.exp(l2*b)
S2 = S - S1
Percent_S2 = (S2/S)*100

List_S = [S1, S2, S, Percent_S2]

s = 56
Unit_B2_1 = l2*np.sin(a*(s-b))+a*np.cos(a*(s-b))
Unit_B2_2 = l2*np.sin(-a*b)+a*np.cos(-a*b)
Unit_B2 = (n2/(l2**2+a**2))*(Unit_B2_1-Unit_B2_2)
Unit_A2 = -(n1/l1)*(np.exp(-l1*(s-b))-np.exp(l1*b))
S_w = Unit_A2 + Unit_B2
S1_w = -(n1/l1)*(np.exp(-l1*(s-b))-np.exp(l1*b))+(n2/l2)*(np.exp(-l2*(s-b))-np.exp(l2*b))
S2_w = S_w - S1_w
Percent_S2_w = (S2_w/S_w)*100

List_S_w = [S1_w, S2_w, S_w, Percent_S2_w]

IntegralDataSet = pd.DataFrame([List_S, List_S_w],
                               index=["Integral Value (Interval: 0 - Infinity)",
                                      "Integral Value (Interval: 0 - 8 Weeks)"],
                               columns=["S1", "S2", "S", "Percent of S2"])
print("")
print("########## Result of Integral Computation (IntegralDataSet) #######################")
print(IntegralDataSet)

########################################################################################################################
# Figures
########################################################################################################################
Figure_object = plt.figure(1, figsize=(11.69, 8.27), dpi=400, edgecolor="black", linewidth=0.5)
#plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)
plt.subplots_adjust(left=0.125, bottom=0.11, right=0.9, top=0.88, wspace=0.2, hspace=0.2)

plt.figtext(0.019, 0.9700, "a", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")
plt.figtext(0.415, 0.9700, "b", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")

plt.figtext(0.019, 0.6300, "c", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")
plt.figtext(0.350, 0.6300, "d", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")
plt.figtext(0.680, 0.6300, "e", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")

plt.figtext(0.019, 0.3050, "f", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")


gs_master = matplotlib.gridspec.GridSpec(nrows=3, ncols=1, height_ratios=[1, 1, 1])
gs_1 = matplotlib.gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=2, subplot_spec=gs_master[0])
gs_2 = matplotlib.gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=3, subplot_spec=gs_master[1])
gs_3 = matplotlib.gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[2])

#ticks_x = [0, 1, 2, 3, 4,  5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
#ticks_y = [-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]

#Axes_obj_01 = Figure_object.add_subplot(gs_1[0])
#Axes_obj_02 = Figure_object.add_subplot(gs_1[1])
#Axes_obj_03 = Figure_object.add_subplot(gs_2[0])
#Axes_obj_04 = Figure_object.add_subplot(gs_2[1])
#Axes_obj_05 = Figure_object.add_subplot(gs_2[2])
#Axes_obj_06 = Figure_object.add_subplot(gs_3[0])

Axes_obj_01 = plt.axes([0.0600644,0.723834-0.060,0.261286*1.275,0.232232*1.275])
Axes_obj_02 = plt.axes([0.579819-0.125,0.723834-0.015,0.399811*1.31,0.232232*1.07])
Axes_obj_03 = plt.axes([0.0600644,0.396547-0.010,0.255435,0.232232])
Axes_obj_04 = plt.axes([0.39213,0.396547-0.010,0.255435,0.232232])
Axes_obj_05 = plt.axes([0.724195,0.396547-0.010,0.255435,0.232232])
Axes_obj_06 = plt.axes([0.0600644,0.0692597-0.010,0.919566,0.232232])

############################################################################################################
# "(a) Cannegieter et al. 2006 Figure 1":
############################################################################################################
#im = Image.open("./achan.jpg") im.show()
Axes_obj_01.set_title("Cannegieter et al. 2006 Figure 1 (\"3D\" Chart)", size=11.4230, fontweight="normal")

Axes_obj_01.yaxis.set_visible(False)
Axes_obj_01.xaxis.set_visible(False)

Axes_obj_01.spines['right'].set_visible(False)
Axes_obj_01.spines['left'].set_visible(False)
Axes_obj_01.spines['top'].set_visible(False)
Axes_obj_01.spines['bottom'].set_visible(False)

img = plt.imread("journal.pmed.0030307.g001.png")
Axes_obj_01.imshow(img)

############################################################################################################
# "(b) Appropriate 2D Bar Chart":
############################################################################################################
Axes_obj_02.set_title("Re-drawing as \"2D\" Bar Chart (unrecognized cycle)", size=11.4230, fontweight="normal")
Axes_obj_02.set_xlabel('Week')
Axes_obj_02.set_ylabel('Cases')
Axes_obj_02.set_ylim(0.0, 80)
#Axes_obj_02.set_xlim(0.0, 90)
Axes_obj_02.set_xticks([0, 1, 2, 3, 4,  5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
Axes_obj_02.set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80])
Axes_obj_02.set_yticklabels(["0", "", "20", "", "40", "", "60", "", "80"])

Axes_obj_02.bar(AnalysisDataSet["Week"], AnalysisDataSet["Cases"], color='blue', width=0.7, alpha=0.3)
Axes_obj_02.scatter(AnalysisDataSet["Week"], AnalysisDataSet["Cases"], color='black')
Axes_obj_02.plot(X_Week, Y_Week, color='black', linestyle='dashed', linewidth=1.0)

for i in range(0, len(AnalysisDataSet["Cases"])):
    Axes_obj_02.text(AnalysisDataSet["Week"][i], AnalysisDataSet["Cases"][i]+5,
                     AnalysisDataSet["Cases"][i], size=10.0, color="black")

############################################################################################################
# "(c) Decreasing Curve":
############################################################################################################
Axes_obj_03.set_title("Decreasing Curve", size=11.4230, fontweight="normal")
Axes_obj_03.set_xlabel('Day')
Axes_obj_03.set_ylabel('Cases')
Axes_obj_03.set_xlim(0.0, 90)
Axes_obj_03.set_ylim(0.0, 80)
Axes_obj_03.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
Axes_obj_03.set_xticklabels(["0", "", "", "30", "", "", "60", "", "", "90"])

Axes_obj_03.set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80])
Axes_obj_03.set_yticklabels(["0", "", "20", "", "40", "", "60", "", "80"])

#Axes_obj_03.xaxis.set_minor_locator(ticker.MultipleLocator(5))
#Axes_obj_03.yaxis.set_minor_locator(ticker.MultipleLocator(5))


Axes_obj_03.plot(X_Day, Y_CF, color='black', linestyle='dashed', linewidth=0.5)
Axes_obj_03.plot(X_Day, Y_MF, color='black', linestyle='solid', linewidth=2.0)
Axes_obj_03.plot(X_Day, Y_EF_U2, color='red', linestyle='dashed', linewidth=0.5)
Axes_obj_03.plot(X_Day, Y_EF_L2, color='blue', linestyle='dashed', linewidth=0.5)
Axes_obj_03.scatter(AnalysisDataSet["Day"], AnalysisDataSet["Cases"], color='red')

for i in range(0, len(AnalysisDataSet["Day"])):

    Axes_obj_03.plot([AnalysisDataSet["Day"][i], AnalysisDataSet["Day"][i]],
                     [n1 * np.exp(-l1 * (AnalysisDataSet["Day"][i]-b)), AnalysisDataSet["Cases"][i]],
                     color='red', linestyle='solid', linewidth=1.5)

############################################################################################################
# "(d) Damped Wave Curve":
############################################################################################################
Axes_obj_04.set_title("Period of Wave", size=11.4230, fontweight="normal")
Axes_obj_04.set_xlabel('Day')
Axes_obj_04.set_ylabel('Residual')
Axes_obj_04.set_xlim(0, 90)
Axes_obj_04.set_ylim(-30, 30)
Axes_obj_04.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
Axes_obj_04.set_xticklabels(["0", "", "", "30", "", "", "60", "", "", "90"])
Axes_obj_04.plot([0.0, 100], [0.0, 0.0], color='black', linestyle='solid', linewidth=0.5)
Axes_obj_04.plot(X_Day, Y_EF_U, color='red', linestyle='dashed', linewidth=1.0)
Axes_obj_04.plot(X_Day, Y_EF_L, color='blue', linestyle='dashed', linewidth=1.0)
Axes_obj_04.plot(X_Day, Y_PF, color='red', linestyle='solid', linewidth=0.5)
Axes_obj_04.scatter(AnalysisDataSet["Day"], Y_DF, color='red')
Axes_obj_04.fill_between(X_Day, Y_PF, Y_EF_L, facecolor='red', alpha=0.05)

Axes_obj_04.vlines(x=Peaks_1[0], ymin=-2.0, ymax=2.0, color="black", linestyle='solid', linewidth=1.5)
Axes_obj_04.vlines(x=Peaks_1[1], ymin=-2.0, ymax=2.0, color="black", linestyle='solid', linewidth=1.5)
Axes_obj_04.vlines(x=Peaks_1[2], ymin=-2.0, ymax=2.0, color="black", linestyle='solid', linewidth=1.5)
Axes_obj_04.vlines(x=Peaks_1[3], ymin=-2.0, ymax=2.0, color="black", linestyle='solid', linewidth=1.5)

Axes_obj_04.vlines(x=Peaks_1[0], ymin=22, ymax=24, color="black", linestyle='solid', linewidth=1.5)
Axes_obj_04.vlines(x=Peaks_1[1], ymin=22, ymax=24, color="black", linestyle='solid', linewidth=1.5)
Axes_obj_04.vlines(x=Peaks_1[2], ymin=22, ymax=24, color="black", linestyle='solid', linewidth=1.5)
Axes_obj_04.vlines(x=Peaks_1[3], ymin=22, ymax=24, color="black", linestyle='solid', linewidth=1.5)

Axes_obj_04.text(Peaks_1[0]+1.0, 26, Peaks_1_round[0],
                 size=10, color="red", ha='center', va='center', fontweight="normal",
                 bbox=dict(boxstyle='round', edgecolor='red', fc='red', alpha=0.0))
Axes_obj_04.text(Peaks_1[1], 26, Peaks_1_round[1],
                 size=10, color="red", ha='center', va='center', fontweight="normal",
                 bbox=dict(boxstyle='round', edgecolor='red', fc='red', alpha=0.0))
Axes_obj_04.text(Peaks_1[2], 26, Peaks_1_round[2],
                 size=10, color="red", ha='center', va='center', fontweight="normal",
                 bbox=dict(boxstyle='round', edgecolor='red', fc='red', alpha=0.0))
Axes_obj_04.text(Peaks_1[3]-1.0, 26, Peaks_1_round[3],
                 size=10, color="red", ha='center', va='center', fontweight="normal",
                 bbox=dict(boxstyle='round', edgecolor='red', fc='red', alpha=0.0))

Axes_obj_04.vlines(x=Peaks_1[0]+(np.pi/2/a), ymin=-30, ymax=1, color="black", linestyle='solid', linewidth=0.5)
Axes_obj_04.vlines(x=Peaks_1[1]+(np.pi/2/a), ymin=-30, ymax=1, color="black", linestyle='solid', linewidth=0.5)

Axes_obj_04.quiver(Peaks_1[0]-(np.pi/2/a)+2*np.pi/a, -25, -(np.pi/a), 0.0,
                   scale_units='xy', angles='xy', scale=1, linestyle='solid', width=0.008)
Axes_obj_04.quiver(Peaks_1[0]-(np.pi/2/a)+2*np.pi/a, -25, (np.pi/a), 0.0,
                   scale_units='xy', angles='xy', scale=1, linestyle='solid', width=0.008)

Axes_obj_04.text(Peaks_1[0]-(np.pi/2/a)+2*np.pi/a, -20,
                 Period_round, size=11.4230, color="red", ha='center', va='center', fontweight="normal",
                 bbox=dict(boxstyle='round', edgecolor='red', fc='red', alpha=0.0))

Axes_obj_04.text(70, -17.5, '\n'.join([''.join([str(Period_round), ' days']), "cycle"]),
                 size=11.4230, color="black", ha='center', va='center',
                 bbox=dict(boxstyle='round', edgecolor='red', fc='white'))

############################################################################################################
# "(e) Added Damped Wave":
############################################################################################################
Axes_obj_05.set_title("Two Types of parts", size=11.4230, fontweight="normal")
Axes_obj_05.set_xlabel('Day')
Axes_obj_05.set_ylabel('Cases')
Axes_obj_05.set_xlim(0.0, 90)
Axes_obj_05.set_ylim(0.0, 80)
Axes_obj_05.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
Axes_obj_05.set_xticklabels(["0", "", "", "30", "", "", "60", "", "", "90"])
Axes_obj_05.set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80])
Axes_obj_05.set_yticklabels(["0", "", "20", "", "40", "", "60", "", "80"])
Axes_obj_05.plot(X_Day, Y_CF, color='black', linestyle='solid', linewidth=1.5)
Axes_obj_05.plot(X_Day, Y_EF_U2, color='red', linestyle='dashed', linewidth=1.0)
Axes_obj_05.plot(X_Day, Y_EF_L2, color='blue', linestyle='dashed', linewidth=1.0)
Axes_obj_05.scatter(AnalysisDataSet["Day"], AnalysisDataSet["Cases"], color='black')
Axes_obj_05.fill_between(X_Day, Y_CF, Y_EF_L2, facecolor='red', alpha=0.1)
Axes_obj_05.fill_between(X_Day, Y_EF_L2, Line_fill, facecolor='blue', alpha=0.1)

### Annotations: Texts###
Axes_obj_05.annotate(
 text='\n'.join(["Periodically part"]),
    xy=(2.5, 40), xytext=(2.5+22.5, 40+25), ha='left', va='center', size=10.0,
    bbox=dict(boxstyle='round', edgecolor='red', fc='white'),
    arrowprops=dict(
        facecolor='red', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0, edgecolor='red', shrink=0.1)
)

Axes_obj_05.annotate(
 text='\n'.join(["Non-periodically part"]),
    xy=(23, 10), xytext=(23+22.5-5.0, 10+25), ha='left', va='center', size=10.0,
    bbox=dict(boxstyle='round', edgecolor='blue', fc='white'),
    arrowprops=dict(
        facecolor='blue', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0, edgecolor='blue', shrink=0.1)
)

############################################################################################################
# "(f) Added Damped Wave":
############################################################################################################
Axes_obj_06.set_title("Peaks of the Cases (Risk), Ratio of the Arias and End of the Wave", size=11.4230, fontweight="normal")
Axes_obj_06.set_xlabel('Day')
Axes_obj_06.set_ylabel('Cases')
Axes_obj_06.set_xlim(0.0, 190)
Axes_obj_06.set_ylim(0.0, 100)
Axes_obj_06.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180])
Axes_obj_06.set_xticklabels(["0", "", "", "30", "", "", "60", "", "", "90", "", "", "120", "", "", "150", "", "", "180"])
Axes_obj_06.set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
Axes_obj_06.set_yticklabels(["0", "", "20", "", "40", "", "60", "", "80", "", "100"])

Axes_obj_06.plot(X_Day, Y_CF, color='black', linestyle='solid', linewidth=1.5)
Axes_obj_06.plot(X_Day, Y_EF_L2, color='blue', linestyle='dashed', linewidth=1.0)
Axes_obj_06.scatter(AnalysisDataSet["Day"], AnalysisDataSet["Cases"], color='black')
Axes_obj_06.fill_between(X_Day, Y_CF, Y_EF_L2, facecolor='red', alpha=0.1)
Axes_obj_06.fill_between(X_Day, Y_EF_L2, Line_fill, facecolor='blue', alpha=0.1)

### Annotations (1): Texts###

y_1 = n1 * np.exp(-l1 * (35-b)) - (n2 * np.exp(-l2 * (35-b)))
y_2 = n1 * np.exp(-l1 * (8.5-b)) + (n2 * np.exp(-l2 * (8.5-b))) * (np.sin(a * (8.5-b)))

Axes_obj_06.annotate(
 text="Y = f(x)",
    xy=(8.5, y_2), xytext=(8.5+5.0, y_2+22.5), ha='left', va='center', size=14.6868,
    bbox=dict(boxstyle='round', edgecolor='white', fc='white'),
    arrowprops=dict(
        facecolor='black', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0, edgecolor='black', shrink=0.1)
)

Axes_obj_06.annotate(
 text="Y = g(x)",
    xy=(33, y_1), xytext=(33+6.0, y_1+22.5), ha='left', va='center', size=14.6868,
    bbox=dict(boxstyle='round', edgecolor='white', fc='white'),
    arrowprops=dict(
        facecolor='blue', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0, edgecolor='blue', shrink=0.1)
)

Axes_obj_06.text(4.0, 45, "$S_2$", size=14.6868, color="red", ha='center', va='center', fontweight="normal",
                 bbox=dict(boxstyle='round', edgecolor='red', fc='red', alpha=0.0))

Axes_obj_06.text(22.5, 10, "$S_1$", size=14.6868, color="blue", ha='center', va='center', fontweight="normal",
                 bbox=dict(boxstyle='round', edgecolor='blue', fc='blue', alpha=0.0))

### Annotations (2): Texts & Allows###
#################################################################################################
Axes_obj_06.text(28, 95, "Analysis Data Set (8 weeks)", size=10.0, color="black", ha='center', va='center',
                 bbox=dict(boxstyle='round', edgecolor='black', fc='white', alpha=0.0))
Axes_obj_06.quiver(11, 95, -11.0, 0.0, scale_units='xy', angles='xy', scale=1, linestyle='solid', width=0.002)
Axes_obj_06.quiver(45, 95, 11.0, 0.0, scale_units='xy', angles='xy', scale=1, linestyle='solid', width=0.002)
Axes_obj_06.vlines(x=56.0, ymin=0.0, ymax=100, color="black", linestyle='solid', linewidth=0.5)

Axes_obj_06.vlines(x=Peaks_1[0], ymin=72.5, ymax=77.5, color="black", linestyle='solid', linewidth=1.5)
Axes_obj_06.vlines(x=Peaks_1[1], ymin=34.5, ymax=39.5, color="black", linestyle='solid', linewidth=1.5)

Axes_obj_06.quiver(Peaks_1[0] + 2.5, 77.5+5.0, -5.0, 0.0,
                   scale_units='xy', angles='xy', scale=1, linestyle='solid', color="red", width=0.004)
Axes_obj_06.quiver(Peaks_1[1] + 2.5-1.25, 39.5+5.0, -5.0-1.25, 0.0,
                   scale_units='xy', angles='xy', scale=1, linestyle='solid', color="red", width=0.004)


Axes_obj_06.annotate(text=Values_round[2], xy=(90, Values[2]+2.5), xytext=(90, Values[2]+22.5),
                     ha='center', va='center', size=10.0,
                     bbox=dict(boxstyle='round', edgecolor='black', fc='white'),
                     arrowprops=dict(facecolor='black', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0,
                                      edgecolor='black', shrink=0.1))

Axes_obj_06.annotate(text=Values_round[3], xy=(120, Values[3]+2.5), xytext=(120, Values[3]+22.5),
                     ha='center', va='center', size=10.0,
                     bbox=dict(boxstyle='round', edgecolor='black', fc='white'),
                     arrowprops=dict(facecolor='black', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0,
                                      edgecolor='black', shrink=0.1))

Axes_obj_06.annotate(text=Values_round[4], xy=(150, Values[4]+2.5), xytext=(150, Values[4]+22.5),
                     ha='center', va='center', size=10.0,
                     bbox=dict(boxstyle='round', edgecolor='black', fc='white'),
                     arrowprops=dict(facecolor='black', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0,
                                      edgecolor='black', shrink=0.1))

Axes_obj_06.annotate(text=Values_round[5], xy=(180, Values[5]+2.5), xytext=(180, Values[5]+22.5),
                     ha='center', va='center', size=10.0,
                     bbox=dict(boxstyle='round', edgecolor='black', fc='white'),
                     arrowprops=dict(facecolor='black', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0,
                                      edgecolor='black', shrink=0.1))

############################################################################################################
text3 = Decimal(str(List_S[3])).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)
text4 = Decimal(str(List_S_w[3])).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)

text01 = " Interval      % of S2 "
text02 = "-----------------------"
text03 = ''.join([" 0 - ∞", "          ",  ''.join([str(text3), " ", "% "])])
text04 = ''.join([" 0 - 8 weeks", "    ", ''.join([str(text4), " ", "% "])])

text01_2 = matplotlib.offsetbox.TextArea(text01,
                                textprops=dict(fontname="monospace", color="black", fontsize=11.4230, fontweight="normal"))
text02_2 = matplotlib.offsetbox.TextArea(text02,
                                textprops=dict(fontname="monospace", color="black", fontsize=11.4230))
text03_2 = matplotlib.offsetbox.TextArea(text03,
                                textprops=dict(fontname="monospace", color="black", fontsize=11.4230))
text04_2 = matplotlib.offsetbox.TextArea(text04,
                                textprops=dict(fontname="monospace", color="black", fontsize=11.4230))

texts_vbox_1 = matplotlib.offsetbox.VPacker(
    children=[text01_2, text02_2, text03_2, text04_2],
                                align="left", mode="fixed", pad=0.0, sep=0.1)
ann_1 = matplotlib.offsetbox.AnnotationBbox(texts_vbox_1,
                                box_alignment=(1, 1), xycoords="data", xy=(190-190*0.01, 100-100*0.055),
                                bboxprops=dict(boxstyle='round', edgecolor='red', fc='white'),
                                frameon=True)
Axes_obj_06.add_artist(ann_1)

############################################################################################################
# Equations
############################################################################################################
#\left( …… \right)
#\left{a\left(x-b\right) \right}

Axes_obj_06.text(60, 85,
                 r"$f(x) = n_{1}e^{-\lambda_{1}(x-b)}+n_{2}e^{-\lambda_{2}(x-b)}\sin{\{a\left(x-b\right)\}}$",
                 horizontalalignment='left', fontsize=14.6868, fontname="monospace")
Axes_obj_06.text(60, 65,
                 r"$g(x) = n_{1}e^{-\lambda_{1}(x-b)}-n_{2}e^{-\lambda_{2}(x-b)}$",
                 horizontalalignment='left', fontsize=14.6868, fontname="monospace")
Axes_obj_06.text(65, 42.5,
                 r"$S_{1} = \int_{0}^{\infty} g(x)\mathrm{d}x$",
                 horizontalalignment='left', fontsize=13.0569, color="blue", alpha=1.0)
Axes_obj_06.text(90, 42.5,
                 r"$S_{2} = S - S_{1} = \int_{0}^{\infty} f(x)\mathrm{d}x - S_{1} = \lim_{t\to\infty}\int_{0}^{t} f(x)\mathrm{d}x -S_{1}$",
                 horizontalalignment='left', fontsize=13.0569, color="red", alpha=1.0)
############################################################################################################

#Figure_object.tight_layout()
my_path = os.path.abspath("Figures")
plt.savefig(os.path.join(my_path, "00_Kimoto_et_al_(2023)_[_Fig_04_].png"))

from PIL import Image # py -m pip install pillow
img = Image.open(os.path.join(my_path, "00_Kimoto_et_al_(2023)_[_Fig_04_].png"))
img_resize = img.resize(size=(2866, 2016))
my_path2 = os.path.abspath("Figures/Fixed_Version/B6")
img_resize.save(os.path.join(my_path2, "00_Kimoto_et_al_(2023)_[_Fig_04_]_B6.png"))

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

