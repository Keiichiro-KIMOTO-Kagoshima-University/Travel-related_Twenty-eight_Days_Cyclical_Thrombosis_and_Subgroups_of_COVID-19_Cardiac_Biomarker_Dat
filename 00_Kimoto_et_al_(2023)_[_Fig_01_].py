########################################################################################################################
# Fig. 1. A hyperbolic shape formed by confidence limits. (00_Kimoto_et_al_(2023)_[_Fig_01_].py)
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
#    Non-linear Regression Analysis
# Coefficient Matrix (Data Set)
# Values for Curves (Week)
# For Ideal Diagram for Explanation (S Curve)
# Figures
#    (a) Calculation of Odds Ratio & Confidence Interval", fontweight="normal
#       Table
#       Equations
#    (b) Binomial Aproximation to the Nomal
#    (c) Definition of Polygon that Should Be Identified in the Figure of Meta-regression Analysis
#       Equations
########################################################################################################################
# List of Import
########################################################################################################################
from typing import List
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
# cd /home/kimoto/.cache/matplotlib
# rm fontlist-v310.json
# rm fontlist-v330.json

print(matplotlib.matplotlib_fname())
print(matplotlib.rcParams["font.family"])
print(matplotlib.rcParams["font.sans-serif"])
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rcParams['font.sans-serif'] = ["Arial"]
########################################################################################################################
# Making Data Sets
########################################################################################################################
print("########## Data Set from Microsoft Excel BINOM.DIST Function #######################")
#Binomial distribution, Mean = np = 0.5*30 = 15, Variance = np(1-p) = 30*0.5*(1-0.5) = 7.5
# SD= Sqrt(7.5) = 2.738612788, 2SD = 5.477225575
# Mean - 2SD = 15 - 5.477225575 = 9.522774425
# Mean + 2SD = 15 + 5.477225575 = 20.47722558

Mean_mSD = 0.5*30 - np.sqrt(30*0.5*(1-0.5))
Mean_pSD = 0.5*30 + np.sqrt(30*0.5*(1-0.5))

Mean_m2SD = 0.5*30 - 2*np.sqrt(30*0.5*(1-0.5))
Mean_p2SD = 0.5*30 + 2*np.sqrt(30*0.5*(1-0.5))

Mean_m3SD = 0.5*30 - 3*np.sqrt(30*0.5*(1-0.5))
Mean_p3SD = 0.5*30 + 3*np.sqrt(30*0.5*(1-0.5))

# Binomial Aproximation to the Nomal and Parabora
Bdata_00 = [0, 9.31322574615479*(10**(-10)), 9.31322574615479*(10**(-10))]
Bdata_01 = [1, 2.79396772384644*(10**(-8)), 2.88709998130798*(10**(-8))]
Bdata_02 = [2, 4.05125319957733*(10**(-7)), 4.33996319770813*(10**(-7))]
Bdata_03 = [3, 3.78116965293884*(10**(-6)), 4.21516597270966*(10**(-6))]
Bdata_04 = [4, 2.55228951573372*(10**(-5)), 2.97380611300468*(10**(-5))]
Bdata_05 = [5, 0.000132719054818, 0.000162457115948]
Bdata_06 = [6, 0.000552996061742, 0.000715453177691]
Bdata_07 = [7, 0.001895986497402, 0.002611439675093]
Bdata_08 = [8, 0.005450961180031, 0.008062400855124]

Bdata_09 = [9, 0.01332457177341, 0.021386972628534]
Bdata_10 = [10, 0.027981600724161, 0.049368573352695]
Bdata_11 = [11, 0.050875637680292, 0.100244211032987]
Bdata_12 = [12, 0.080553092993796, 0.180797304026783]
Bdata_13 = [13, 0.111535051837564, 0.292332355864346]
Bdata_14 = [14, 0.13543542008847, 0.427767775952816]
Bdata_15 = [15, 0.144464448094368, 0.572232224047184]
Bdata_16 = [16, 0.13543542008847, 0.707667644135654]
Bdata_17 = [17, 0.111535051837564, 0.819202695973218]
Bdata_18 = [18, 0.080553092993796, 0.899755788967013]
Bdata_19 = [19, 0.050875637680292, 0.950631426647306]
Bdata_20 = [20, 0.027981600724161, 0.978613027371466]
Bdata_21 = [21, 0.01332457177341, 0.991937599144876]

Bdata_22 = [22, 0.005450961180031, 0.997388560324907]
Bdata_23 = [23, 0.001895986497402, 0.99928454682231]
Bdata_24 = [24, 0.000552996061742, 0.999837542884052]
Bdata_25 = [25, 0.000132719054818, 0.99997026193887]
Bdata_26 = [26, 2.55228951573372*(10**(-5)), 0.999995784834027]
Bdata_27 = [27, 3.78116965293884*(10**(-6)), 0.99999956600368]
Bdata_28 = [28, 4.05125319957733*(10**(-7)), 0.999999971129]
Bdata_29 = [29, 2.79396772384644*(10**(-8)), 0.999999999068677]
Bdata_30 = [30, 9.31322574615479*(10**(-10)), 1]

OriginalDataSet = pd.DataFrame([Bdata_00, Bdata_01, Bdata_02, Bdata_03, Bdata_04, Bdata_05,
                                Bdata_06, Bdata_07, Bdata_08, Bdata_09, Bdata_10,
                                Bdata_11, Bdata_12, Bdata_13, Bdata_14, Bdata_15,
                                Bdata_16, Bdata_17, Bdata_18, Bdata_19, Bdata_20,
                                Bdata_21, Bdata_22, Bdata_23, Bdata_24, Bdata_25,
                                Bdata_26, Bdata_27, Bdata_28, Bdata_29, Bdata_30],
                               index=["Bdata_00", "Bdata_01", "Bdata_02", "Bdata_03", "Bdata_04", "Bdata_05",
                                      "Bdata_06", "Bdata_07", "Bdata_08", "Bdata_09", "Bdata_10",
                                      "Bdata_11", "Bdata_12", "Bdata_13", "Bdata_14", "Bdata_15",
                                      "Bdata_16", "Bdata_17", "Bdata_18", "Bdata_19", "Bdata_20",
                                      "Bdata_21", "Bdata_22", "Bdata_23", "Bdata_24", "Bdata_25",
                                      "Bdata_26", "Bdata_27", "Bdata_28", "Bdata_29", "Bdata_30"],
                               columns=["N", "Data", "Data (cumulative value)"]
                               )

OriginalDataSet["Confidence Limit"] = 1/np.sqrt(OriginalDataSet["Data"])
ParaboraDataSet = OriginalDataSet[(OriginalDataSet["N"] > 9) & (OriginalDataSet["N"] < 21)]

print("")
print(OriginalDataSet)
print(ParaboraDataSet)

############################################################################################################
#    Non-linear Regression Analysis
############################################################################################################
def reg_func(parameter: ["a", "b", "c"], x, y):
    a = parameter[0]
    b = parameter[1]
    c = parameter[2]
    _Output_ = y - (a*x**2+b*x+c)
    return _Output_
############################################################################################################
x = ParaboraDataSet["N"]
y = ParaboraDataSet["Confidence Limit"]
parameter_0 = [0.1, -3.0, 30]
RegResult_Original = optimize.leastsq(reg_func, parameter_0, args=(x, y), full_output=True)
Result_0 = RegResult_Original[0]
Result = [Result_0[0], Result_0[1], Result_0[2]]

print("")
print("########## Evaluation of Result on the Regression Analysis ##########")
print(["Reg. Analysis Result:", RegResult_Original[-1], "1, 2, 3 or 4, the solution was found."])
print(Result)


############################################################################################################
#    Non-linear Regression Analysis (catenary)
############################################################################################################
def reg_func_catenary(parameter: ["a", "b", "c", "d"], x, y):
    a = parameter[0]
    b = parameter[1]
    c = parameter[2]
    d = parameter[3]
    _Output_ = y - (a*((np.exp((x-c)/b)+np.exp(-(x-c)/b))/2) + d)
    return _Output_

############################################################################################################
x = ParaboraDataSet["N"]
y = ParaboraDataSet["Confidence Limit"]
parameter_0 = [3.0, 3.0, 15, 0.0]
RegResult_Original_catenary = optimize.leastsq(reg_func_catenary, parameter_0, args=(x, y), full_output=True)
Result_0_catenary = RegResult_Original_catenary[0]
Result_catenary = [Result_0_catenary[0], Result_0_catenary[1], Result_0_catenary[2], Result_0_catenary[3]]

print("")
print("########## Evaluation of Result on the Regression Analysis (catenary) ##########")
print(["Reg. Analysis Result:", RegResult_Original_catenary[-1], "1, 2, 3 or 4, the solution was found."])
print(Result_catenary)

############################################################################################################
# Coefficient Matrix (Data Set)
############################################################################################################
CoefficientDataSet = pd.DataFrame([Result], index=["Reg. Result"], columns=["a", "b", "c"])
print("")
print("########## Result of Non-linear Regression Analysis (CoefficientDataSet) ##########")
print(CoefficientDataSet)
########################################################################################################################
# Values for Curves (parabora)
########################################################################################################################
a = CoefficientDataSet.at["Reg. Result", "a"]
b = CoefficientDataSet.at["Reg. Result", "b"]
c = CoefficientDataSet.at["Reg. Result", "c"]

X_parabora = np.linspace(15-10, 15+10, 100)
Y_parabora = a*X_parabora**2+b*X_parabora+c

########################################################################################################################
# Values for Curves (catenary)
########################################################################################################################
a = Result_catenary[0]
b = Result_catenary[1]
c = Result_catenary[2]
d = Result_catenary[3]

X_catenary = np.linspace(15-10, 15+10, 100)
Y_catenary = a*((np.exp((X_catenary-c)/b)+np.exp(-(X_catenary-c)/b))/2) + d

########################################################################################################################
# For Ideal Diagram for Explanation (S Curve)
########################################################################################################################

x_ideal = np.linspace(0, 10, 1000)
y_ideal = 7.5 / (1 + np.exp(-2.0 * (x_ideal - 5))) + 6.25
x_ideal2 = np.array([0.0, 2.5, 5.0, 7.5, 10.0])
y_ideal2 = 7.5 / (1 + np.exp(-2.0 * (x_ideal2 - 5))) + 6.25

y_ideal_u = (7.5 / (1 + np.exp(-2.0 * (x_ideal - 5)))) + 6.25 + 0.1*(x_ideal - 5.0)**2 + 2.0
y_ideal_l = (7.5 / (1 + np.exp(-2.0 * (x_ideal - 5)))) + 6.25 - 0.1*(x_ideal - 5.0)**2 - 2.0
y_ideal2_u = (7.5 / (1 + np.exp(-2.0 * (x_ideal2 - 5)))) + 6.25 + 0.1*(x_ideal2 - 5.0)**2 + 2.0
y_ideal2_l = (7.5 / (1 + np.exp(-2.0 * (x_ideal2 - 5)))) + 6.25 - 0.1*(x_ideal2 - 5.0)**2 - 2.0
########################################################################################################################
# Figures
########################################################################################################################
Figure_object = plt.figure(1, figsize=(11.69, 8.27), dpi=400, edgecolor="black", linewidth=0.5)

plt.figtext(0.0225, 0.95, "a", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")
plt.figtext(0.0225, 0.75, "b", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")
plt.figtext(0.5000, 0.95, "c", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")
plt.figtext(0.0225, 0.45, "d", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")

gs_master = matplotlib.gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[1, 1])
gs_1 = matplotlib.gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=2, subplot_spec=gs_master[0])
gs_2 = matplotlib.gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[1])

Axes_obj_01 = Figure_object.add_subplot(gs_1[0])
Axes_obj_02 = Figure_object.add_subplot(gs_1[1])
Axes_obj_03 = Figure_object.add_subplot(gs_2[0])

Axes_obj_01.set_title("Calculation of Odds Ratio & Confidence Interval", fontweight="normal", fontsize=11.4230)
Axes_obj_02.set_title("Numerical Experiment by Using of Binomial Data", fontweight="normal", fontsize=11.4230)
#Axes_obj_02.set_xlabel('Binomial Distribution as Approximation of Normal Distribution')
Axes_obj_02.set_xlabel('')
Axes_obj_02.set_ylabel('N & Integrated Value of N', fontsize=11.4320, color='blue')
Axes_obj_03.set_title(
    "Definition of the Diagram that Should Be Identified in the Figure of Meta-regression Analysis",
                      fontweight="normal", fontsize=11.4230)

ax1 = Axes_obj_02.twinx()
ax1.set_ylabel(r"$1/\sqrt{N}}$", fontsize=11.4320, color='red')

#Axes_obj_03.text(5.0, 18.25, "Target Diagram",
#                 horizontalalignment='center', fontsize=11.4230, fontweight="normal", color="black")

Axes_obj_03.text(2.0, 19.0, "OR + LN(OR)",
                 horizontalalignment='left', fontsize=11.4230, fontweight="normal", color="red")
Axes_obj_03.text(2.0, 17.0, "OR: NOT LN(OR)",
                 horizontalalignment='left', fontsize=11.4230, fontweight="normal", color="blue")
Axes_obj_03.text(2.0, 15.0, "OR - LN(OR)",
                 horizontalalignment='left', fontsize=11.4230, fontweight="normal", color="red")

Axes_obj_02.text(6.0, 0.4, "Parabola (Quadratic Function)",
                 horizontalalignment='center', fontsize=10, fontweight="normal", color="magenta", rotation=-70, zorder=10)

Axes_obj_03.text(25, 5.5, "Completing the square of the quadratic function",
                 horizontalalignment='center', fontsize=10, fontweight="normal", color="black")

Axes_obj_03.text(6.0, 6.25, "Formula for the Curve of Distribution",
                 horizontalalignment='left', fontsize=10, fontweight="normal")

Axes_obj_03.text(28.2-2.0, np.max(y_ideal)+0.4, "Sigmoid Function",
                 horizontalalignment='center', fontsize=10, fontweight="normal", color="blue")

Axes_obj_03.text(26.075, 20.15-0.4, "Quadratic Function",
                 horizontalalignment='center', fontsize=10, fontweight="normal", color="red")

Axes_obj_03.text(26.075, 11.15-0.4, "Quadratic Function",
                 horizontalalignment='center', fontsize=10, fontweight="normal", color="red")

Axes_obj_03.text(17.0, 0.4, "Latent distribution that generate target diagram (M: Mean)",
                 horizontalalignment='center', fontsize=10, fontweight="normal", color="black")

Axes_obj_03.text(5.0, 0.4, "M",
                 horizontalalignment='center', fontsize=11.4230, fontweight="normal", color="black")

Axes_obj_03.text(33.15-0.5-2.0, y_ideal2_u[4], "S",
                 horizontalalignment='center', fontsize=11.4230, fontweight="normal", color="blue")
Axes_obj_03.text(33.15-2.0, y_ideal2_u[4], "+",
                 horizontalalignment='center', fontsize=11.4230, fontweight="normal", color="black")
Axes_obj_03.text(33.15+0.5-2.0+1.0, y_ideal2_u[4], "U curve",
                 horizontalalignment='center', fontsize=11.4230, fontweight="normal", color="red")

Axes_obj_03.text(33.15-2.0+0.75, np.max(y_ideal), "S curve",
                 horizontalalignment='center', fontsize=11.4230, fontweight="normal", color="blue")

Axes_obj_03.text(33.15-0.5-2.0, y_ideal2_l[4], "S",
                 horizontalalignment='center', fontsize=11.4230, fontweight="normal", color="blue")
Axes_obj_03.text(33.15-2.0, y_ideal2_l[4], "-",
                 horizontalalignment='center', fontsize=11.4230, fontweight="normal", color="black")
Axes_obj_03.text(33.15+0.5-2.0+1.0, y_ideal2_l[4], "U curve",
                 horizontalalignment='center', fontsize=11.4230, fontweight="normal", color="red")

############################################################################################################
# (a) Calculation of Odds Ratio & Confidence Interval", fontweight="normal
############################################################################################################
#Axes_obj_01.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, length=0.0)
Axes_obj_01.yaxis.set_visible(False)
Axes_obj_01.xaxis.set_visible(False)
Axes_obj_01.spines['right'].set_visible(False)
Axes_obj_01.spines['left'].set_visible(False)
Axes_obj_01.spines['top'].set_visible(False)
Axes_obj_01.spines['bottom'].set_visible(False)

Axes_obj_01.set_xlim(0.0, 1.0)
Axes_obj_01.set_ylim(0.0, 1.0)

###################################################################################################
# Table
###################################################################################################
Axes_obj_01.hlines(y=0.9750, xmin=0.0125, xmax=0.9875, color="black", linestyle="solid", linewidth=1.6318)
Axes_obj_01.hlines(y=0.8750, xmin=0.0125, xmax=0.9875, color="black", linestyle="solid", linewidth=1.6318)
Axes_obj_01.hlines(y=0.6750, xmin=0.0125, xmax=0.9875, color="black", linestyle="solid", linewidth=1.6318)
Axes_obj_01.hlines(y=0.5750, xmin=0.0125, xmax=0.9875, color="black", linestyle="solid", linewidth=1.6318)

Axes_obj_01.hlines(y=0.5500, xmin=0.001, xmax=0.999, color="black", linestyle="solid", linewidth=1.0000)
Axes_obj_01.hlines(y=0.001, xmin=0.001, xmax=0.999, color="black", linestyle="solid", linewidth=1.0000)

Axes_obj_01.vlines(x=0.001, ymin=0.001, ymax=0.5500, color="black", linestyle="solid", linewidth=1.0000)
Axes_obj_01.vlines(x=0.999, ymin=0.001, ymax=0.5500, color="black", linestyle="solid", linewidth=1.0000)


Axes_obj_01.text(0.13125, 0.925, "Exposure", size=11.4230,
                 color="black", horizontalalignment="center", verticalalignment="center", fontweight="normal")
Axes_obj_01.text(0.36875, 0.925, "Case", size=11.4230,
                 color="black", horizontalalignment="center", verticalalignment="center", fontweight="normal")
Axes_obj_01.text(0.63125, 0.925, "Control", size=11.4230,
                 color="black", horizontalalignment="center", verticalalignment="center", fontweight="normal")
Axes_obj_01.text(0.86875, 0.925, "Total", size=11.4230,
                 color="black", horizontalalignment="center", verticalalignment="center", fontweight="normal")

Axes_obj_01.text(0.13125, 0.825, "Yes", size=11.4230,
                 color="black", horizontalalignment="center", verticalalignment="center", fontweight="normal")
Axes_obj_01.text(0.36875, 0.825, "a", size=11.4230,
                 color="black", horizontalalignment="center", verticalalignment="center", fontweight="normal")
Axes_obj_01.text(0.63125, 0.825, "b", size=11.4230,
                 color="black", horizontalalignment="center", verticalalignment="center", fontweight="normal")
Axes_obj_01.text(0.86875, 0.825, "a+b", size=11.4230,
                 color="black", horizontalalignment="center", verticalalignment="center", fontweight="normal")

Axes_obj_01.text(0.13125, 0.725, "No", size=11.4230,
                 color="black", horizontalalignment="center", verticalalignment="center", fontweight="normal")
Axes_obj_01.text(0.36875, 0.725, "c", size=11.4230,
                 color="black", horizontalalignment="center", verticalalignment="center", fontweight="normal")
Axes_obj_01.text(0.63125, 0.725, "d", size=11.4230,
                 color="black", horizontalalignment="center", verticalalignment="center", fontweight="normal")
Axes_obj_01.text(0.86875, 0.725, "c+d", size=11.4230,
                 color="black", horizontalalignment="center", verticalalignment="center", fontweight="normal")

Axes_obj_01.text(0.13125, 0.625, "Total", size=11.4230,
                 color="black", horizontalalignment="center", verticalalignment="center", fontweight="normal")
Axes_obj_01.text(0.36875, 0.625, "a+c", size=11.4230,
                 color="black", horizontalalignment="center", verticalalignment="center", fontweight="normal")
Axes_obj_01.text(0.63125, 0.625, "b+d", size=11.4230,
                 color="black", horizontalalignment="center", verticalalignment="center", fontweight="normal")
Axes_obj_01.text(0.86875, 0.625, "a+b+c+d", size=11.4230,
                 color="black", horizontalalignment="center", verticalalignment="center", fontweight="normal")


###################################################################################################
# Equations
###################################################################################################
#matplotlib.rc('text', usetex=True)
#matplotlib.rc('text.latex', preamble='\usepackage{color}')

Axes_obj_01.text(0.0125, 0.475,
        r"$Odds\ Ratio\ (OR) = \frac{a \times d}{b \times c} = \frac{a/c}{b/d} = \frac{a/b}{c/d}$",
                 horizontalalignment='left', fontsize=14.6868)
Axes_obj_01.text(0.0125, 0.375-0.04,
        r"$95\%CI=\exp{\left\{\log_{e}(OR) \quad± 1.96\sqrt{\frac{1}{a}+\frac{1}{b}+\frac{1}{c}+\frac{1}{d}}\right\}}$",
                 horizontalalignment='left', fontsize=14.6868)
Axes_obj_01.text(0.0215, 0.275-0.08,
        r"$\log_{e}(95\%CI)=\log_{e}(OR) \quad± 1.96\sqrt{\frac{1}{a}+\frac{1}{b}+\frac{1}{c}+\frac{1}{d}}$",
                 horizontalalignment='left', fontsize=14.6868)
Axes_obj_01.text(0.276, 0.175-0.15/2-0.035,
                 r"$\simeq\log_{e}(OR) \quad± 1.96\times\frac{1}{\sqrt{N}}$",
                 horizontalalignment='left', fontsize=14.6868)


Axes_obj_01.fill_between(np.linspace(0.699+0.025, 0.774+0.025, 1000), 0.1525*np.ones(1000), 0.0150*np.ones(1000),
                         facecolor='yellow', alpha=0.3)

Axes_obj_01.vlines(x=0.699+0.025, ymin=0.015, ymax=0.1525, color="red", linestyle='solid', linewidth=1.0)
Axes_obj_01.vlines(x=0.774+0.025, ymin=0.015, ymax=0.1525, color="red", linestyle='solid', linewidth=1.0)

Axes_obj_01.hlines(y=0.1525, xmin=0.699+0.025, xmax=0.774+0.025, color="red", linestyle="solid", linewidth=1.0)
Axes_obj_01.hlines(y=0.0150, xmin=0.699+0.025, xmax=0.774+0.025, color="red", linestyle="solid", linewidth=1.0)

############################################################################################################

############################################################################################################
# (b) Binomial Aproximation to the Nomal
############################################################################################################
Axes_obj_02.tick_params(direction="in")
Axes_obj_02.set_xlim(0.0, 30.0)
Axes_obj_02.set_ylim(0.0, 1.4)
Axes_obj_02.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
Axes_obj_02.set_yticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0", "1.2", ""])

Axes_obj_02.bar(OriginalDataSet["N"], OriginalDataSet["Data"], color='blue', width=0.7, alpha=0.5, zorder=10)
Axes_obj_02.scatter(OriginalDataSet["N"], OriginalDataSet["Data (cumulative value)"], color='blue', alpha=0.3, zorder=10)

ax1.tick_params(direction="in")
ax1.plot(X_parabora, Y_parabora, color='magenta', linestyle='solid', linewidth=1.6318, zorder=9)

ax1.plot(X_catenary, Y_catenary, color='black', linestyle='dashed', linewidth=1.6318, zorder=10)




ax1.scatter(OriginalDataSet["N"], OriginalDataSet["Confidence Limit"], color='black', zorder=10)
ax1.scatter(ParaboraDataSet["N"], ParaboraDataSet["Confidence Limit"], color='red', zorder=10)
#ax1.set_ylabel(r"$1/\sqrt{N}}$", fontsize=13.0549, color='red')
ax1.set_ylim(0.0, 15.5)

Axes_obj_02.vlines(x=Mean_mSD, ymin=1.3-0.05, ymax=1.3+0.05, color='gray', linestyle='solid', linewidth=1.0, zorder=7)
Axes_obj_02.vlines(x=Mean_pSD, ymin=1.3-0.05, ymax=1.3+0.05, color='gray', linestyle='solid', linewidth=1.0, zorder=7)

Axes_obj_02.vlines(x=Mean_m2SD, ymin=0.0, ymax=1.4, color='gray', linestyle='solid', linewidth=1.0, zorder=7)
Axes_obj_02.vlines(x=Mean_p2SD, ymin=0.0, ymax=1.4, color='gray', linestyle='solid', linewidth=1.0, zorder=7)

Axes_obj_02.vlines(x=Mean_m3SD, ymin=0.0, ymax=0.5, color='gray', linestyle='solid', linewidth=1.0, zorder=7)
Axes_obj_02.vlines(x=Mean_m3SD, ymin=0.7, ymax=1.4, color='gray', linestyle='solid', linewidth=1.0, zorder=7)
Axes_obj_02.vlines(x=Mean_p3SD, ymin=0.0, ymax=1.4, color='gray', linestyle='solid', linewidth=1.0, zorder=7)

Axes_obj_02.vlines(x=15, ymin=1.15, ymax=1.3+0.05, color='gray', linestyle='solid', linewidth=1.0, zorder=7)


Axes_obj_02.fill_between(np.linspace(Mean_p3SD, Mean_m3SD, 1000), 1.4*np.ones(1000), 0.0*np.ones(1000),
                         facecolor='blue', alpha=0.025, zorder=6)
Axes_obj_02.fill_between(np.linspace(Mean_p2SD, Mean_m2SD, 1000), 1.4*np.ones(1000), 0.0*np.ones(1000),
                         facecolor='blue', alpha=0.025, zorder=7)


Axes_obj_02.text((Mean_m3SD + Mean_m2SD)/2, 1.3+0.03, "SD",
                 horizontalalignment='center', fontsize=10, fontweight="normal", color="black", zorder=8)
Axes_obj_02.text((Mean_m2SD + Mean_mSD)/2, 1.3+0.03, "SD",
                 horizontalalignment='center', fontsize=10, fontweight="normal", color="black", zorder=8)
Axes_obj_02.text((Mean_mSD + 15)/2, 1.3+0.03, "SD",
                 horizontalalignment='center', fontsize=10, fontweight="normal", color="black", zorder=8)
Axes_obj_02.text((15 + Mean_pSD)/2, 1.3+0.03, "SD",
                 horizontalalignment='center', fontsize=10, fontweight="normal", color="black", zorder=8)
Axes_obj_02.text((Mean_pSD + Mean_p2SD)/2, 1.3+0.03, "SD",
                 horizontalalignment='center', fontsize=10, fontweight="normal", color="black", zorder=8)
Axes_obj_02.text((Mean_p2SD + Mean_p3SD)/2, 1.3+0.03, "SD",
                 horizontalalignment='center', fontsize=10, fontweight="normal", color="black", zorder=8)

Axes_obj_02.quiver((Mean_m3SD + Mean_m2SD)/2, 1.3, -2.7386/2, 0.0,
                   scale_units='xy', angles='xy', scale=1, linestyle='solid', linewidth=0.4079/1.25, width=0.004/1.25)
Axes_obj_02.quiver((Mean_m3SD + Mean_m2SD)/2, 1.3, 2.7386/2, 0.0,
                   scale_units='xy', angles='xy', scale=1, linestyle='solid', linewidth=0.4079/1.25, width=0.004/1.25)

Axes_obj_02.quiver((Mean_m2SD + Mean_mSD)/2, 1.3, -2.7386/2, 0.0,
                   scale_units='xy', angles='xy', scale=1, linestyle='solid', linewidth=0.4079/1.25, width=0.004/1.25)
Axes_obj_02.quiver((Mean_m2SD + Mean_mSD)/2, 1.3, 2.7386/2, 0.0,
                   scale_units='xy', angles='xy', scale=1, linestyle='solid', linewidth=0.4079/1.25, width=0.004/1.25)

Axes_obj_02.quiver((Mean_mSD + 15)/2, 1.3, -2.7386/2, 0.0,
                   scale_units='xy', angles='xy', scale=1, linestyle='solid', linewidth=0.4079/1.25, width=0.004/1.25)
Axes_obj_02.quiver((Mean_mSD + 15)/2, 1.3, 2.7386/2, 0.0,
                   scale_units='xy', angles='xy', scale=1, linestyle='solid', linewidth=0.4079/1.25, width=0.004/1.25)


Axes_obj_02.quiver((Mean_pSD + 15)/2, 1.3, -2.7386/2, 0.0,
                   scale_units='xy', angles='xy', scale=1, linestyle='solid', linewidth=0.4079/1.25, width=0.004/1.25)
Axes_obj_02.quiver((Mean_pSD + 15)/2, 1.3, 2.7386/2, 0.0,
                   scale_units='xy', angles='xy', scale=1, linestyle='solid', linewidth=0.4079/1.25, width=0.004/1.25)

Axes_obj_02.quiver((Mean_p2SD + Mean_pSD)/2, 1.3, -2.7386/2, 0.0,
                   scale_units='xy', angles='xy', scale=1, linestyle='solid', linewidth=0.4079/1.25, width=0.004/1.25)
Axes_obj_02.quiver((Mean_p2SD + Mean_pSD)/2, 1.3, 2.7386/2, 0.0,
                   scale_units='xy', angles='xy', scale=1, linestyle='solid', linewidth=0.4079/1.25, width=0.004/1.25)

Axes_obj_02.quiver((Mean_p3SD + Mean_p2SD)/2, 1.3, -2.7386/2, 0.0,
                   scale_units='xy', angles='xy', scale=1, linestyle='solid', linewidth=0.4079/1.25, width=0.004/1.25)
Axes_obj_02.quiver((Mean_p3SD + Mean_p2SD)/2, 1.3, 2.7386/2, 0.0,
                   scale_units='xy', angles='xy', scale=1, linestyle='solid', linewidth=0.4079/1.25, width=0.004/1.25)


Axes_obj_02.quiver(Mean_mSD, 1.3-0.07, -2.7386, 0.0,
                   scale_units='xy', angles='xy', scale=1, linestyle='solid', linewidth=0.4079, width=0.004)
Axes_obj_02.quiver(Mean_mSD, 1.3-0.07, 2.7386, 0.0,
                   scale_units='xy', angles='xy', scale=1, linestyle='solid', linewidth=0.4079, width=0.004)

Axes_obj_02.quiver(Mean_pSD, 1.3-0.07, -2.7386, 0.0,
                   scale_units='xy', angles='xy', scale=1, linestyle='solid', linewidth=0.4079, width=0.004)
Axes_obj_02.quiver(Mean_pSD, 1.3-0.07, 2.7386, 0.0,
                   scale_units='xy', angles='xy', scale=1, linestyle='solid', linewidth=0.4079, width=0.004)


Axes_obj_02.text(Mean_mSD, 1.15, "2×SD",
                 horizontalalignment='center', fontsize=10, fontweight="bold", color="black", zorder=8)
Axes_obj_02.text(Mean_pSD, 1.15, "2×SD",
                 horizontalalignment='center', fontsize=10, fontweight="bold", color="black", zorder=8)


############################################################################################################
# (c) Definition of Polygon that Should Be Identified in the Figure of Meta-regression Analysis
############################################################################################################
#Axes_obj_03.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, length=0.0)
Axes_obj_03.tick_params(labelbottom=False, length=0.0)
Axes_obj_03.set_xlim(-1.0, 23+12)
Axes_obj_03.set_ylim(0, 21)
Axes_obj_03.set_yticks([0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0])
Axes_obj_03.set_yticklabels(
    ["        ", "        ", "        ", "        ", "        ", "        ", "        ", "        ", "        ", ])

Axes_obj_03.plot(x_ideal, y_ideal, linewidth=1.6318, color='blue')
Axes_obj_03.plot(x_ideal, y_ideal_u, linewidth=1.5, color='red')
Axes_obj_03.plot(x_ideal, y_ideal_l, linewidth=1.5, color='red')

Axes_obj_03.scatter([x_ideal2[1], x_ideal2[2], x_ideal2[3]], [y_ideal2[1], y_ideal2[2], y_ideal2[3]],
                    linewidth=1.6318, color='black', zorder=10)
Axes_obj_03.scatter([x_ideal2[1], x_ideal2[2], x_ideal2[3]], [y_ideal2_u[1], y_ideal2_u[2], y_ideal2_u[3]],
                    linewidth=1.6318, color='black', marker='_', zorder=9)
Axes_obj_03.scatter([x_ideal2[1], x_ideal2[2], x_ideal2[3]], [y_ideal2_l[1], y_ideal2_l[2], y_ideal2_l[3]],
                    linewidth=1.6318, color='black', marker='_', zorder=9)

for i in range(1, len(x_ideal2)-1):
    Axes_obj_03.vlines(x=x_ideal2[i], ymin=y_ideal2_l[i], ymax=y_ideal2_u[i],
                       color='black', linestyle='solid', linewidth=1.6318, zorder=9)

x_ideal = np.linspace(0, 10, 1000)
y_ideal = 7.5 / (1 + np.exp(-2.0 * (x_ideal - 5))) + 6.25

diff_y_ideal = (7.5*2.0*np.exp(-2.0*(x_ideal-5)))/(np.exp(-2.0*(x_ideal-5))+1)**2
Axes_obj_03.plot(x_ideal, diff_y_ideal, color='black', linestyle='dashed', linewidth=1.0, zorder=9)

Axes_obj_03.vlines(x=x_ideal[0], ymin=y_ideal2_l[0], ymax=y_ideal2_u[0], color='black', linestyle='dashed', linewidth=0.5)
Axes_obj_03.vlines(x=x_ideal[-1], ymin=y_ideal2_l[4], ymax=y_ideal2_u[4], color='black', linestyle='dashed', linewidth=0.5)


Axes_obj_03.scatter([0.0], [19.0], color='red', marker='_', zorder=9)
Axes_obj_03.scatter([0.0], [17.0], color='blue', zorder=9)
Axes_obj_03.scatter([0.0], [15.0], color='red', marker='_', zorder=9)
Axes_obj_03.vlines(x=0.0, ymin=17.0, ymax=19.0, color='gray', linestyle='solid', linewidth=1.6318)
Axes_obj_03.vlines(x=0.0, ymin=15.0, ymax=17.0, color='gray', linestyle='solid', linewidth=1.6318)

###################################################################################################
# Equations
###################################################################################################

Axes_obj_03.text(10.5+2.0, y_ideal2_u[4],
    r"$g_{1}(x) = \left[\ \frac{K_{1}}{1+\exp{\{-L(x-M)\}}}+K_{2}\ \right]+\left(\ ax^2+bx+c\ \right)$",
    horizontalalignment='left', fontsize=14.6868,  fontweight="normal")
Axes_obj_03.text(11.35+2.0-0.5, np.max(y_ideal),
    r"$f(x) = \quad\frac{K_{1}}{1+\exp{\{-L(x-M)\}}}+K_{2}$",
    horizontalalignment='left', fontsize=14.6868,  fontweight="normal")
Axes_obj_03.text(10.5+2.0, y_ideal2_l[4],
    r"$g_{2}(x) = \left[\ \frac{K_{1}}{1+\exp{\{-L(x-M)\}}}+K_{2}\ \right]-\left(\ ax^2+bx+c\ \right)$",
    horizontalalignment='left', fontsize=14.6868,  fontweight="normal")
Axes_obj_03.text(6.0, np.max(diff_y_ideal),
    r"$\frac{df(x)}{dx}=\frac{K_{1}L\exp{ \{-L(x - M)\} }}{[\exp{ \{-L(x - M) \} }+1]^2}$",
    horizontalalignment='left', fontsize=14.6868,  fontweight="normal")

Axes_obj_03.text(17.5, 2.5,
    r"$ax^2+bx+c = a\left(x+{\frac {b}{2a}}\right)^{2}-{\frac{b^{2}-4ac}{4a}},$",
    horizontalalignment='left', fontsize=14.6868,  fontweight="normal")

Axes_obj_03.text(30.5, 2.5,
    r"$M=-\frac{b}{2a}$",
    horizontalalignment='left', fontsize=14.6868,  fontweight="normal")

###################################################################################################
Axes_obj_03.fill_between(x_ideal, y_ideal_u, y_ideal_l, facecolor='red', alpha=0.1)

Axes_obj_03.fill_between(np.linspace(15.0, 22.75, 1000), (20.15+0.2)*np.ones(1000), (17.05-0.2)*np.ones(1000),
                         facecolor='blue', alpha=0.1)
Axes_obj_03.fill_between(np.linspace(15.0, 22.75, 1000), (15.60+0.2)*np.ones(1000), (12.5-0.2)*np.ones(1000),
                         facecolor='blue', alpha=0.1)
Axes_obj_03.fill_between(np.linspace(15.0, 22.75, 1000), (11.15+0.2)*np.ones(1000), (8.05-0.2)*np.ones(1000),
                         facecolor='blue', alpha=0.1)


Axes_obj_03.fill_between(np.linspace(23.4, 28.75, 1000), (19.25+0.3)*np.ones(1000), (17.90-0.2)*np.ones(1000),
                         facecolor='red', alpha=0.1)
Axes_obj_03.fill_between(np.linspace(23.4, 28.75, 1000), (10.25+0.3)*np.ones(1000), (8.90-0.2)*np.ones(1000),
                         facecolor='red', alpha=0.1)

Axes_obj_03.fill_between(np.linspace(17.3, 21.8, 1000), 3.95*np.ones(1000), 2.1*np.ones(1000),
                         facecolor='red', alpha=0.1)

Axes_obj_03.fill_between(np.linspace(24.355, 25.355, 1000), 1.3*np.ones(1000), 4.7*np.ones(1000),
                         facecolor='yellow', alpha=0.5)

Axes_obj_03.fill_between(np.linspace(32.7125, 33.555, 1000), 1.3*np.ones(1000), 4.7*np.ones(1000),
                         facecolor='yellow', alpha=0.5)

Axes_obj_03.quiver(12, y_ideal2_u[4], -1.0, 0.0,
                   scale_units='xy', angles='xy', scale=1, linestyle='solid', linewidth=0.4079, width=0.004)
Axes_obj_03.quiver(12, np.max(y_ideal), -1.0, 0.0,
                   scale_units='xy', angles='xy', scale=1, linestyle='solid', linewidth=0.4079, width=0.004)
Axes_obj_03.quiver(12, y_ideal2_l[4], -1.0, 0.0,
                   scale_units='xy', angles='xy', scale=1, linestyle='solid', linewidth=0.4079, width=0.004)

Axes_obj_03.quiver(8.5, 0.75, -1.0, 0.0,
                   scale_units='xy', angles='xy', scale=1, linestyle='solid', linewidth=0.4079, width=0.004)

Axes_obj_03.quiver(24.0, np.max(y_ideal)+0.4, -1.0, 0.0,
                   scale_units='xy', angles='xy', scale=1, linestyle='solid', linewidth=0.4079, width=0.004, color="blue")


Axes_obj_03.quiver(1.25, 19, -0.75, 0.0,
                   scale_units='xy', angles='xy', scale=1, linestyle='solid', color="red", linewidth=0.4079, width=0.003)
Axes_obj_03.quiver(1.25, 17, -0.75, 0.0,
                   scale_units='xy', angles='xy', scale=1, linestyle='solid', color="blue",  linewidth=0.4079, width=0.003)
Axes_obj_03.quiver(1.25, 15, -0.75, 0.0,
                   scale_units='xy', angles='xy', scale=1, linestyle='solid', color="red",  linewidth=0.4079, width=0.003)
############################################################################################################

############################################################################################################
Figure_object.tight_layout()

my_path = os.path.abspath("Figures")
plt.savefig(os.path.join(my_path, "00_Kimoto_et_al_(2023)_[_Fig_01_].png"))

from PIL import Image # py -m pip install pillow
img = Image.open(os.path.join(my_path, "00_Kimoto_et_al_(2023)_[_Fig_01_].png"))
img_resize = img.resize(size=(2866, 2016))
my_path2 = os.path.abspath("Figures/Fixed_Version/B6")
img_resize.save(os.path.join(my_path2, "00_Kimoto_et_al_(2023)_[_Fig_01_]_B6.png"))
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

############################################################################################################


