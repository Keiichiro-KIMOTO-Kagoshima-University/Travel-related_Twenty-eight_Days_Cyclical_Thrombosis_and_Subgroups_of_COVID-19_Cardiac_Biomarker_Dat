########################################################################################################################
# Fig. S2. Finding shapes using a “key” to search and “latent patient population” onset with a strength of risk.
# (01_Kimoto_et_al_(2023)_[Fig_S_02])
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
# Contents
########################################################################################################################
# List of Import
# Information of Figures
# Data Sets Preparation Step: Figures (a) , (b), and (c)
# Plot Step
# Data Sets Preparation Step: Figures (d) , (e), and (f)
# Plot Step
# Data Sets Preparation Step: Figures (g) , (h), and (i)
# Plot Step
########################################################################################################################
# List of Import
########################################################################################################################
from typing import List
import pandas as pd
import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import gridspec
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
# Information of Figures
########################################################################################################################
Figure_object = plt.figure(1, figsize=(11.69, 8.27), dpi=400, edgecolor="black", linewidth=0.5)
#plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
plt.subplots_adjust(left=0.125, bottom=0.11, right=0.9, top=0.88, wspace=0.2, hspace=0.2)

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

Axes_obj_01 = plt.axes([0.0128315+0.0225, 0.68811, 0.28657, 0.267956])
Axes_obj_02 = plt.axes([0.356715, 0.68811, 0.28657, 0.267956])
Axes_obj_03 = plt.axes([0.700599-0.0225, 0.68811, 0.28657, 0.267956])
Axes_obj_04 = plt.axes([0.0128315+0.0225, 0.353124, 0.28657, 0.267956])
Axes_obj_05 = plt.axes([0.356715, 0.353124, 0.28657, 0.267956])
Axes_obj_06 = plt.axes([0.700599-0.0225, 0.353124, 0.28657, 0.267956])
Axes_obj_07 = plt.axes([0.0128315+0.0225, 0.0181378, 0.28657, 0.267956])
Axes_obj_08 = plt.axes([0.356715, 0.0181378, 0.28657, 0.267956])
Axes_obj_09 = plt.axes([0.700599-0.0225, 0.0181378, 0.28657, 0.267956])

plt.figtext(0.0128315+0.0225-0.005, 0.68811+0.267956+0.0125, "a", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")
plt.figtext(0.356715-0.005, 0.68811+0.267956+0.0125, "b", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")
plt.figtext(0.700599-0.0225-0.005, 0.68811+0.267956+0.0125, "c", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")

plt.figtext(0.0128315+0.0225-0.005, 0.353124+0.267956+0.0125, "d", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")
plt.figtext(0.356715-0.005, 0.353124+0.267956+0.0125, "e", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")
plt.figtext(0.700599-0.0225-0.005, 0.353124+0.267956+0.0125, "f", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")

plt.figtext(0.0128315+0.0225-0.005, 0.0181378+0.267956+0.0125, "g", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")
plt.figtext(0.356715-0.005, 0.0181378+0.267956+0.0125, "h", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")
plt.figtext(0.700599-0.0225-0.005, 0.0181378+0.267956+0.0125, "i", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")

########################################################################################################################
#    Data Sets Preparation Step: Figures (a) , (b), and (c)
########################################################################################################################
# ORIGINAL DATA
Circle_X = [0.00,  -3.5,  -5.00,  -3.50,  0.0,  3.50,  5.00,  3.5]
Circle_Y = [5.00,  3.5,  0.00,  -3.50,  -5.0,  -3.50,  0.00,  3.5]

Triangle_X = [2.50,  -3.0,  8.00]
Triangle_Y = [10.00,  0.0,  0.00]

Square_X = [-6.00,  -6.0,  3.00,  3.0]
Square_Y = [7.00,  -2.0,  -2.00,  7.0]

ALL_X = [Circle_X[0], Circle_X[1], Circle_X[2], Circle_X[3], Circle_X[4], Circle_X[5], Circle_X[6], Circle_X[7],
 Triangle_X[0], Triangle_X[1], Triangle_X[2],
     Square_X[0], Square_X[1], Square_X[2], Square_X[3]]

ALL_Y = [Circle_Y[0], Circle_Y[1], Circle_Y[2], Circle_Y[3], Circle_Y[4], Circle_Y[5], Circle_Y[6], Circle_Y[7],
 Triangle_Y[0], Triangle_Y[1], Triangle_Y[2],
     Square_Y[0], Square_Y[1], Square_Y[2], Square_Y[3]]

# ORIGINAL DATA (for plot)
Circle_X_P = [0.00,  -3.5,  -5.00,  -3.50,  0.0,  3.50,  5.00,  3.5, 0.00]
Circle_Y_P = [5.00,  3.5,  0.00,  -3.50,  -5.0,  -3.50,  0.00,  3.5, 5.00]

Triangle_X_P = [2.50,  -3.0,  8.00, 2.50]
Triangle_Y_P = [10.00,  0.0,  0.00, 10.00]

Square_X_P = [-6.00,  -6.0,  3.00,  3.0, -6.00]
Square_Y_P = [7.00,  -2.0,  -2.00,  7.0, 7.00]

ALL_X_P = [Circle_X[0], Circle_X[1], Circle_X[2], Circle_X[3], Circle_X[4], Circle_X[5], Circle_X[6], Circle_X[7],
 Triangle_X[0], Triangle_X[1], Triangle_X[2],
     Square_X[0], Square_X[1], Square_X[2], Square_X[3],
           Circle_X[0]]

ALL_Y_P = [Circle_Y[0], Circle_Y[1], Circle_Y[2], Circle_Y[3], Circle_Y[4], Circle_Y[5], Circle_Y[6], Circle_Y[7],
 Triangle_Y[0], Triangle_Y[1], Triangle_Y[2],
     Square_Y[0], Square_Y[1], Square_Y[2], Square_Y[3],
           Circle_Y[0]]

# Circle DATA
X = np.linspace(-5, 5, 100)
Y1 = np.sqrt(25-X**2)
Y2 = -np.sqrt(25-X**2)

X_A = np.linspace(-1.25, 1.25, 100)
Y1_A = np.sqrt(25-X_A**2)
Y2_A = -np.sqrt(25-X_A**2)

X_B = np.linspace(-5, -4.8125, 100)
Y1_B = np.sqrt(25-X_B**2)
Y2_B = -np.sqrt(25-X_B**2)

X_C = np.linspace(4.8125, 5, 100)
Y1_C = np.sqrt(25-X_C**2)
Y2_C = -np.sqrt(25-X_C**2)

# Crue DATA
Square_X2 = [-4, -6, -6, -4, 1, 3, 3, 1]
Square_Y2 = [7, 5, 0, -2, -2, 0, 5, 7]

Triangle_X2 = [1.4, -1.9, -1, 6, 6.9, 3.6]
Triangle_Y2 = [8, 2, 0, 0, 2, 8]

########################################################################################################################
#    Plot Step:
########################################################################################################################

Axes_obj_01.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
Axes_obj_01.set_title("Scatter Plot (Data): [Question]", fontweight="bold")
Axes_obj_01.set_xlabel('')
Axes_obj_01.set_ylabel('')
Axes_obj_01.tick_params(length=0.0)
Axes_obj_01.set_xlim(-8.5, 10.0)
Axes_obj_01.set_ylim(-7.5, 11.0)
Axes_obj_01.scatter(ALL_X, ALL_Y, color='black')

Axes_obj_02.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
Axes_obj_02.set_title("Hint & Easy Reasoning", fontweight="bold")
Axes_obj_02.set_xlabel('')
Axes_obj_02.set_ylabel('')
Axes_obj_02.tick_params(length=0.0)
Axes_obj_02.set_xlim(-8.5, 10.0)
Axes_obj_02.set_ylim(-7.5, 11.0)

Axes_obj_02.plot(X_A, Y1_A, color='black')
Axes_obj_02.plot(X_A, Y2_A, color='black')
Axes_obj_02.plot(X_B, Y1_B, color='black')
Axes_obj_02.plot(X_B, Y2_B, color='black')
Axes_obj_02.plot(X_C, Y1_C, color='black')
Axes_obj_02.plot(X_C, Y2_C, color='black')

Axes_obj_02.plot([Triangle_X[0], Triangle_X2[0]], [Triangle_Y[0], Triangle_Y2[0]], color='black')
Axes_obj_02.plot([Triangle_X[1], Triangle_X2[1]], [Triangle_Y[1], Triangle_Y2[1]], color='black')
Axes_obj_02.plot([Triangle_X[1], Triangle_X2[2]], [Triangle_Y[1], Triangle_Y2[2]], color='black')
Axes_obj_02.plot([Triangle_X[2], Triangle_X2[3]], [Triangle_Y[2], Triangle_Y2[3]], color='black')
Axes_obj_02.plot([Triangle_X[2], Triangle_X2[4]], [Triangle_Y[2], Triangle_Y2[4]], color='black')
Axes_obj_02.plot([Triangle_X[0], Triangle_X2[5]], [Triangle_Y[0], Triangle_Y2[5]], color='black')

Axes_obj_02.plot([Square_X[0], Square_X2[0]], [Square_Y[0], Square_Y2[0]], color='black')
Axes_obj_02.plot([Square_X[0], Square_X2[1]], [Square_Y[0], Square_Y2[1]], color='black')
Axes_obj_02.plot([Square_X[1], Square_X2[2]], [Square_Y[1], Square_Y2[2]], color='black')
Axes_obj_02.plot([Square_X[1], Square_X2[3]], [Square_Y[1], Square_Y2[3]], color='black')
Axes_obj_02.plot([Square_X[2], Square_X2[4]], [Square_Y[2], Square_Y2[4]], color='black')
Axes_obj_02.plot([Square_X[2], Square_X2[5]], [Square_Y[2], Square_Y2[5]], color='black')
Axes_obj_02.plot([Square_X[3], Square_X2[6]], [Square_Y[3], Square_Y2[6]], color='black')
Axes_obj_02.plot([Square_X[3], Square_X2[7]], [Square_Y[3], Square_Y2[7]], color='black')

Axes_obj_02.scatter(ALL_X, ALL_Y, color='black')

Axes_obj_03.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
Axes_obj_03.set_title("Layered Diagrams: [Answer]", fontweight="bold")
Axes_obj_03.set_xlabel('')
Axes_obj_03.set_ylabel('')
Axes_obj_03.tick_params(length=0.0)
Axes_obj_03.set_xlim(-8.5, 10.0)
Axes_obj_03.set_ylim(-7.5, 11.0)
Axes_obj_03.plot(Triangle_X_P, Triangle_Y_P, color='black')
Axes_obj_03.plot(Square_X_P, Square_Y_P, color='black')
Axes_obj_03.plot(X, Y1, color='black')
Axes_obj_03.plot(X, Y2, color='black')
Axes_obj_03.scatter(ALL_X, ALL_Y, color='black')


########################################################################################################################
#    Data Sets Preparation Step: Figures (d) , (e), and (f)
########################################################################################################################
data00=[-2.02707197, -1.67179528, -1.42492352, -1.23211820, -1.07161494, -0.93235428, -0.80795031, -0.69440700,
       -0.58897703, -0.48970523, -0.39506762, -0.30393331, -0.21539531, -0.12856502, -0.04274434, 0.04274434,
       0.12856502, 0.21539531, 0.30393331, 0.39506762, 0.48970523, 0.58897703, 0.69440700, 0.80795031, 0.93235428,
       1.07161494, 1.23211820, 1.42492352, 1.67179528, 2.02707197]

data01 = np.linspace(4, 4, len(data00))
data1 = [x + y for (x, y) in zip(data00, data01)]
data2 = np.linspace(1.0, 1.0, len(data1))
data3 = np.linspace(0.25, 0.25, len(data1))

X = np.linspace(-3.5+4.0, 3.5+4.0, 100)
X2 = np.linspace(-1.25+4.0, 1.25+4.0, 100)
Y1 = 1/(1+np.exp(-1.452555*(X-4.0)))
Y2 = (1.452555*np.exp(-1.452555*(X-4.0)))/(np.exp(-1.452555*(X-4.0))+1)**2

X0 = 0+4.0
A = (1.452555*np.exp(-1.452555*(X0-4.0)))/(np.exp(-1.452555*(X0-4.0))+1)**2
Y3 = (A*32.5)*(X2-4.0) + (1/(1+np.exp(-1.452555*(X0-4.0))))*32.5


########################################################################################################################
#    Plot Step:
########################################################################################################################
Axes_obj_04.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
Axes_obj_04.set_title(' \nDistribution of Sensitivity', fontweight="bold")
#Axes_obj_04.set_xlabel('1/Sensitivity (Amount of Exposure)')
#Axes_obj_04.set_ylabel('Frequency of Cases')
Axes_obj_04.set_xlabel('')
Axes_obj_04.set_ylabel('')
Axes_obj_04.tick_params(length=0.0)

Axes_obj_04.set_xlim(-3.75+4.0, 3.75+4.0)
Axes_obj_04.set_ylim(0.0, 10.0)


Axes_obj_04.hist(data1, bins=12, range=(-3.5+4.0, 3.5+4.0), ec='black',  alpha=0.125, rwidth=1.0, cumulative=False,
                 align='mid', orientation='vertical', log=False, color="red", label=False)
Axes_obj_04.plot(X, 17.5*Y2, color='red', linewidth=2.5)
Axes_obj_04.scatter(data1, data3, marker='o', facecolors='none', edgecolors='red')

Axes_obj_04.text(1.5, 4.25, 'High\nSensitivity', color="black", size=11, ha='center', va='center')
Axes_obj_04.text(6.5, 4.25, 'Low\nSensitivity', color="black", size=11, ha='center', va='center')
Axes_obj_04.text(4.0, 8.0, '"Homogeneous" Population\n (One Peak Mountain)', color="black", size=11,
                 ha='center', va='center')

########################################################################################################################
Axes_obj_05.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
Axes_obj_05.set_title('Integration of Distribution', fontweight="bold")
#Axes_obj_05.set_xlabel('1/Sensitivity (Amount of Exposure)')
#Axes_obj_05.set_ylabel('Cumulative Number')
#Axes_obj_05.set_xlabel('')
Axes_obj_05.set_ylabel('')
Axes_obj_05.tick_params(length=0.0)
Axes_obj_05.set_xlim(-3.75+4.0, 3.75+4.0)
Axes_obj_05.set_ylim(0.0, 40.0)

Axes_obj_05.hist(data1, bins=12, range=(-3.5+4.0, 3.5+4.0), ec='black', alpha=0.125, rwidth=1.0, cumulative=True,
                 align='mid', orientation='vertical', log=False, color="red", label=True)
Axes_obj_05.plot(X, 32.5*Y1, color="red", linewidth=2.5)
#Axes_obj_05.plot(X2, Y3, color='black', linestyle='dashed')
Axes_obj_05.plot(X, 17.5*Y2, color='red', linewidth=1.0)

Axes_obj_05.text(0.0+4.0, 35.0, 'Cumulative Number of Thromboembolism\nwith Increase of Exposure',
                 color="black", size=11, ha='center', va='center')

Axes_obj_05.text(2.125+0.2, 27.5, 'Exponential curve part\n(the part log-linearity)', color="black",
                 size=11.0, ha='center', va='center')

Axes_obj_05.quiver(2.0+0.25, 22.5, 2.0-0.25, 0.0, color="black", angles='xy', scale_units='xy', scale=1)
Axes_obj_05.quiver(2.0+0.25, 22.5, -2.0, 0.0, color="black", angles='xy', scale_units='xy', scale=1)

Axes_obj_05.vlines(x=2.0+0.25+2.0-0.25, ymin=0.0, ymax=22.5, color="black", linestyle='dashed', linewidth=1.0)

########################################################################################################################
Axes_obj_06.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
Axes_obj_06.set_title('Two Distributions (Sub-groups)', fontweight="bold")
#Axes_obj_06.set_xlabel('1/Sensitivity (Amount of Exposure)')
#Axes_obj_06.set_ylabel('Cumulative Number')
Axes_obj_06.set_xlabel('')
Axes_obj_06.set_ylabel('')
Axes_obj_06.tick_params(length=0.0)

Axes_obj_06.set_xlim(-3.75+4.0, 12.5)
Axes_obj_06.set_ylim(0.0, 35.0)

Axes_obj_06.hist(data1, bins=12, range=(0.5, 12.5), ec='black', alpha=0.1, rwidth=1.0, cumulative=True,
                 align='mid', orientation='vertical', log=False, color="red", label=True)

Axes_obj_06.plot(X, 32.5*Y1, color='red', linewidth=1.0)
Axes_obj_06.plot(X, 17.5*Y2, color='red', linewidth=1.0)

data1_2 = [x + 4.0 for x in data1]

Axes_obj_06.hist(data1_2, bins=12, range=(0.5, 12.5), ec='black', alpha=0.1, rwidth=1.0, cumulative=True,
                 align='mid', orientation='vertical', log=False, color="blue", label=True)

Axes_obj_06.plot(X+4.0, 32.5*Y1, color='blue', linewidth=1.0)
Axes_obj_06.plot(X+4.0, 17.5*Y2, color='blue', linewidth=1.0)

Points_1_x = [3.0, 4.0, 5.0]
Points_1_y = [32.5*(1/(1+np.exp(-1.452555*(Points_1_x[0]-4.0)))),
              32.5*(1/(1+np.exp(-1.452555*(Points_1_x[1]-4.0)))),
              32.5*(1/(1+np.exp(-1.452555*(Points_1_x[2]-4.0))))]

Points_2_x = [7.0, 8.0, 9.0]
Points_2_y = [32.5*(1/(1+np.exp(-1.452555*(Points_2_x[0]-8.0)))),
              32.5*(1/(1+np.exp(-1.452555*(Points_2_x[1]-8.0)))),
              32.5*(1/(1+np.exp(-1.452555*(Points_2_x[2]-8.0))))]

Points_1_y_U = [Points_1_y[0]+5.5, Points_1_y[1]+5.5/3, Points_1_y[2]+5.5]
Points_1_y_L = [Points_1_y[0]-5.5, Points_1_y[1]-5.5/3, Points_1_y[2]-5.5]
Points_2_y_U = [Points_2_y[0]+5.5, Points_2_y[1]+5.5/3, Points_2_y[2]+5.5]
Points_2_y_L = [Points_2_y[0]-5.5, Points_2_y[1]-5.5/3, Points_2_y[2]-5.5]

Axes_obj_06.scatter(Points_1_x, Points_1_y, color='red')
Axes_obj_06.scatter(Points_2_x, Points_2_y, color='blue')

Axes_obj_06.scatter(Points_1_x[0], Points_1_y_U[0], color="red", marker='_')
Axes_obj_06.scatter(Points_1_x[1], Points_1_y_U[1], color="red", marker='_')
Axes_obj_06.scatter(Points_1_x[2], Points_1_y_U[2], color="red", marker='_')

Axes_obj_06.scatter(Points_1_x[0], Points_1_y_L[0], color="red", marker='_')
Axes_obj_06.scatter(Points_1_x[1], Points_1_y_L[1], color="red", marker='_')
Axes_obj_06.scatter(Points_1_x[2], Points_1_y_L[2], color="red", marker='_')

Axes_obj_06.text(Points_1_x[0]-0.75, Points_1_y_U[0]-2.0, "C.I.\nwide", color="black", size=10, ha='center', va='center')
Axes_obj_06.text(Points_1_x[1]-1.25, Points_1_y_U[1]+2.0, "C.I.\nnarrow", color="black", size=10, ha='center', va='center')
Axes_obj_06.text(Points_1_x[2]-0.75, Points_1_y_U[2]-2.0, "C.I.\nwide", color="black", size=10, ha='center', va='center')

Axes_obj_06.vlines(x=Points_1_x[0],
                   ymin=Points_1_y_L[0], ymax=Points_1_y_U[0], color="red", linestyle='solid', linewidth=1.5)
Axes_obj_06.vlines(x=Points_1_x[1],
                   ymin=Points_1_y_L[1], ymax=Points_1_y_U[1], color="red", linestyle='solid', linewidth=1.5)
Axes_obj_06.vlines(x=Points_1_x[2],
                   ymin=Points_1_y_L[2], ymax=Points_1_y_U[2], color="red", linestyle='solid', linewidth=1.5)

Axes_obj_06.scatter(Points_2_x[0], Points_2_y_U[0], color="blue", marker='_')
Axes_obj_06.scatter(Points_2_x[1], Points_2_y_U[1], color="blue", marker='_')
Axes_obj_06.scatter(Points_2_x[2], Points_2_y_U[2], color="blue", marker='_')

Axes_obj_06.scatter(Points_2_x[0], Points_2_y_L[0], color="blue", marker='_')
Axes_obj_06.scatter(Points_2_x[1], Points_2_y_L[1], color="blue", marker='_')
Axes_obj_06.scatter(Points_2_x[2], Points_2_y_L[2], color="blue", marker='_')

Axes_obj_06.text(Points_2_x[0]-0.75, Points_2_y_U[0]-2.0,
                 "C.I.\nwide", color="black", size=10, ha='center', va='center')
Axes_obj_06.text(Points_2_x[1]-1.25, Points_2_y_U[1]+2.0,
                 "C.I.\nnarrow", color="black", size=10, ha='center', va='center')
Axes_obj_06.text(Points_2_x[2]-0.75, Points_2_y_U[2]-2.0,
                 "C.I.\nwide", color="black", size=10, ha='center', va='center')

Axes_obj_06.vlines(x=Points_2_x[0],
                   ymin=Points_2_y_L[0], ymax=Points_2_y_U[0] , color="blue", linestyle='solid', linewidth=1.5)
Axes_obj_06.vlines(x=Points_2_x[1],
                   ymin=Points_2_y_L[1], ymax=Points_2_y_U[1] , color="blue", linestyle='solid', linewidth=1.5)
Axes_obj_06.vlines(x=Points_2_x[2],
                   ymin=Points_2_y_L[2], ymax=Points_2_y_U[2] , color="blue", linestyle='solid', linewidth=1.5)

########################################################################################################################
#    Data Sets Preparation Step: Figures (g) , (h), and (i)
########################################################################################################################
# Sigmoid Curve
# Circle DATA

X = np.linspace(-3, 3, 100)
Line_fill1 = -5.0*np.ones(100)
Y0 = 2.0*(1.0 / (1 + np.exp(-2.5 * (X - 0.0))) - 0.5)
YU = 0.25*(X - 0.0)**2 + 1.0
YL = - 0.25*(X - 0.0)**2 - 1.0
Y0_U = 2.0*(1.0 / (1.0 + np.exp(-2.5 * (X - 0.0)))-0.5) + 0.25*(X - 0.0)**2 + 1.0
Y0_L = 2.0*(1.0 / (1.0 + np.exp(-2.5 * (X - 0.0)))-0.5) - 0.25*(X - 0.0)**2 - 1.0
Y0_diff = (5.0*np.exp(-2.5*X))/(np.exp(-2.5*X)+1)**2

X2 = np.linspace(-3, 3, 30)
Line_fill2 = -5.0*np.ones(30)
YU_2 = 0.25*(X2 - 0.0)**2 + 1.0
YL_2 = - 0.25*(X2 - 0.0)**2 - 1.0

Data_YU1 = [0.25*((-2) - 0.0)**2 + 1.0, 0.25*(0 - 0.0)**2 + 1.0, 0.25*(2 - 0.0)**2 + 1.0]
Data_YL1 = [- 0.25*((-2) - 0.0)**2 - 1.0, - 0.25*(0 - 0.0)**2 - 1.0, - 0.25*(2 - 0.0)**2 - 1.0]

Data_YU2 = [2.0*(1.0 / (1.0 + np.exp(-2.5 * ((-2) - 0.0)))-0.5) + 0.25*((-2) - 0.0)**2 + 1.0,
           2.0*(1.0 / (1.0 + np.exp(-2.5 * (0 - 0.0)))-0.5) + 0.25*(0 - 0.0)**2 + 1.0,
           2.0*(1.0 / (1.0 + np.exp(-2.5 * (2 - 0.0)))-0.5) + 0.25*(2 - 0.0)**2 + 1.0]

Data_YL2 = [2.0*(1.0 / (1.0 + np.exp(-2.5 * ((-2) - 0.0)))-0.5) - 0.25*((-2) - 0.0)**2 - 1.0,
           2.0*(1.0 / (1.0 + np.exp(-2.5 * (0 - 0.0)))-0.5) - 0.25*(0 - 0.0)**2 - 1.0,
           2.0*(1.0 / (1.0 + np.exp(-2.5 * (2 - 0.0)))-0.5) - 0.25*(2 - 0.0)**2 - 1.0]
########################################################################################################################
#    Plot Step:
########################################################################################################################

Axes_obj_07.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, length=0.0)
Axes_obj_07.set_title("Sigmoid Curve (S Curve)", fontweight="bold")
Axes_obj_07.set_xlabel('')
Axes_obj_07.set_ylabel('')
Axes_obj_07.set_xlim(-3.0, 3.0)
Axes_obj_07.set_ylim(-5.0, 5.0)

Axes_obj_07.plot(X, Y0, color='red')
Axes_obj_07.plot(X, Y0_diff-5.0, color='red')
Axes_obj_07.scatter([0], [0], color='black', s=20)
Axes_obj_07.fill_between(X, Y0_diff-5.0, Line_fill1, facecolor='red', alpha=0.2)

Axes_obj_07.quiver(0.0, -3.0, 0.0, 8.0, color="black", angles='xy', scale_units='xy', scale=1,
                   headwidth=3*1.5, headlength=5*1.5, headaxislength=4.5*1.5)
Axes_obj_07.quiver(-3.0, 0.0, 6.0, 0.0, color="black", angles='xy', scale_units='xy', scale=1,
                   headwidth=3*1.5, headlength=5*1.5, headaxislength=4.5*1.5)
Axes_obj_07.text(0.125, -0.5-0.25, "O", color="black", size=12.5)
Axes_obj_07.text(3-0.35, 0+0.275, "X", color="black", size=12.5)
Axes_obj_07.text(0+0.15, 5-1.0, "Y", color="black", size=12.5)
#Axes_obj_07.text(1.5, 2.5, "Y=f(X)", color="black", size=12.5, ha='center', va='center')


########################################################################################################################
Axes_obj_08.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, length=0.0)
Axes_obj_08.set_title("Parabola (U Shape) of Interval", fontweight="bold")
Axes_obj_08.set_xlabel('')
Axes_obj_08.set_ylabel('')
Axes_obj_08.set_xlim(-3.0, 3.0)
Axes_obj_08.set_ylim(-5.0, 5.0)

Axes_obj_08.plot(X, YU, color='red', linewidth=0.5)
Axes_obj_08.plot(X, YL, color='red', linewidth=0.5)
Axes_obj_08.plot(X, Y0_diff-5.0, color='red')
Axes_obj_08.fill_between(X, YU, YL, facecolor='red', alpha=0.05)
Axes_obj_08.fill_between(X, Y0_diff-5.0, Line_fill1, facecolor='red', alpha=0.2)

Axes_obj_08.scatter([0], [0], color='black', s=20)

Axes_obj_08.scatter([-2, 0, 2], [0, 0, 0], color='red', s=50, zorder=2)
Axes_obj_08.scatter([-2, 0, 2], Data_YU1, color="red", marker='_', s=50)
Axes_obj_08.scatter([-2, 0, 2], Data_YL1, color="red", marker='_', s=50)
Axes_obj_08.vlines(x=-2, ymin=Data_YL1[0], ymax=Data_YU1[0], color="red", linestyle='solid', linewidth=2.5)
Axes_obj_08.vlines(x=0, ymin=Data_YL1[1], ymax=Data_YU1[1], color="red", linestyle='solid', linewidth=2.5)
Axes_obj_08.vlines(x=2, ymin=Data_YL1[2], ymax=Data_YU1[2], color="red", linestyle='solid', linewidth=2.5)

Axes_obj_08.text(-2, Data_YU1[0]+1.0,
                 "C.I.\nwide", color="black", size=10, ha='center', va='center')
Axes_obj_08.text(0-0.5, Data_YU1[1]+1.0,
                 "C.I.\nnarrow", color="black", size=10, ha='center', va='center')
Axes_obj_08.text(2, Data_YU1[2]+1.0,
                 "C.I.\nwide", color="black", size=10, ha='center', va='center')

Axes_obj_08.text(-2, -4.5, "N=Small", color="black", size=10, ha='center', va='center')
Axes_obj_08.text(0, -3.35, "N=Large", color="black", size=10, ha='center', va='center')
Axes_obj_08.text(2, -4.5, "N=Small", color="black", size=10, ha='center', va='center')

Axes_obj_08.quiver(0.0, -3.0, 0.0, 8.0, color="black", angles='xy', scale_units='xy', scale=1,
                   headwidth=3*1.5, headlength=5*1.5, headaxislength=4.5*1.5)
Axes_obj_08.quiver(-3.0, 0.0, 6.0, 0.0, color="black", angles='xy', scale_units='xy', scale=1,
                   headwidth=3*1.5, headlength=5*1.5, headaxislength=4.5*1.5)
Axes_obj_08.text(0.125, -0.5-0.25, "O", color="black", size=12.5)
Axes_obj_08.text(3-0.35, 0+0.275, "X", color="black", size=12.5)
Axes_obj_08.text(0+0.15, 5-1.0, "Y", color="black", size=12.5)
#Axes_obj_08.text(1.5, 2.5, "Y=g(X)", color="black", size=12.5, ha='center', va='center')

########################################################################################################################
Axes_obj_09.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, length=0.0)
Axes_obj_09.set_title("Synthetic Curve (S + U)", fontweight="bold")
Axes_obj_09.set_xlabel('')
Axes_obj_09.set_ylabel('')
Axes_obj_09.set_xlim(-3.0, 3.0)
Axes_obj_09.set_ylim(-5.0, 5.0)

Axes_obj_09.plot(X, Y0, color='red', linewidth=0.5)
Axes_obj_09.plot(X, Y0_U, color='red', linewidth=0.5)
Axes_obj_09.plot(X, Y0_L, color='red', linewidth=0.5)
Axes_obj_09.plot(X, Y0_diff-5.0, color='red')
Axes_obj_09.fill_between(X, Y0_U, Y0_L, facecolor='red', alpha=0.05)
Axes_obj_09.fill_between(X, Y0_diff-5.0, Line_fill1, facecolor='red', alpha=0.2)

#Axes_obj_09.scatter([0, 0, 0], [-1, 0, 1], color='black', s=20)
Axes_obj_09.scatter([0], [0], color='black', s=20)

Axes_obj_09.scatter([-2, 0, 2],
                    [2.0*(1.0 / (1 + np.exp(-2.5 * (-2 - 0.0))) - 0.5),
                     2.0*(1.0 / (1 + np.exp(-2.5 * (0 - 0.0))) - 0.5),
                     2.0*(1.0 / (1 + np.exp(-2.5 * (2 - 0.0))) - 0.5)],
                    color='red', s=50, zorder=2)
Axes_obj_09.scatter([-2, 0, 2], Data_YU2, color="red", marker='_', s=50)
Axes_obj_09.scatter([-2, 0, 2], Data_YL2, color="red", marker='_', s=50)

Axes_obj_09.vlines(x=-2, ymin=Data_YL2[0], ymax=Data_YU2[0], color="red", linestyle='solid', linewidth=2.5)
Axes_obj_09.vlines(x=0, ymin=Data_YL2[1], ymax=Data_YU2[1], color="red", linestyle='solid', linewidth=2.5)
Axes_obj_09.vlines(x=2, ymin=Data_YL2[2], ymax=Data_YU2[2], color="red", linestyle='solid', linewidth=2.5)

Axes_obj_09.text(-2, Data_YU2[0]+1.0,
                 "C.I.\nwide", color="black", size=10, ha='center', va='center')
Axes_obj_09.text(0-0.5, Data_YU2[1]+1.0,
                 "C.I.\nnarrow", color="black", size=10, ha='center', va='center')
Axes_obj_09.text(2, Data_YU2[2]+1.0,
                 "C.I.\nwide", color="black", size=10, ha='center', va='center')

Axes_obj_09.text(-2, -4.5, "N=Small", color="black", size=10, ha='center', va='center')
Axes_obj_09.text(0, -3.35, "N=Large", color="black", size=10, ha='center', va='center')
Axes_obj_09.text(2, -4.5, "N=Small", color="black", size=10, ha='center', va='center')

Axes_obj_09.quiver(0.0, -3.0, 0.0, 8.0, color="black", angles='xy', scale_units='xy', scale=1,
                   headwidth=3*1.5, headlength=5*1.5, headaxislength=4.5*1.5)
Axes_obj_09.quiver(-3.0, 0.0, 6.0, 0.0, color="black", angles='xy', scale_units='xy', scale=1,
                   headwidth=3*1.5, headlength=5*1.5, headaxislength=4.5*1.5)
Axes_obj_09.text(0.125, -0.5-0.125, "O", color="black", size=12.5)
Axes_obj_09.text(3-0.35, 0+0.275, "X", color="black", size=12.5)
Axes_obj_09.text(0+0.15, 5-1.0, "Y", color="black", size=12.5)
#Axes_obj_09.text(-1.5, 2.75, "Y=f(X) + g(X)", color="black", size=12.5, ha='center', va='center')
#Axes_obj_09.text(1.5, -2.75, "Y=f(X) - g(X)", color="black", size=12.5, ha='center', va='center')

#Figure_object.tight_layout()
my_path = os.path.abspath("Figures")
plt.savefig(os.path.join(my_path, "01_Kimoto_et_al_(2023)_[Fig_S_02]"))

from PIL import Image # py -m pip install pillow
img = Image.open(os.path.join(my_path, "01_Kimoto_et_al_(2023)_[Fig_S_02].png"))
img_resize = img.resize(size=(2150, 1512))
my_path2 = os.path.abspath("Figures/Fixed_Version/B6")
img_resize.save(os.path.join(my_path2, "01_Kimoto_et_al_(2023)_[Fig_S_02]_B6.png"))

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