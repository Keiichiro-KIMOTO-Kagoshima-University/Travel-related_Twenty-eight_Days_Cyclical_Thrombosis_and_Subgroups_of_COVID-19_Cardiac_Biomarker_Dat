########################################################################################################################
# Fig. S7. Displaying case data as a bar chart “by 1-year” after accumulating from several studies (male)
# (01_Kimoto_et_al_(2023)_[Fig_S_07].py)
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

import sys
import pprint
pprint.pprint(sys.path)
sys.path.append("C:\\Users\\Keiichiro Kimoto\\Dropbox\\003_Program\\Anaconda\\Lib\\site-packages")
#sys.path.append('C:\Users\Keiichiro Kimoto\Dropbox\003_Program\Anaconda\pkgs')

pprint.pprint(sys.path)
#from scikit-learn import linear_model
from sklearn import linear_model
clf = linear_model.LinearRegression(fit_intercept=True, normalize=False, copy_X=False, n_jobs=1)

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

###############################################
# Data
###############################################
Data_Male_1977_Symington = [30, 42, 43, 65, 68]
Data_Male_1988_Cruickshank = [31, 48, 51, 60]
Data_Male_1999_Crerel = [26, 29, 44, 45, 47, 51, 57, 58, 66, 66, 66, 66, 68, 71, 77]
Data_Male_2006_Parkin = [32, 36, 39, 49]
Data_Male_Marge = [30, 42, 43, 65, 68, 31, 48, 51, 60, 26, 29, 44, 45, 47, 51, 57, 58, 66, 66, 66, 66, 68, 71, 77, 32, 36, 39, 49]

Data_Female_1977_Symington = [48, 60, 84]
Data_Female_1988_Cruickshank = [43]
Data_Female_1999_Crerel = []
Data_Female_2006_Parkin = [42, 47, 47, 51, 54, 57, 58]

Male_Age_25 = [25, 0, 0, 0, 0, 0]
Male_Age_26 = [26, 0, 0, 1, 0, 1]
Male_Age_27 = [27, 0, 0, 0, 0, 0]
Male_Age_28 = [28, 0, 0, 0, 0, 0]
Male_Age_29 = [29, 0, 0, 1, 0, 1]
Male_Age_30 = [30, 1, 0, 0, 0, 1]
Male_Age_31 = [31, 0, 1, 0, 0, 1]
Male_Age_32 = [32, 0, 0, 0, 1, 1]
Male_Age_33 = [33, 0, 0, 0, 0, 0]
Male_Age_34 = [34, 0, 0, 0, 0, 0]
Male_Age_35 = [35, 0, 0, 0, 0, 0]
Male_Age_36 = [36, 0, 0, 0, 1, 1]
Male_Age_37 = [37, 0, 0, 0, 0, 0]
Male_Age_38 = [38, 0, 0, 0, 0, 0]
Male_Age_39 = [39, 0, 0, 0, 1, 1]
Male_Age_40 = [40, 0, 0, 0, 0, 0]
Male_Age_41 = [41, 0, 0, 0, 0, 0]
Male_Age_42 = [42, 1, 0, 0, 0, 1]
Male_Age_43 = [43, 1, 0, 0, 0, 1]
Male_Age_44 = [44, 0, 0, 1, 0, 1]
Male_Age_45 = [45, 0, 0, 1, 0, 1]
Male_Age_46 = [46, 0, 0, 0, 0, 0]
Male_Age_47 = [47, 0, 0, 1, 0, 1]
Male_Age_48 = [48, 0, 1, 0, 0, 1]
Male_Age_49 = [49, 0, 0, 0, 1, 1]
Male_Age_50 = [50, 0, 0, 0, 0, 0]
Male_Age_51 = [51, 0, 1, 1, 0, 2]
Male_Age_52 = [52, 0, 0, 0, 0, 0]
Male_Age_53 = [53, 0, 0, 0, 0, 0]
Male_Age_54 = [54, 0, 0, 0, 0, 0]
Male_Age_55 = [55, 0, 0, 0, 0, 0]
Male_Age_56 = [56, 0, 0, 0, 0, 0]
Male_Age_57 = [57, 0, 0, 1, 0, 1]
Male_Age_58 = [58, 0, 0, 1, 0, 1]
Male_Age_59 = [59, 0, 0, 0, 0, 0]
Male_Age_60 = [60, 0, 1, 0, 0, 1]
Male_Age_61 = [61, 0, 0, 0, 0, 0]
Male_Age_62 = [62, 0, 0, 0, 0, 0]
Male_Age_63 = [63, 0, 0, 0, 0, 0]
Male_Age_64 = [64, 0, 0, 0, 0, 0]
Male_Age_65 = [65, 1, 0, 0, 0, 1]
Male_Age_66 = [66, 0, 0, 4, 0, 4]
Male_Age_67 = [67, 0, 0, 0, 0, 0]
Male_Age_68 = [68, 1, 0, 1, 0, 2]
Male_Age_69 = [69, 0, 0, 0, 0, 0]
Male_Age_70 = [70, 0, 0, 0, 0, 0]
Male_Age_71 = [71, 0, 0, 1, 0, 1]
Male_Age_72 = [72, 0, 0, 0, 0, 0]
Male_Age_73 = [73, 0, 0, 0, 0, 0]
Male_Age_74 = [74, 0, 0, 0, 0, 0]
Male_Age_75 = [75, 0, 0, 0, 0, 0]
Male_Age_76 = [76, 0, 0, 0, 0, 0]
Male_Age_77 = [77, 0, 0, 1, 0, 1]
Male_Age_78 = [78, 0, 0, 0, 0, 0]
Male_Age_79 = [79, 0, 1, 0, 0, 1]
Male_Age_80 = [80, 0, 0, 0, 0, 0]
Male_Age_81 = [81, 0, 0, 0, 0, 0]
Male_Age_82 = [82, 0, 0, 0, 0, 0]
Male_Age_83 = [83, 0, 0, 0, 0, 0]
Male_Age_84 = [84, 0, 0, 0, 0, 0]
Male_Age_85 = [85, 0, 0, 0, 0, 0]

DataSet_Male = pd.DataFrame([Male_Age_25, Male_Age_26, Male_Age_27, Male_Age_28, Male_Age_29, Male_Age_30, Male_Age_31,
                             Male_Age_32, Male_Age_33, Male_Age_34, Male_Age_35, Male_Age_36, Male_Age_37, Male_Age_38,
                             Male_Age_39, Male_Age_40, Male_Age_41, Male_Age_42, Male_Age_43, Male_Age_44, Male_Age_45,
                             Male_Age_46, Male_Age_47, Male_Age_48, Male_Age_49, Male_Age_50, Male_Age_51, Male_Age_52,
                             Male_Age_53, Male_Age_54, Male_Age_55, Male_Age_56, Male_Age_57, Male_Age_58, Male_Age_59,
                             Male_Age_60, Male_Age_61, Male_Age_62, Male_Age_63, Male_Age_64, Male_Age_65, Male_Age_66,
                             Male_Age_67, Male_Age_68, Male_Age_69, Male_Age_70, Male_Age_71, Male_Age_72, Male_Age_73,
                             Male_Age_74, Male_Age_75, Male_Age_76, Male_Age_77, Male_Age_78, Male_Age_79, Male_Age_80,
                             Male_Age_81, Male_Age_82, Male_Age_83, Male_Age_84, Male_Age_85],
                            index=["Male_Age_25", "Male_Age_26", "Male_Age_27", "Male_Age_28", "Male_Age_29",
                                   "Male_Age_30", "Male_Age_31", "Male_Age_32", "Male_Age_33", "Male_Age_34",
                                   "Male_Age_35", "Male_Age_36", "Male_Age_37", "Male_Age_38", "Male_Age_39",
                                   "Male_Age_40", "Male_Age_41", "Male_Age_42", "Male_Age_43", "Male_Age_44",
                                   "Male_Age_45", "Male_Age_46", "Male_Age_47", "Male_Age_48", "Male_Age_49",
                                   "Male_Age_50", "Male_Age_51", "Male_Age_52", "Male_Age_53", "Male_Age_54",
                                   "Male_Age_55", "Male_Age_56", "Male_Age_57", "Male_Age_58", "Male_Age_59",
                                   "Male_Age_60", "Male_Age_61", "Male_Age_62", "Male_Age_63", "Male_Age_64",
                                   "Male_Age_65", "Male_Age_66", "Male_Age_67", "Male_Age_68", "Male_Age_69",
                                   "Male_Age_70", "Male_Age_71", "Male_Age_72", "Male_Age_73", "Male_Age_74",
                                   "Male_Age_75", "Male_Age_76", "Male_Age_77", "Male_Age_78", "Male_Age_79",
                                   "Male_Age_80", "Male_Age_81", "Male_Age_82", "Male_Age_83", "Male_Age_84",
                                   "Male_Age_85"],
                                columns=["Year", "Symington", "Cruickshank", "Crerel", "Parkin", "Total"])

DataSet_Male_1977_Symington = DataSet_Male[DataSet_Male["Symington"] != 0]
DataSet_Male_1988_Cruickshank = DataSet_Male[DataSet_Male["Cruickshank"] != 0]
DataSet_Male_1999_Crerel = DataSet_Male[DataSet_Male["Crerel"] != 0]
DataSet_Male_2006_Parkin = DataSet_Male[DataSet_Male["Parkin"] != 0]
DataSet_Male_Total = DataSet_Male[DataSet_Male["Total"] != 0]

########################################################################################################################
# Figures
########################################################################################################################
Figure_object = plt.figure(1, figsize=(11.69, 8.27), dpi=400, edgecolor="black", linewidth=0.5)
#plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)
plt.subplots_adjust(left=0.125 , bottom=0.11 , right=0.9 , top=0.88 , wspace=0.2 , hspace=0.2)

plt.figtext(0.019, 0.832433+0.13, "a", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")
plt.figtext(0.019, 0.635879+0.13, "b", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")
plt.figtext(0.019, 0.439326+0.13, "c", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")
plt.figtext(0.019, 0.242772+0.13, "d", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")
plt.figtext(0.019, 0.0462179+0.13, "e", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")

#gs_master = matplotlib.gridspec.GridSpec(nrows=5, ncols=1, height_ratios=[1, 1, 1, 1, 1])
#gs_1 = matplotlib.gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[0])
#gs_2 = matplotlib.gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[1])
#gs_3 = matplotlib.gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[2])
#gs_4 = matplotlib.gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[3])
#gs_5 = matplotlib.gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[4])

#Axes_obj_01 = Figure_object.add_subplot(gs_1[0])
#Axes_obj_02 = Figure_object.add_subplot(gs_2[0])
#Axes_obj_03 = Figure_object.add_subplot(gs_3[0])
#Axes_obj_04 = Figure_object.add_subplot(gs_4[0])
#Axes_obj_05 = Figure_object.add_subplot(gs_5[0])

Axes_obj_01 = plt.axes([0.0450141, 0.832433, 0.942154, 0.125144])
Axes_obj_02 = plt.axes([0.0450141, 0.635879, 0.942154, 0.125144])
Axes_obj_03 = plt.axes([0.0450141, 0.439326, 0.942154, 0.125144])
Axes_obj_04 = plt.axes([0.0450141, 0.242772, 0.942154, 0.125144])
Axes_obj_05 = plt.axes([0.0450141, 0.0462179, 0.942154, 0.125144])

############################################################################################################
# "(a)
############################################################################################################
Axes_obj_01.set_title("Pulmonary embolism after travel, Symington & Stack, 1977 (male data, n = 5)",
                      size=11.4230, fontweight="normal")
#Axes_obj_01.set_xlabel('Year')
Axes_obj_01.set_ylabel('Count')
Axes_obj_01.set_xticks([25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85])
Axes_obj_01.xaxis.set_minor_locator(ticker.MultipleLocator(1))
Axes_obj_01.set_xlim(24, 86)
Axes_obj_01.set_ylim(0.0, 8.5)
Axes_obj_01.set_yticks([1, 2, 3, 4, 5, 6, 7, 8])
Axes_obj_01.bar(DataSet_Male["Year"], DataSet_Male["Symington"], color='blue', width=0.7, alpha=0.3)

for i in range(0, len(DataSet_Male_1977_Symington["Symington"])):
    Axes_obj_01.text(DataSet_Male_1977_Symington["Year"][i], DataSet_Male_1977_Symington["Symington"][i]+0.3,
                     DataSet_Male_1977_Symington["Symington"][i], ha='center', va='center', size=8.0, color="gray")

#from scipy.stats import gaussian_kde
#kde_model = gaussian_kde(Data_Male_1977_Symington)
#x = np.linspace(24, 86, num=100)
#y = kde_model(x)
#Axes_obj_01.plot(x, y*(1/np.max(y)), color="blue", linestyle="solid", linewidth=0.7)

Axes_obj_01.fill_between(np.linspace(52.0-0.5, 56+0.5, 1000), (8.5-0.0)*np.ones(1000), (0.0-0.0)*np.ones(1000),
                         facecolor='yellow', alpha=0.3)

Axes_obj_01.fill_between(np.linspace(61.0-0.5, 64+0.5, 1000), (8.5-0.0)*np.ones(1000), (0.0-0.0)*np.ones(1000),
                         facecolor='yellow', alpha=0.3)

Axes_obj_01.fill_between(np.linspace(72.0-0.5, 76+0.5, 1000), (8.5-0.0)*np.ones(1000), (0.0-0.0)*np.ones(1000),
                         facecolor='yellow', alpha=0.3)

############################################################################################################
# "(b)
############################################################################################################
Axes_obj_02.set_title("Pulmonary embolism after travel, Cruickshank, Gorlin, & Jennett, 1988 (male data, n = 5)",
                      size=11.4230, fontweight="normal")
#Axes_obj_02.set_xlabel('Year')
Axes_obj_02.set_xticks([25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85])
Axes_obj_02.xaxis.set_minor_locator(ticker.MultipleLocator(1))
Axes_obj_02.set_ylabel('Count')
Axes_obj_02.set_xlim(24, 86)
Axes_obj_02.set_ylim(0.0, 8.5)
Axes_obj_02.set_yticks([1, 2, 3, 4, 5, 6, 7, 8])
Axes_obj_02.bar(DataSet_Male["Year"], DataSet_Male["Cruickshank"], color='blue', width=0.7, alpha=0.3)

for i in range(0, len(DataSet_Male_1988_Cruickshank["Cruickshank"])):
    Axes_obj_02.text(DataSet_Male_1988_Cruickshank["Year"][i], DataSet_Male_1988_Cruickshank["Cruickshank"][i]+0.3,
                     DataSet_Male_1988_Cruickshank["Cruickshank"][i], ha='center', va='center', size=8.0, color="gray")

#from scipy.stats import gaussian_kde
#kde_model = gaussian_kde(Data_Male_1988_Cruickshank)
#x = np.linspace(24, 86, num=100)
#y = kde_model(x)
#Axes_obj_02.plot(x, y*(1/np.max(y)), color="blue", linestyle="solid", linewidth=0.7)

Axes_obj_02.fill_between(np.linspace(52.0-0.5, 56+0.5, 1000), (8.5-0.0)*np.ones(1000), (0.0-0.0)*np.ones(1000),
                         facecolor='yellow', alpha=0.3)

Axes_obj_02.fill_between(np.linspace(61.0-0.5, 64+0.5, 1000), (8.5-0.0)*np.ones(1000), (0.0-0.0)*np.ones(1000),
                         facecolor='yellow', alpha=0.3)

Axes_obj_02.fill_between(np.linspace(72.0-0.5, 76+0.5, 1000), (8.5-0.0)*np.ones(1000), (0.0-0.0)*np.ones(1000),
                         facecolor='yellow', alpha=0.3)

Axes_obj_02.annotate(
 text='\n'.join(["Dr. Cruickshank, age 48", "(personal paper in Lancet)"]),
    xy=(48, 2.0), xytext=(48, 6.0), ha='center', va='center', size=10.0,
    bbox=dict(boxstyle='round', edgecolor='blue', fc='white'),
    arrowprops=dict(
        facecolor='black', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0, edgecolor='black', shrink=0.1)
)

Axes_obj_02.annotate(
 text='\n'.join(["Dr. Jennett, age 60", "(personal paper in Lancet)"]),
    xy=(60, 2.0), xytext=(60, 6.0), ha='center', va='center', size=10.0,
    bbox=dict(boxstyle='round', edgecolor='blue', fc='white'),
    arrowprops=dict(
        facecolor='black', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0, edgecolor='black', shrink=0.1)
)

############################################################################################################
# "(c)
############################################################################################################
Axes_obj_03.set_title("Pulmonary embolism after travel, Clérel & Caillard, 1999 (male data, n = 15)",
                      size=11.4230, fontweight="normal")
#Axes_obj_03.set_xlabel('Year')
Axes_obj_03.set_xticks([25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85])
Axes_obj_03.xaxis.set_minor_locator(ticker.MultipleLocator(1))
Axes_obj_03.set_ylabel('Count')
Axes_obj_03.set_xlim(24, 86)
Axes_obj_03.set_ylim(0.0, 8.5)
Axes_obj_03.set_yticks([1, 2, 3, 4, 5, 6, 7, 8])
Axes_obj_03.bar(DataSet_Male["Year"], DataSet_Male["Crerel"], color='blue', width=0.7, alpha=0.3)

for i in range(0, len(DataSet_Male_1999_Crerel["Crerel"])):
    Axes_obj_03.text(DataSet_Male_1999_Crerel["Year"][i], DataSet_Male_1999_Crerel["Crerel"][i]+0.3,
                     DataSet_Male_1999_Crerel["Crerel"][i], ha='center', va='center', size=8.0, color="gray")

#from scipy.stats import gaussian_kde
#kde_model = gaussian_kde(Data_Male_1999_Crerel)
#x = np.linspace(24, 86, num=100)
#y = kde_model(x)
#Axes_obj_03.plot(x, y*(4/np.max(y)), color="blue", linestyle="solid", linewidth=0.7)

Axes_obj_03.fill_between(np.linspace(52.0-0.5, 56+0.5, 1000), (8.5-0.0)*np.ones(1000), (0.0-0.0)*np.ones(1000),
                         facecolor='yellow', alpha=0.3)

Axes_obj_03.fill_between(np.linspace(61.0-0.5, 64+0.5, 1000), (8.5-0.0)*np.ones(1000), (0.0-0.0)*np.ones(1000),
                         facecolor='yellow', alpha=0.3)

Axes_obj_03.fill_between(np.linspace(72.0-0.5, 76+0.5, 1000), (8.5-0.0)*np.ones(1000), (0.0-0.0)*np.ones(1000),
                         facecolor='yellow', alpha=0.3)

############################################################################################################
# "(d)
############################################################################################################
Axes_obj_04.set_title("Pulmonary embolism after travel, Parkin et al., 2006 (male data, n = 4)", size=11.4230, fontweight="normal")
#Axes_obj_04.set_xlabel('Year')
Axes_obj_04.set_xticks([25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85])
Axes_obj_04.xaxis.set_minor_locator(ticker.MultipleLocator(1))
Axes_obj_04.set_ylabel('Count')
Axes_obj_04.set_xlim(24, 86)
Axes_obj_04.set_ylim(0.0, 8.5)
Axes_obj_04.set_yticks([1, 2, 3, 4, 5, 6, 7, 8])
Axes_obj_04.bar(DataSet_Male["Year"], DataSet_Male["Parkin"], color='blue', width=0.7, alpha=0.3)

for i in range(0, len(DataSet_Male_2006_Parkin["Parkin"])):
    Axes_obj_04.text(DataSet_Male_2006_Parkin["Year"][i], DataSet_Male_2006_Parkin["Parkin"][i]+0.3,
                     DataSet_Male_2006_Parkin["Parkin"][i], ha='center', va='center', size=8.0, color="gray")

#from scipy.stats import gaussian_kde
#kde_model = gaussian_kde(Data_Male_2006_Parkin)
#x = np.linspace(24, 86, num=100)
#y = kde_model(x)
#Axes_obj_04.plot(x, y*(1/np.max(y)), color="blue", linestyle="solid", linewidth=0.7)

Axes_obj_04.fill_between(np.linspace(52.0-0.5, 56+0.5, 1000), (8.5-0.0)*np.ones(1000), (0.0-0.0)*np.ones(1000),
                         facecolor='yellow', alpha=0.3)

Axes_obj_04.fill_between(np.linspace(61.0-0.5, 64+0.5, 1000), (8.5-0.0)*np.ones(1000), (0.0-0.0)*np.ones(1000),
                         facecolor='yellow', alpha=0.3)

Axes_obj_04.fill_between(np.linspace(72.0-0.5, 76+0.5, 1000), (8.5-0.0)*np.ones(1000), (0.0-0.0)*np.ones(1000),
                         facecolor='yellow', alpha=0.3)

############################################################################################################

############################################################################################################
# "(e)
############################################################################################################
Axes_obj_05.set_title("Pulmonary embolism after travel, four studies, 1977 - 2006 (marged male data, n = 29)", size=11.4230, fontweight="normal")
#Axes_obj_05.set_xlabel('Year')
Axes_obj_05.set_xticks([25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85])
Axes_obj_05.xaxis.set_minor_locator(ticker.MultipleLocator(1))
Axes_obj_05.set_ylabel('Count')
Axes_obj_05.set_xlim(24, 86)
Axes_obj_05.set_ylim(0.0, 8.5)
Axes_obj_05.set_yticks([1, 2, 3, 4, 5, 6, 7, 8])
Axes_obj_05.bar(DataSet_Male["Year"], DataSet_Male["Total"], color='blue', width=0.7, alpha=0.3)

for i in range(0, len(DataSet_Male_Total["Total"])):
    Axes_obj_05.text(DataSet_Male_Total["Year"][i], DataSet_Male_Total["Total"][i]+0.3,
                     DataSet_Male_Total["Total"][i], ha='center', va='center', size=8.0, color="gray")

#from scipy.stats import gaussian_kde
#kde_model = gaussian_kde(Data_Male_Marge)
#x = np.linspace(24, 86, num=100)
#y = kde_model(x)
#Axes_obj_05.plot(x, y*(4/np.max(y)), color="blue", linestyle="solid", linewidth=0.7)

Axes_obj_05.fill_between(np.linspace(29.0-0.5, 32+0.5, 1000), (8.5-0.0)*np.ones(1000), (0.0-0.0)*np.ones(1000),
                         facecolor='blue', alpha=0.1)

Axes_obj_05.fill_between(np.linspace(42.0-0.5, 51+0.5, 1000), (8.5-0.0)*np.ones(1000), (0.0-0.0)*np.ones(1000),
                         facecolor='blue', alpha=0.1)


Axes_obj_05.fill_between(np.linspace(52.0-0.5, 56+0.5, 1000), (8.5-0.0)*np.ones(1000), (0.0-0.0)*np.ones(1000),
                         facecolor='yellow', alpha=0.3)

Axes_obj_05.fill_between(np.linspace(61.0-0.5, 64+0.5, 1000), (8.5-0.0)*np.ones(1000), (0.0-0.0)*np.ones(1000),
                         facecolor='yellow', alpha=0.3)

Axes_obj_05.fill_between(np.linspace(72.0-0.5, 76+0.5, 1000), (8.5-0.0)*np.ones(1000), (0.0-0.0)*np.ones(1000),
                         facecolor='yellow', alpha=0.3)

Axes_obj_05.annotate(
 text='\n'.join(["Around 30", "Transition", "phase?"]),
    xy=(30.5, 8.5/2+1.75), xytext=(30.5, 8.5/2+1.75), ha='center', va='center', size=10.0,
    bbox=dict(boxstyle='round', edgecolor='red', fc='white'),
    arrowprops=dict(
        facecolor='black', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0, edgecolor='black', shrink=0.1)
)

Axes_obj_05.annotate(
 text='\n'.join(["Age 42 - 51", "Middle crisis or", "Karoshi?"]),
    xy=(46.5, 8.5/2+1.75), xytext=(46.5, 8.5/2+1.75), ha='center', va='center', size=10.0,
    bbox=dict(boxstyle='round', edgecolor='red', fc='white'),
    arrowprops=dict(
        facecolor='black', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0, edgecolor='black', shrink=0.1)
)

Axes_obj_05.annotate(
 text='\n'.join(["Age 57 - 60", "Pre- retirement", "Transition?"]),
    xy=(58.5, 8.5/2+1.75), xytext=(58.5, 8.5/2+1.75), ha='center', va='center', size=10.0,
    bbox=dict(boxstyle='round', edgecolor='red', fc='white'),
    arrowprops=dict(
        facecolor='black', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0, edgecolor='black', shrink=0.1)
)

Axes_obj_05.annotate(
 text='\n'.join(["Age 66, Retirement age?"]),
    xy=(66, 4.75), xytext=(66, 7.25), ha='left', va='center', size=10.0,
    bbox=dict(boxstyle='round', edgecolor='red', fc='white'),
    arrowprops=dict(
        facecolor='black', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0, edgecolor='black', shrink=0.1)
)

Axes_obj_05.annotate(
 text='\n'.join(["Age 65 - 71", "Retirement and", "enjoying travel?"]),
    xy=(68, 8.5/2+0.5), xytext=(81.5, 8.5/2+0.5), ha='center', va='center', size=10.0,
    bbox=dict(boxstyle='round', edgecolor='red', fc='white'),
    arrowprops=dict(
        facecolor='black', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0, edgecolor='black', shrink=0.1)
)

text='\n'.join(["Age 33 - 41", "Relatively few onset"])
Axes_obj_05.text((33+41)/2, 8.5/2-1.5, text, horizontalalignment='center', fontsize=10.0,  fontweight="normal")

text='\n'.join(["Age 52 - 56", "No onset"])
Axes_obj_05.text((52+56)/2, 8.5/2-3.5, text, horizontalalignment='center', fontsize=10.0,  fontweight="normal")

text='\n'.join(["Age 61 - 64", "No onset"])
Axes_obj_05.text((61+64)/2, 8.5/2-3.5, text, horizontalalignment='center', fontsize=10.0,  fontweight="normal")

text='\n'.join(["Age 72 - 76", "No onset"])
Axes_obj_05.text((72+76)/2, 8.5/2-3.5, text, horizontalalignment='center', fontsize=10.0,  fontweight="normal")


############################################################################################################


#Figure_object.tight_layout()
my_path = os.path.abspath("Figures")
plt.savefig(os.path.join(my_path, "01_Kimoto_et_al_(2023)_[Fig_S_07].png"))

from PIL import Image # py -m pip install pillow
img = Image.open(os.path.join(my_path, "01_Kimoto_et_al_(2023)_[Fig_S_07].png"))
img_resize = img.resize(size=(2866, 2016))
my_path2 = os.path.abspath("Figures/Fixed_Version/B6")
img_resize.save(os.path.join(my_path2, "01_Kimoto_et_al_(2023)_[Fig_S_07]_B6.png"))

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



