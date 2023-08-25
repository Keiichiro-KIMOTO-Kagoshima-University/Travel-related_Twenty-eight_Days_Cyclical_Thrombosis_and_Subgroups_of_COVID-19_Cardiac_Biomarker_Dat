########################################################################################################################
# Fig. S6. Visualized the change (trend) in the number of papers (trend analysis).(01_Kimoto_et_al_(2023)_[Fig_S_06].py)
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
Year_Count_2022 = [2022, 1, 0, 0, 1]
Year_Count_2021 = [2021, 1, 0, 0, 1]
Year_Count_2020 = [2020, 0, 0, 0, 0]
Year_Count_2019 = [2019, 0, 0, 1, 1]
Year_Count_2018 = [2018, 0, 0, 2, 2]
Year_Count_2017 = [2017, 2, 0, 0, 2]
Year_Count_2016 = [2016, 1, 1, 1, 3]
Year_Count_2015 = [2015, 0, 0, 0, 0]
Year_Count_2014 = [2014, 2, 0, 0, 2]
Year_Count_2013 = [2013, 1, 0, 1, 2]
Year_Count_2012 = [2012, 0, 0, 2, 2]
Year_Count_2011 = [2011, 2, 0, 0, 2]
Year_Count_2010 = [2010, 2, 1, 1, 4]
Year_Count_2009 = [2009, 1, 0, 2, 3]
Year_Count_2008 = [2008, 1, 1, 1, 3]
Year_Count_2007 = [2007, 1, 0, 2, 3]
Year_Count_2006 = [2006, 7, 0, 4, 11]
Year_Count_2005 = [2005, 1, 0, 1, 2]
Year_Count_2004 = [2004, 2, 1, 2, 5]
Year_Count_2003 = [2003, 7, 1, 2, 10]
Year_Count_2002 = [2002, 6, 1, 1, 8]
Year_Count_2001 = [2001, 9, 2, 4, 15]
Year_Count_2000 = [2000, 1, 1, 0, 2]
Year_Count_1999 = [1999, 3, 0, 0, 3]
Year_Count_1998 = [1998, 1, 0, 0, 1]
Year_Count_1997 = [1997, 1, 0, 0, 1]
Year_Count_1996 = [1996, 0, 0, 0, 0]
Year_Count_1995 = [1995, 1, 0, 0, 1]
Year_Count_1994 = [1994, 2, 0, 0, 2]
Year_Count_1993 = [1993, 0, 0, 0, 0]
Year_Count_1992 = [1992, 2, 0, 0, 2]
Year_Count_1991 = [1991, 0, 0, 0, 0]
Year_Count_1990 = [1990, 0, 0, 0, 0]
Year_Count_1989 = [1989, 1, 0, 0, 1]
Year_Count_1988 = [1988, 2, 0, 0, 2]
Year_Count_1987 = [1987, 0, 0, 0, 0]
Year_Count_1986 = [1986, 0, 0, 0, 0]
Year_Count_1985 = [1985, 1, 0, 0, 1]
Year_Count_1984 = [1984, 0, 0, 0, 0]
Year_Count_1983 = [1983, 0, 0, 0, 0]
Year_Count_1982 = [1982, 0, 0, 0, 0]
Year_Count_1981 = [1981, 0, 0, 0, 0]
Year_Count_1980 = [1980, 0, 0, 0, 0]
Year_Count_1979 = [1979, 0, 0, 0, 0]
Year_Count_1978 = [1978, 0, 1, 0, 0]
Year_Count_1977 = [1977, 0, 0, 0, 0]
Year_Count_1976 = [1976, 0, 0, 0, 0]
Year_Count_1975 = [1975, 0, 0, 0, 0]

DataSet_Year_Count = pd.DataFrame([Year_Count_2022, Year_Count_2021, Year_Count_2020, Year_Count_2019, Year_Count_2018,
                                   Year_Count_2017, Year_Count_2016, Year_Count_2015, Year_Count_2014, Year_Count_2013,
                                   Year_Count_2012, Year_Count_2011, Year_Count_2010, Year_Count_2009, Year_Count_2008,
                                   Year_Count_2007, Year_Count_2006, Year_Count_2005, Year_Count_2004, Year_Count_2003,
                                   Year_Count_2002, Year_Count_2001, Year_Count_2000, Year_Count_1999, Year_Count_1998,
                                   Year_Count_1997, Year_Count_1996, Year_Count_1995, Year_Count_1994, Year_Count_1993,
                                   Year_Count_1992, Year_Count_1991, Year_Count_1990, Year_Count_1989, Year_Count_1988,
                                   Year_Count_1987, Year_Count_1986, Year_Count_1985, Year_Count_1984, Year_Count_1983,
                                   Year_Count_1982, Year_Count_1981, Year_Count_1980, Year_Count_1979, Year_Count_1978,
                                   Year_Count_1977, Year_Count_1976, Year_Count_1975],
                                index=["Year_Count_2022", "Year_Count_2021", "Year_Count_2020", "Year_Count_2019",
                                       "Year_Count_2018", "Year_Count_2017", "Year_Count_2016", "Year_Count_2015",
                                       "Year_Count_2014", "Year_Count_2013", "Year_Count_2012", "Year_Count_2011",
                                       "Year_Count_2010", "Year_Count_2009", "Year_Count_2008", "Year_Count_2007",
                                       "Year_Count_2006", "Year_Count_2005", "Year_Count_2004", "Year_Count_2003",
                                       "Year_Count_2002", "Year_Count_2001", "Year_Count_2000", "Year_Count_1999",
                                       "Year_Count_1998", "Year_Count_1997", "Year_Count_1996", "Year_Count_1995",
                                       "Year_Count_1994", "Year_Count_1993", "Year_Count_1992", "Year_Count_1991",
                                       "Year_Count_1990", "Year_Count_1989", "Year_Count_1988", "Year_Count_1987",
                                       "Year_Count_1986", "Year_Count_1985", "Year_Count_1984", "Year_Count_1983",
                                       "Year_Count_1982", "Year_Count_1981", "Year_Count_1980", "Year_Count_1979",
                                       "Year_Count_1978", "Year_Count_1977", "Year_Count_1976", "Year_Count_1975"],
                                columns=["Year", "Economy Class", "Travellers", "Travel-related", "Total"])

###############################################
# Data (Except Review and Meta-analysis)
###############################################
Year_Count_2022 = [2022, 1, 0, 0, 1]
Year_Count_2021 = [2021, 0, 0, 0, 0]
Year_Count_2020 = [2020, 0, 0, 0, 0]
Year_Count_2019 = [2019, 0, 0, 1, 1]
Year_Count_2018 = [2018, 0, 0, 1, 1]
Year_Count_2017 = [2017, 2, 0, 0, 2]
Year_Count_2016 = [2016, 1, 1, 1, 3]
Year_Count_2015 = [2015, 0, 0, 0, 0]
Year_Count_2014 = [2014, 2, 0, 0, 2]
Year_Count_2013 = [2013, 1, 0, 0, 1]
Year_Count_2012 = [2012, 0, 0, 1, 1]
Year_Count_2011 = [2011, 1, 0, 0, 1]
Year_Count_2010 = [2010, 1, 1, 0, 2]
Year_Count_2009 = [2009, 1, 0, 0, 1]
Year_Count_2008 = [2008, 1, 1, 0, 2]
Year_Count_2007 = [2007, 1, 0, 1, 2]
Year_Count_2006 = [2006, 6, 0, 3, 9]
Year_Count_2005 = [2005, 1, 0, 0, 1]
Year_Count_2004 = [2004, 1, 0, 2, 3]
Year_Count_2003 = [2003, 6, 1, 2, 9]
Year_Count_2002 = [2002, 6, 1, 1, 8]
Year_Count_2001 = [2001, 8, 1, 4, 13]
Year_Count_2000 = [2000, 1, 0, 0, 1]
Year_Count_1999 = [1999, 3, 0, 0, 3]
Year_Count_1998 = [1998, 1, 0, 0, 1]
Year_Count_1997 = [1997, 0, 0, 0, 0]
Year_Count_1996 = [1996, 0, 0, 0, 0]
Year_Count_1995 = [1995, 1, 0, 0, 1]
Year_Count_1994 = [1994, 2, 0, 0, 2]
Year_Count_1993 = [1993, 0, 0, 0, 0]
Year_Count_1992 = [1992, 2, 0, 0, 2]
Year_Count_1991 = [1991, 0, 0, 0, 0]
Year_Count_1990 = [1990, 0, 0, 0, 0]
Year_Count_1989 = [1989, 1, 0, 0, 1]
Year_Count_1988 = [1988, 2, 0, 0, 2]
Year_Count_1987 = [1987, 0, 0, 0, 0]
Year_Count_1986 = [1986, 0, 0, 0, 0]
Year_Count_1985 = [1985, 1, 0, 0, 1]
Year_Count_1984 = [1984, 0, 0, 0, 0]
Year_Count_1983 = [1983, 0, 0, 0, 0]
Year_Count_1982 = [1982, 0, 0, 0, 0]
Year_Count_1981 = [1981, 0, 0, 0, 0]
Year_Count_1980 = [1980, 0, 0, 0, 0]
Year_Count_1979 = [1979, 0, 0, 0, 0]
Year_Count_1978 = [1978, 0, 1, 0, 0]
Year_Count_1977 = [1977, 0, 0, 0, 0]
Year_Count_1976 = [1976, 0, 0, 0, 0]
Year_Count_1975 = [1975, 0, 0, 0, 0]

DataSet_Year_Count_N0_RV = pd.DataFrame([Year_Count_2022, Year_Count_2021, Year_Count_2020, Year_Count_2019, Year_Count_2018,
                                   Year_Count_2017, Year_Count_2016, Year_Count_2015, Year_Count_2014, Year_Count_2013,
                                   Year_Count_2012, Year_Count_2011, Year_Count_2010, Year_Count_2009, Year_Count_2008,
                                   Year_Count_2007, Year_Count_2006, Year_Count_2005, Year_Count_2004, Year_Count_2003,
                                   Year_Count_2002, Year_Count_2001, Year_Count_2000, Year_Count_1999, Year_Count_1998,
                                   Year_Count_1997, Year_Count_1996, Year_Count_1995, Year_Count_1994, Year_Count_1993,
                                   Year_Count_1992, Year_Count_1991, Year_Count_1990, Year_Count_1989, Year_Count_1988,
                                   Year_Count_1987, Year_Count_1986, Year_Count_1985, Year_Count_1984, Year_Count_1983,
                                   Year_Count_1982, Year_Count_1981, Year_Count_1980, Year_Count_1979, Year_Count_1978,
                                   Year_Count_1977, Year_Count_1976, Year_Count_1975],
                                index=["Year_Count_2022", "Year_Count_2021", "Year_Count_2020", "Year_Count_2019",
                                       "Year_Count_2018", "Year_Count_2017", "Year_Count_2016", "Year_Count_2015",
                                       "Year_Count_2014", "Year_Count_2013", "Year_Count_2012", "Year_Count_2011",
                                       "Year_Count_2010", "Year_Count_2009", "Year_Count_2008", "Year_Count_2007",
                                       "Year_Count_2006", "Year_Count_2005", "Year_Count_2004", "Year_Count_2003",
                                       "Year_Count_2002", "Year_Count_2001", "Year_Count_2000", "Year_Count_1999",
                                       "Year_Count_1998", "Year_Count_1997", "Year_Count_1996", "Year_Count_1995",
                                       "Year_Count_1994", "Year_Count_1993", "Year_Count_1992", "Year_Count_1991",
                                       "Year_Count_1990", "Year_Count_1989", "Year_Count_1988", "Year_Count_1987",
                                       "Year_Count_1986", "Year_Count_1985", "Year_Count_1984", "Year_Count_1983",
                                       "Year_Count_1982", "Year_Count_1981", "Year_Count_1980", "Year_Count_1979",
                                       "Year_Count_1978", "Year_Count_1977", "Year_Count_1976", "Year_Count_1975"],
                                columns=["Year", "Economy Class", "Travellers", "Travel-related", "Total"])

########################################################################################################################
print("")
print("###############################################################################################################")
print("########## Economy Class Syndrome, Travellers Thrombosis, and Travel-related thromboembolism ##########")
print("---------------------------------------------------------------------------------------------------------------")

print(DataSet_Year_Count)

DataSet_Year_Count_Economy = DataSet_Year_Count[DataSet_Year_Count["Economy Class"] != 0]
DataSet_Year_Count_Travellers = DataSet_Year_Count[DataSet_Year_Count["Travellers"] != 0]
DataSet_Year_Count_Travel_related = DataSet_Year_Count[DataSet_Year_Count["Travel-related"] != 0]

#print("")
#print(DataSet_Year_Count_Travel_related)

print("---------------------------------------------------------------------------------------------------------------")
print("Economy Class", "Line = ", len(DataSet_Year_Count_Economy["Economy Class"]))
print("Travellers", "Line = ", len(DataSet_Year_Count_Travellers["Travellers"]))
print("Travel-related", "Line = ", len(DataSet_Year_Count_Travel_related["Travel-related"]))
print("---------------------------------------------------------------------------------------------------------------")
print("Economy Class", "Count = ", sum(DataSet_Year_Count_Economy["Economy Class"]))
print("Travellers", "Count = ", sum(DataSet_Year_Count_Travellers["Travellers"]))
print("Travel-related", "Count = ", sum(DataSet_Year_Count_Travel_related["Travel-related"]))
print("---------------------------------------------------------------------------------------------------------------")

DataSet_Year_Count_Economy_N0_RV = DataSet_Year_Count_N0_RV[DataSet_Year_Count_N0_RV["Economy Class"] != 0]
DataSet_Year_Count_Travellers_N0_RV = DataSet_Year_Count_N0_RV[DataSet_Year_Count_N0_RV["Travellers"] != 0]
DataSet_Year_Count_Travel_related_N0_RV = DataSet_Year_Count_N0_RV[DataSet_Year_Count_N0_RV["Travel-related"] != 0]

########################################################################################################################
# Figures
########################################################################################################################
Figure_object = plt.figure(1, figsize=(11.69, 8.27), dpi=400, edgecolor="black", linewidth=0.5)
#plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)
plt.subplots_adjust(left=0.125, bottom=0.11, right=0.9, top=0.88, wspace=0.2, hspace=0.2)

plt.figtext(0.019, 0.788375+0.18, "a", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")
plt.figtext(0.019, 0.548670+0.18, "b", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")
plt.figtext(0.019, 0.308965+0.18, "c", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")
plt.figtext(0.019, 0.0692597+0.18, "d", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")

#gs_master = matplotlib.gridspec.GridSpec(nrows=4, ncols=1, height_ratios=[1, 1, 1, 1])
#gs_1 = matplotlib.gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[0])
#gs_2 = matplotlib.gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[1])
#gs_3 = matplotlib.gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[2])
#gs_4 = matplotlib.gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[3])

#Axes_obj_01 = Figure_object.add_subplot(gs_1[0])
#Axes_obj_02 = Figure_object.add_subplot(gs_2[0])
#Axes_obj_03 = Figure_object.add_subplot(gs_3[0])
#Axes_obj_04 = Figure_object.add_subplot(gs_4[0])

Axes_obj_01 = plt.axes([0.0638871, 0.788375, 0.913858, 0.169203])
Axes_obj_02 = plt.axes([0.0638871, 0.54867, 0.913858, 0.169203])
Axes_obj_03 = plt.axes([0.0638871, 0.308965, 0.913858, 0.169203])
Axes_obj_04 = plt.axes([0.0638871, 0.0692597, 0.913858, 0.169203])


############################################################################################################
# "(a)
############################################################################################################
Axes_obj_01.set_title("Search keyword in titles of scientific papers: \"Economy Class Syndrome\"", size=11.4230, fontweight="normal")
#Axes_obj_01.set_xlabel('Year')
Axes_obj_01.set_ylabel('Count')
Axes_obj_01.set_xlim(1970, 2025)
Axes_obj_01.set_ylim(0.0, 10)
Axes_obj_01.set_xticks([1970, 1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025])
Axes_obj_01.set_xticklabels(["1970", "1975", "1980", "1985", "1990", "1995", "2000", "2005", "2010", "2015", "2020", "2025"])
Axes_obj_01.xaxis.set_minor_locator(ticker.MultipleLocator(1))
Axes_obj_01.yaxis.set_minor_locator(ticker.MultipleLocator(1))

Axes_obj_01.fill_between(np.linspace(1993, 1998, 1000), (10-0.0)*np.ones(1000), (0.0-0.0)*np.ones(1000),
                         facecolor='orange', alpha=0.2, zorder=7)

Axes_obj_01.bar(DataSet_Year_Count_N0_RV["Year"], DataSet_Year_Count_N0_RV["Economy Class"], color='white', width=0.7, alpha=1.0, zorder=8)
Axes_obj_01.bar(DataSet_Year_Count["Year"], DataSet_Year_Count["Economy Class"], color='red', width=0.7, alpha=0.3, zorder=9)
Axes_obj_01.bar(DataSet_Year_Count_N0_RV["Year"], DataSet_Year_Count_N0_RV["Economy Class"], color='red', width=0.7, alpha=0.3, zorder=10)

for i in range(0, len(DataSet_Year_Count_Economy["Economy Class"])):
    Axes_obj_01.text(DataSet_Year_Count_Economy["Year"][i], DataSet_Year_Count_Economy["Economy Class"][i]+0.3,
                     DataSet_Year_Count_Economy["Economy Class"][i], ha='center', va='center', size=8.0, color="gray")

Axes_obj_01.annotate(
 text='\n'.join(["Year", "1988"]),
    xy=(1988, 2.5), xytext=(1988, 6.5), ha='center', va='center', size=10.0,
    bbox=dict(boxstyle='round', edgecolor='red', fc='white'),
    arrowprops=dict(
        facecolor='black', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0, edgecolor='black', shrink=0.1)
)

Axes_obj_01.annotate(
 text='\n'.join(["Year", "1985"]),
    xy=(1985, 2.5), xytext=(1985, 6.5), ha='center', va='center', size=10.0,
    bbox=dict(boxstyle='round', edgecolor='red', fc='white', linestyle="dashed"),
    arrowprops=dict(
        facecolor='black', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0, edgecolor='black', shrink=0.1)
)

Axes_obj_01.annotate(
 text='\n'.join(["Year 1977, The word", "\"Economy Class Syndrome\"", "first described by", "Symington & Stack"]),
    xy=(1977, 0.25), xytext=(1977, 6.0), ha='center', va='center', size=10.0,
    bbox=dict(boxstyle='round', edgecolor='red', fc='white'),
    arrowprops=dict(
        facecolor='black', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0, edgecolor='black', shrink=0.1)
)


Axes_obj_01.annotate(
 text='\n'.join(["American College of Chest Physicians (2012)",
                 "No evidence to support \'economy class syndrome\'"]),
    xy=(2016, 8.5), xytext=(2016, 8.5), ha='center', va='center', size=9.0,
    bbox=dict(boxstyle='round', edgecolor='orange', fc='lemonchiffon'),
    arrowprops=dict(
        facecolor='black', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0, edgecolor='black', shrink=0.1)
)

Axes_obj_01.annotate(
 text='\n'.join(["    Clark et al. (2018) title: Long-haul travel and     ",
                 " venous thrombosis: What is the evidence? "]),
    xy=(2016, 5.75), xytext=(2016, 5.75), ha='center', va='center', size=9.0,
    bbox=dict(boxstyle='round', edgecolor='orange', fc='lemonchiffon'),
    arrowprops=dict(
        facecolor='black', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0, edgecolor='black', shrink=0.1)
)

text='\n'.join(["1993-1998", "WHI trial for", "HRT", "recruiting", "period"])
Axes_obj_01.text(1995.5, 4.0, text, horizontalalignment='center', fontsize=10.0,  fontweight="normal")

Axes_obj_01.annotate(
 text='\n'.join(["WHI Report"]),
    xy=(2002, 7.5), xytext=(2002, 9.0), ha="left", va='center', size=10.0,
    bbox=dict(boxstyle='round', edgecolor='red', fc='yellow'),
    arrowprops=dict(
        facecolor='black', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0, edgecolor='black', shrink=0.1)
)

Axes_obj_01.quiver(1998.25, 4.0, 1.75, 4.0,
                   scale_units='xy', angles='xy', scale=1, linestyle='solid', color='red', linewidth=0.4079, width=0.004)

Axes_obj_01.quiver(2003.5, 6.75, 1.75, -4.5,
                   scale_units='xy', angles='xy', scale=1, linestyle='solid', color='red', linewidth=0.4079, width=0.004)

Axes_obj_01.quiver(2007, 3.5, 7.5, 0.0,
                   scale_units='xy', angles='xy', scale=1, linestyle='solid', color='red', linewidth=0.4079, width=0.004)

############################################################################################################
# "(b)
############################################################################################################
Axes_obj_02.set_title("Search keyword in titles of scientific papers: \"Traveller's\" (e.g., Traveller's thrombosis)", size=11.4230, fontweight="normal")
#Axes_obj_02.set_xlabel('Year')
Axes_obj_02.xaxis.set_minor_locator(ticker.MultipleLocator(1))
Axes_obj_02.set_ylabel('Count')
Axes_obj_02.set_xlim(1970, 2025)
Axes_obj_02.set_ylim(0.0, 10)
Axes_obj_02.set_xticks([1970, 1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025])
Axes_obj_02.set_xticklabels(["1970", "1975", "1980", "1985", "1990", "1995", "2000", "2005", "2010", "2015", "2020", "2025"])
Axes_obj_02.yaxis.set_minor_locator(ticker.MultipleLocator(1))

Axes_obj_02.fill_between(np.linspace(1993, 1998, 1000), (10-0.0)*np.ones(1000), (0.0-0.0)*np.ones(1000),
                         facecolor='orange', alpha=0.2, zorder=7)

Axes_obj_02.bar(DataSet_Year_Count["Year"], DataSet_Year_Count["Travellers"], color='blue', width=0.7, alpha=0.3)
Axes_obj_02.bar(DataSet_Year_Count_N0_RV["Year"], DataSet_Year_Count_N0_RV["Travellers"], color='blue', width=0.7, alpha=0.3)

for i in range(0, len(DataSet_Year_Count_Travellers["Travellers"])):
    Axes_obj_02.text(DataSet_Year_Count_Travellers["Year"][i], DataSet_Year_Count_Travellers["Travellers"][i]+0.3,
                     DataSet_Year_Count_Travellers["Travellers"][i], ha='center', va='center', size=8.0, color="gray")

Axes_obj_02.annotate(
 text='\n'.join(["Year", "2000"]),
    xy=(2000, 2.0), xytext=(2000, 6.0), ha='center', va='center', size=10.0,
    bbox=dict(boxstyle='round', edgecolor='blue', fc='white'),
    arrowprops=dict(
        facecolor='black', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0, edgecolor='black', shrink=0.1)
)

Axes_obj_02.annotate(
 text='\n'.join(["Year 1978", "Ikkala E. Duodecim.", "1978;94(4):225.", "Article in Finnish"]),
    xy=(1978, 2.25), xytext=(1978, 6.5), ha='center', va='center', size=10.0,
    bbox=dict(boxstyle='round', edgecolor='blue', fc='white', linestyle="dashed"),
    arrowprops=dict(
        facecolor='black', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0, edgecolor='black', shrink=0.1)
)

text='\n'.join(["1993-1998", "WHI trial for", "HRT", "recruiting", "period"])
Axes_obj_02.text(1995.5, 4.0, text, horizontalalignment='center', fontsize=10.0,  fontweight="normal")

############################################################################################################
# "(c)
############################################################################################################
Axes_obj_03.set_title("Search keyword in titles of scientific papers: \"Travel-related\" (e.g., Travel-related thrombosis)",
                      size=11.4230, fontweight="normal")
#Axes_obj_03.set_xlabel('Year')
Axes_obj_03.xaxis.set_minor_locator(ticker.MultipleLocator(1))
Axes_obj_03.set_ylabel('Count')
Axes_obj_03.set_xlim(1970, 2025)
Axes_obj_03.set_ylim(0.0, 10)
Axes_obj_03.set_xticks([1970, 1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025])
Axes_obj_03.set_xticklabels(["1970", "1975", "1980", "1985", "1990", "1995", "2000", "2005", "2010", "2015", "2020", "2025"])
Axes_obj_03.yaxis.set_minor_locator(ticker.MultipleLocator(1))

Axes_obj_03.fill_between(np.linspace(1993, 1998, 1000), (10-0.0)*np.ones(1000), (0.0-0.0)*np.ones(1000),
                         facecolor='orange', alpha=0.2, zorder=7)

Axes_obj_03.bar(DataSet_Year_Count["Year"], DataSet_Year_Count["Travel-related"], color='green', width=0.7, alpha=0.3)
Axes_obj_03.bar(DataSet_Year_Count_N0_RV["Year"], DataSet_Year_Count_N0_RV["Travel-related"], color='green', width=0.7, alpha=0.3)

for i in range(0, len(DataSet_Year_Count_Travel_related["Travel-related"])):
    Axes_obj_03.text(DataSet_Year_Count_Travel_related["Year"][i], DataSet_Year_Count_Travel_related["Travel-related"][i]+0.3,
                     DataSet_Year_Count_Travel_related["Travel-related"][i], ha='center', va='center', size=8.0, color="gray")

Axes_obj_03.annotate(
 text='\n'.join(["Year", "2001"]),
    xy=(2001, 4.75), xytext=(2001, 8.0), ha='center', va='center', size=10.0,
    bbox=dict(boxstyle='round', edgecolor='green', fc='white'),
    arrowprops=dict(
        facecolor='black', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0, edgecolor='black', shrink=0.1)
)

Axes_obj_03.annotate(
 text='\n'.join(["Year, 2006", "MEGA study", "Cannegieter et al."]),
    xy=(2006, 4.75), xytext=(2006, 7.75), ha='center', va='center', size=10.0,
    bbox=dict(boxstyle='round', edgecolor='green', fc='white'),
    arrowprops=dict(
        facecolor='black', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0, edgecolor='black', shrink=0.1)
)

text='\n'.join(["1993-1998", "WHI trial for", "HRT", "recruiting", "period"])
Axes_obj_03.text(1995.5, 4.0, text, horizontalalignment='center', fontsize=10.0,  fontweight="normal")

############################################################################################################
# "(d)
############################################################################################################
Axes_obj_04.set_title("Time-course changes in conceptual framework for thrombosis research", size=11.4230, fontweight="normal")
#Axes_obj_04.set_title("XXX", size=11.4230, fontweight="normal")
#Axes_obj_04.set_xlabel('Year')
#Axes_obj_04.set_ylabel('Count')

Axes_obj_04.tick_params(length=0.0); Axes_obj_04.set_xticklabels([]); Axes_obj_04.set_yticklabels([])
#Axes_obj_04.spines['right'].set_visible(False)
#Axes_obj_04.spines['left'].set_visible(False)
#Axes_obj_04.spines['top'].set_visible(False)
#Axes_obj_04.spines['bottom'].set_visible(False)

Axes_obj_04.set_xlim(0.0, 100)
Axes_obj_04.set_ylim(0.0, 100)

Axes_obj_04.annotate(
 text='\n'.join(["First research phase"]),
    xy=(12.51, 85), xytext=(12.51, 85), ha='center', va='center', size=10.0,
    bbox=dict(boxstyle='round', edgecolor='white', fc='white'),
    arrowprops=dict(
        facecolor='black', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0, edgecolor='black', shrink=0.1)
)

Axes_obj_04.annotate(
 text='\n'.join(["Second research phase"]),
    xy=(37.53, 85), xytext=(37.53, 85), ha='center', va='center', size=10.0,
    bbox=dict(boxstyle='round', edgecolor='white', fc='white'),
    arrowprops=dict(
        facecolor='black', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0, edgecolor='black', shrink=0.1)
)

Axes_obj_04.annotate(
 text='\n'.join(["Third research phase"]),
    xy=(62.55, 85), xytext=(62.55, 85), ha='center', va='center', size=10.0,
    bbox=dict(boxstyle='round', edgecolor='white', fc='white'),
    arrowprops=dict(
        facecolor='black', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0, edgecolor='black', shrink=0.1)
)

Axes_obj_04.annotate(
 text='\n'.join(["Fifth research phase?"]),
    xy=(87.57, 85), xytext=(87.57, 85), ha='center', va='center', size=10.0,
    bbox=dict(boxstyle='round', edgecolor='white', fc='white'),
    arrowprops=dict(
        facecolor='black', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0, edgecolor='black', shrink=0.1)
)


Axes_obj_04.annotate(
 text='\n'.join(["", "Conceptual framework:", "", " \"Economy Class Syndrome\" ", ""]),
    xy=(12.51+15, 40), xytext=(12.51, 40), ha='center', va='center', size=10.0,
    bbox=dict(boxstyle='round', edgecolor='red', fc='white'),
    arrowprops=dict(
        facecolor='black', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0, edgecolor='black', shrink=0.1)
)

Axes_obj_04.annotate(
 text='\n'.join(["", "Conceptual framework:", "", " \"Traveller's\" thrombosis ", ""]),
    xy=(37.53+15, 40), xytext=(37.53, 40), ha='center', va='center', size=10.0,
    bbox=dict(boxstyle='round', edgecolor='blue', fc='white'),
    arrowprops=dict(
        facecolor='black', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0, edgecolor='black', shrink=0.1)
)

Axes_obj_04.annotate(
 text='\n'.join(["", "Conceptual framework:", "", " \"Travel-related\" thrombosis ", ""]),
    xy=(62.55+15, 40), xytext=(62.55, 40), ha='center', va='center', size=10.0,
    bbox=dict(boxstyle='round', edgecolor='green', fc='white'),
    arrowprops=dict(
        facecolor='black', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0, edgecolor='black', shrink=0.1)
)

Axes_obj_04.annotate(
 text='\n'.join(["", "More appropriate or more", "comprehensive conceptual", "framework for thrombosis", " research is needed?", ""]),
    xy=(87.57, 40), xytext=(87.57, 40), ha='center', va='center', size=10.0,
    bbox=dict(boxstyle='round', edgecolor='Orange', fc='lemonchiffon'),
    arrowprops=dict(
        facecolor='black', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0, edgecolor='black', shrink=0.1)
)

text='\n'.join(["1993-1998", "WHI trial for", "HRT", "recruiting", "period"])
Axes_obj_04.text(1995.5, 4.0, text, horizontalalignment='center', fontsize=10.0,  fontweight="normal")

############################################################################################################


#Figure_object.tight_layout()
my_path = os.path.abspath("Figures")
plt.savefig(os.path.join(my_path, "01_Kimoto_et_al_(2023)_[Fig_S_06].png"))

from PIL import Image # py -m pip install pillow
img = Image.open(os.path.join(my_path, "01_Kimoto_et_al_(2023)_[Fig_S_06].png"))
img_resize = img.resize(size=(2866, 2016))
my_path2 = os.path.abspath("Figures/Fixed_Version/B6")
img_resize.save(os.path.join(my_path2, "01_Kimoto_et_al_(2023)_[Fig_S_06]_B6.png"))

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


