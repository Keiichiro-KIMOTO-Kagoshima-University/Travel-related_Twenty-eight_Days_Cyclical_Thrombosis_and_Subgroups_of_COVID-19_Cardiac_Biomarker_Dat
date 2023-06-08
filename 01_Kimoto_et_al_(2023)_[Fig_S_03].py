########################################################################################################################
# Fig. S3. Curve fitting to the dataset reported by Philbrick et al. (01_Kimoto_et_al_(2023)_[Fig_S_03].py)
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
# Philbrick JT, Shumate R, Siadaty MS, Becker DM. Air travel and venous thromboembolism: a systematic review.
# J Gen Intern Med. 2007 Jan;22(1):107-14. doi: 10.1007/s11606-006-0016-0. PMID: 17351849; PMCID: PMC1824715.

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
############################################################################################################
########### Philbrick DataSet (Philbrick et al. 2007) (PhilbrickDataSet_Table2) ##########
############################################################################################################
#Clerel = [1, "Clerel", "Bull Acad Natl Med",  10,  1999, "32000000", "Air", "mean 12.7 h", "", "", "12.7", "PE", "15", "32000000", "–", "0.5", "ppm", "(", "0.3", "ppm", "–", "0.8", "ppm", ")", "–", "per million"]


Clerel = [1, "Clérel", "Bull Acad Natl Med",  10,  1999, "241000000", "Air", "mean 12.7 h", "", "", "12.7", "PE", "64", "241000000", "–", "0.5", "ppm", "(", "0.3", "ppm", "–", "0.8", "ppm", ")", "–", "per million"]
Lapostolle_1 = [2, "Lapostolle", "N Engl J Med",  17,  2001, "135290000", "Air", "<3 h", "0", "3", "1.5", "PE", "0", "88490000", "–", "0", "ppm", "(", "0", "ppm", "–", "0.04", "ppm", ")", "†", "per million"]
Lapostolle_2 = [3, "Lapostolle", "N Engl J Med",  17,  2001, "135290000", "Air", ">= 3 to < 6 h", "3", "6", "4.5", "PE", "1", "9180000", "–", "0.11", "ppm", "(", "0.01", "ppm", "–", "0.71", "ppm", ")", "–", "per million"]
Lapostolle_3 = [4, "Lapostolle", "N Engl J Med",  17,  2001, "135290000", "Air", ">= 6 to < 9 h", "6", "9", "7.5", "PE", "9", "22530000", "–", "0.4", "ppm", "(", "0.19", "ppm", "–", "0.79", "ppm", ")", "–", "per million"]
Lapostolle_4 = [5, "Lapostolle", "N Engl J Med",  17,  2001, "135290000", "Air", ">= 9 to < 12 h", "9", "12", "10.5", "PE", "33", "12370000", "–", "2.66", "ppm", "(", "1.83", "ppm", "–", "3.79", "ppm", ")", "–", "per million"]
Lapostolle_5 = [6, "Lapostolle", "N Engl J Med",  17,  2001, "135290000", "Air", ">= 12 h", "12", "24", "13.5", "PE", "13", "2720000", "–", "4.77", "ppm", "(", "2.66", "ppm", "–", "8.41", "ppm", ")", "–", "per million"]
Belcaro_1 = [7, "Belcaro", "Angiology",  18,  2001, "778", "Air", "10–15 h", "10", "15", "12.5", "Low-risk‡ DVT", "0", "355", "–", "0", "%", "(", "0", "%", "–", "1", "%", ")", "†", "%"]
Belcaro_2 = [8, "Belcaro", "Angiology",  18,  2001, "778", "Air", "10–15 h", "10", "15", "12.5", "High-risk‡ DVT", "11", "389", "–", "2.8", "%", "(", "1.4", "%", "–", "5", "%", ")", "–", "%"]
Belcaro_3 = [9, "Belcaro", "Angiology",  18,  2001, "778", "Air", "10–15 h", "10", "15", "12.5", "DVT", "11", "744", "–", "", "%", "(", "", "%", "–", "", "%", ")", "–", "%"]
Schwarz_1 = [10, "Schwarz", "Blood Coagul Fibrinolysis",  19,  2002, "320", "Air", ">8 h", "8", "24", "12", "DVT", "0", "160", "–", "0", "%", "(", "0", "%", "–", "2.3", "%", ")", "–", "%"]
Schwarz_2 = [11, "Schwarz", "Blood Coagul Fibrinolysis",  19,  2002, "320", "–", "–", "", "", "", "DVT", "0", "160", "–", "0", "%", "(", "0", "%", "–", "2.3", "%", ")", "–", "%"]
Schwarz_3 = [12, "Schwarz", "Arch Intern Med",  20,  2003, "2311", "Air", ">8 h", "8", "24", "12", "DVT", "7", "964", "–", "0.7", "%", "(", "0.3", "%", "–", "1.5", "%", ")", "–", "%"]
Schwarz_4 = [13, "Schwarz", "Arch Intern Med",  20,  2003, "2311", "–", "–", "", "", "", "DVT", "2", "1213", "–", "0.2", "%", "(", "0.02", "%", "–", "0.6", "%", ")", "–", "%"]
Perez_Rodriguez_1 = [14, "Pérez-Rodríguez", "Arch Intern Med",  21,  2003, "41035332", "Air", "<6 h", "0", "6", "3", "PE", "0", "28038726", "–", "0", "ppm", "(", "", "ppm", "–", "", "ppm", ")", "†", "per million"]
Perez_Rodriguez_2 = [15, "Pérez-Rodríguez", "Arch Intern Med",  21,  2003, "41035332", "Air", "6–8 h", "6", "8", "7", "PE", "1", "3926208", "–", "0.25", "ppm", "(", "0", "ppm", "–", "0.75", "ppm", ")", "–", "per million"]
Perez_Rodriguez_3 = [16, "Pérez-Rodríguez", "Arch Intern Med",  21,  2003, "41035332", "Air", ">8 h", "8", "24", "11", "PE", "15", "9070398", "–", "1.65", "ppm", "(", "0.81", "ppm", "–", "2.49", "ppm", ")", "–", "per million"]
Hughes_1 = [17, "Hughes", "Lancet",  22,  2003, "878", "Air", "39.4 h (mean)", "", "", "", "VTE", "9", "878", "PE:4;DVT:5", "1.03", "%", "(", "0.5", "%", "–", "1.9", "%", ")", "–", "%"]
Hughes_2 = [18, "Hughes", "Lancet",  22,  2003, "878", "Air", "<24 h", "0", "24", "12", "VTE", "0", "123", "–", "0", "%", "(", "0.5", "%", "–", "3", "%", ")", "–", "%"]
Hughes_3 = [19, "Hughes", "Lancet",  22,  2003, "878", "Air", ">24 h", "", "", "", "VTE", "9", "752", "PE:4;DVT:5", "1.2", "%", "(", "0.6", "%", "–", "2.3", "%", ")", "–", "%"]
Kelman = [20, "Kelman", "Lancet",  23,  2003, "9257842", "Air", "–", "", "", "", "VTE", "246", "9257842", "–", "26.6", "ppm", "(", "23", "ppm", "–", "30", "ppm", ")", "–", "per million"]
Jacobson = [21, "Jacobson", "S Afr Med J",  24,  2003, "899", "Air", "11 h", "", "", "11", "DVT", "0", "434", "–", "0", "%", "(", "0", "%", "–", "0.9", "%", ")", "–", "%"]
Gajic_1 = [22, "Gajic", "Mayo Clin Proc",  25,  2005, "8860", "Air", ">5000 km", "", "", "", "VTE", "11", "223", "–", "4.9", "%", "(", "2.5", "%", "–", "8.7", "%", ")", "†", "%"]
Gajic_2 = [23, "Gajic", "Mayo Clin Proc",  25,  2005, "8860", "Air", "<5000 km", "", "", "", "VTE", "13", "8637", "–", "0.2", "%", "(", "0.08", "%", "–", "0.3", "%", ")", "–", "%"]

PhilbrickDataSet_Table2 = pd.DataFrame(
    [Clerel, Lapostolle_1, Lapostolle_2, Lapostolle_3, Lapostolle_4, Lapostolle_5, Belcaro_1, Belcaro_2, Belcaro_3,
     Schwarz_1, Schwarz_2, Schwarz_3, Schwarz_4, Perez_Rodriguez_1, Perez_Rodriguez_2, Perez_Rodriguez_3, Hughes_1,
     Hughes_2, Hughes_3, Kelman, Jacobson, Gajic_1, Gajic_2],
   index=["Clerel", "Lapostolle_1", "Lapostolle_2", "Lapostolle_3", "Lapostolle_4", "Lapostolle_5", "Belcaro_1",
          "Belcaro_2", "Belcaro_3", "Schwarz_1", "Schwarz_2", "Schwarz_3", "Schwarz_4", "Perez_Rodriguez_1",
          "Perez_Rodriguez_2", "Perez_Rodriguez_3", "Hughes_1", "Hughes_2", "Hughes_3", "Kelman", "Jacobson",
          "Gajic_1", "Gajic_2"],
   columns=["No.", "Author", "Journal", "Ref", "Year", "Subjects", "Travel", "Duration", "Min", "Max", "Mean Time",
            "Type", "with VTE", "Total", "Note", "Incidence", "U1", "P1", "CI (L)", "U2", "H1", "CI (U)", "U3", "P2",
            "Sig.", "Unit"]
)

for i in PhilbrickDataSet_Table2.columns:
    PhilbrickDataSet_Table2[i] = PhilbrickDataSet_Table2[i].replace('', np.nan)

PhilbrickDataSet_Table2["Subjects"] = PhilbrickDataSet_Table2["Subjects"].astype(int)
PhilbrickDataSet_Table2["Min"] = PhilbrickDataSet_Table2["Min"].astype(float)
PhilbrickDataSet_Table2["Max"] = PhilbrickDataSet_Table2["Max"].astype(float)
PhilbrickDataSet_Table2["Mean Time"] = PhilbrickDataSet_Table2["Mean Time"].astype(float)
PhilbrickDataSet_Table2["with VTE"] = PhilbrickDataSet_Table2["with VTE"].astype(float)
PhilbrickDataSet_Table2["Total"] = PhilbrickDataSet_Table2["Total"].astype(float)
PhilbrickDataSet_Table2["Incidence"] = PhilbrickDataSet_Table2["Incidence"].astype(float)
PhilbrickDataSet_Table2["CI (L)"] = PhilbrickDataSet_Table2["CI (L)"].astype(float)
PhilbrickDataSet_Table2["CI (U)"] = PhilbrickDataSet_Table2["CI (U)"].astype(float)

print("")
print("###############################################################################################################")

#import ebcic
#from ebcic import *
#print_interval(Params(k=1, n=501255, confi_perc=95.0))[0]


import statsmodels.stats.proportion as st

Results = st.proportion_confint(PhilbrickDataSet_Table2["with VTE"],
                                 PhilbrickDataSet_Table2["Total"], alpha=0.05, method='wilson')

P = PhilbrickDataSet_Table2["with VTE"] / PhilbrickDataSet_Table2["Total"]
PhilbrickDataSet_Table2["[Exact] Calc %"] = (PhilbrickDataSet_Table2["with VTE"] / PhilbrickDataSet_Table2["Total"]) * 100
PhilbrickDataSet_Table2["P3"] = "("
PhilbrickDataSet_Table2["[Exact] CI(L) %"] = Results[0] * 100
PhilbrickDataSet_Table2["H2"] = "–"
PhilbrickDataSet_Table2["[Exact] CI(U) %"] = Results[1] * 100
PhilbrickDataSet_Table2["P4"] = ")"

P = PhilbrickDataSet_Table2["with VTE"] / PhilbrickDataSet_Table2["Total"]
PhilbrickDataSet_Table2["[Exact] Calc PM"] = (PhilbrickDataSet_Table2["with VTE"] / PhilbrickDataSet_Table2["Total"]) * 1000000
PhilbrickDataSet_Table2["P5"] = "("
PhilbrickDataSet_Table2["[Exact] CI(L) PM"] = Results[0] * 1000000
PhilbrickDataSet_Table2["H3"] = "–"
PhilbrickDataSet_Table2["[Exact] CI(U) PM"] = Results[1] * 1000000
PhilbrickDataSet_Table2["P6"] = ")"


print("")
print("###############################################################################################################")
print("########## Philbrick DataSet Table2 (Philbrick et al. 2007) (PhilbrickDataSet_Table2) ##########")
print(PhilbrickDataSet_Table2)


print("")
print("###############################################################################################################")
Data_per_M = PhilbrickDataSet_Table2[PhilbrickDataSet_Table2["Unit"] == "per million"]
Data_per_M = Data_per_M.drop(
    ["No.", "Journal", "Ref", "Subjects", "Min", "Max", "Note", "Sig.", "Unit", "[Exact] Calc %", "P3",  "[Exact] CI(L) %", "H2",  "[Exact] CI(U) %", "P4"], axis=1)
print("########## Philbrick DataSet Table2 (per million) (Data_per_M) ##########")
print(Data_per_M)
Data_per_M = Data_per_M.drop("Kelman", axis=0)
print("")
print("########## Drop Kelman ##########")
print(Data_per_M)
print("")
print("########## Sort Mean Time ##########")
Data_per_M = Data_per_M.sort_values(by="Mean Time")
Data_per_M = Data_per_M.rename(index={"Lapostolle_1": "01",
                                      "Perez_Rodriguez_1": "02",
                                      "Lapostolle_2": "03",
                                      "Perez_Rodriguez_2": "04",
                                      "Lapostolle_3": "05",
                                      "Lapostolle_4": "06",
                                      "Perez_Rodriguez_3": "07",
                                      "Clerel": "08",
                                      "Lapostolle_5": "09"}
                               )
print(Data_per_M)

print("")
print("########## Sort Author ##########")
Data_per_M_Systematic_Review = Data_per_M.sort_values(by="Author")
print(Data_per_M_Systematic_Review)


print("")
print("###############################################################################################################")
Data_per_C = PhilbrickDataSet_Table2[PhilbrickDataSet_Table2["Unit"] == "%"]
Data_per_C = Data_per_C.drop(
    ["No.", "Journal", "Ref", "Subjects", "Min", "Max", "Note", "Sig.", "Unit", "[Exact] Calc PM", "P5",  "[Exact] CI(L) PM", "H3",  "[Exact] CI(U) PM", "P6"], axis=1)
print("########## Philbrick DataSet Table2 (per cent) (Data_per_C) ##########")
print(Data_per_C)
Data_per_C = Data_per_C.drop(
    ["Belcaro_1", "Belcaro_2", "Schwarz_2", "Schwarz_4", "Hughes_1", "Hughes_2", "Hughes_3", "Gajic_1", "Gajic_2"],
    axis=0)
print("")
print("########## Drop Belcaro_1, Belcaro_2, Schwarz_2, Schwarz_4, Hughes_1, Hughes_2, Hughes_3, Gajic_1, Gajic_2"
      " ##########")
print(Data_per_C)
print("")
print("########## Sort Mean Time ##########")
Data_per_C = Data_per_C.sort_values(by="Mean Time")
Data_per_C = Data_per_C.rename(index={"Jacobson": "01", "Schwarz_1": "02", "Schwarz_3": "03", "Belcaro_3": "04"})
print(Data_per_C)

print("########## Sort Author ##########")
Data_per_C_Systematic_Review = Data_per_C.sort_values(by="Author")
print(Data_per_C_Systematic_Review)

print("")
print("###############################################################################################################")
print("########## Label Data (LabelDataSet) ##########")
Label_01 = ["01", 1.5, 1.5, "red", "per_M", "Lapostolle  2001  (<3 h) [1.5 h]"]
Label_02 = ["02", 3.0, 3.0,  "red", "per_M", "Pérez-Rodríguez  2003   (<6 h) [3 h]"]
Label_03 = ["03", 4.5, 4.5,  "red", "per_M", "Lapostolle  2001  (3-6 h) [4.5 h]"]
Label_04 = ["04", 7.0, 6.75,  "red", "per_M", "Pérez-Rodríguez  2003  (6–8 h) [7 h]"]
Label_05 = ["05", 7.5, 7.75,  "red", "per_M", "Lapostolle  2001  (6-9 h) [7.5 h]"]
Label_06 = ["06", 10.5, 11,   "red", "per_M", "Lapostolle  2001  (9-12 h) [10.5 h]"]
Label_07 = ["07", 11.0, 12,   "red", "per_M", "Pérez-Rodríguez  2003 (>8 h) [11 h]"]
Label_08 = ["01", 11.2, 13,  "blue", "per_C", "Jacobson  2003  (11 h)"]
Label_09 = ["02", 11.8, 14,   "blue", "per_C", "Schwarz  2002  (>8 h) [12 h]"]
Label_10 = ["03", 12.2, 15,   "blue", "per_C", "Schwarz  2003  (>8 h) [12 h]"]
Label_11 = ["04", 12.5, 16,  "blue", "per_C", "Belcaro  2001  (10–15 h) [12.5 h]"]
Label_12 = ["08", 12.7, 17, "red", "per_M", "Clérel  1999  (mean 12.7 h)"]
Label_13 = ["09", 13.5, 18, "red", "per_M", "Lapostolle  2001  (>= 12 h) [13.5 h]"]

LabelDataSet = pd.DataFrame(
    [Label_01, Label_02, Label_03, Label_04, Label_05, Label_06, Label_07, Label_08, Label_09, Label_10,
     Label_11, Label_12, Label_13],
    index=["Label_01", "Label_02", "Label_03", "Label_04", "Label_05",
           "Label_06", "Label_07", "Label_08", "Label_09", "Label_10", "Label_11", "Label_12", "Label_13"],
    columns=["No", "Point_x", "Label_Point_x", "LColor", "Type", "Label"])
print(LabelDataSet)
########################################################################################################################
# Regression Analysis
########################################################################################################################

############################################################################################################
# Definition of Regression Function
############################################################################################################
def reg_func_s1(parameter: ["k1", "k2", "l", "m"], x, y):
    k1 = parameter[0]
    k2 = 0.0
    l = 1.0
    m = parameter[1]
    _Output_ = y - (k1 / (1 + np.exp(-l * (x - m))) + k2)
    return _Output_

def reg_func_s2(parameter: ["k1", "k2", "l", "m"], x, y):
    k1 = parameter[0]
    k2 = 0.0
    l = parameter[1]
    m = parameter[2]
    _Output_ = y - (k1 / (1 + np.exp(-l * (x - m))) + k2)
    return _Output_
############################################################################################################
# Non-linear Regression Analysis on "Risk Group A"
############################################################################################################
# S Curve
x = Data_per_M["Mean Time"]
y = Data_per_M["[Exact] Calc PM"]
parameter_0 = [3.0, 9.0]
RegResult_Original_AS = optimize.leastsq(reg_func_s1, parameter_0, args=(x, y), full_output=True)
Result_AS0 = RegResult_Original_AS[0]
Result_AS = [Result_AS0[0], 0.0,  1.0, Result_AS0[1]]

############################################################################################################
# Non-linear Regression Analysis on "Risk Group  B"
############################################################################################################
# S Curve
x = Data_per_C["Mean Time"]
y = Data_per_C["[Exact] Calc %"]
parameter_0 = [50,  1.5, 14.0]
RegResult_Original_BS = optimize.leastsq(reg_func_s2, parameter_0, args=(x, y), full_output=True)
Result_BS0 = RegResult_Original_BS[0]
Result_BS = [Result_BS0[0], 0.0, Result_BS0[1], Result_BS0[2]]

############################################################################################################
# Evaluation of Result on the Regression Analysis
############################################################################################################
print("")
print("########## Evaluation of Result on the Regression Analysis ##########")
print("Reg. Result on A (S Curve)         ", "Evaluation of Result:", RegResult_Original_AS[-1],
      "(1, 2, 3 or 4, the solution was found.)")
print("Reg. Result on A (CI Curve (U))    ", "Evaluation of Result:", RegResult_Original_BS[-1],
      "(1, 2, 3 or 4, the solution was found.)")

CoefficientDataSet = pd.DataFrame([Result_AS, Result_BS],
                                  index=["Reg. Result on A (S Curve)         ",
                                         "Reg. Result on B (S Curve)         "],
                                  columns=["K1", "K2", "L", "M (Mean)"])

print("")
print("########## Result of Linear Regression Analysis (CoefficientDataSet_L) ##########")
print(CoefficientDataSet)

############################################################################################################
# Regression Curves on Risk Group A
############################################################################################################
X_A_S = np.linspace(0.0, 20.0, 1000)

# Y_AS, Y_AS_Diff
k1 = CoefficientDataSet.at["Reg. Result on A (S Curve)         ", "K1"]
k2 = CoefficientDataSet.at["Reg. Result on A (S Curve)         ", "K2"]
l = CoefficientDataSet.at["Reg. Result on A (S Curve)         ", "L"]
m = CoefficientDataSet.at["Reg. Result on A (S Curve)         ", "M (Mean)"]
Y_AS = k1 / (1 + np.exp(-l * (X_A_S - m))) + k2

Point_Y_A = k1 / (1 + np.exp(-l * (m - m))) + k2
Point_X_A = m
############################################################################################################
# Regression Curves on Risk Group B
############################################################################################################
X_B_S = np.linspace(0.0, 20.0, 1000)

# Y_BS, Y_BS_Diff
k1 = CoefficientDataSet.at["Reg. Result on B (S Curve)         ", "K1"]
k2 = CoefficientDataSet.at["Reg. Result on B (S Curve)         ", "K2"]
l = CoefficientDataSet.at["Reg. Result on B (S Curve)         ", "L"]
m = CoefficientDataSet.at["Reg. Result on B (S Curve)         ", "M (Mean)"]
Y_BS = k1 / (1 + np.exp(-l * (X_B_S - m))) + k2

Point_Y_B = k1 / (1 + np.exp(-l * (m - m))) + k2
Point_X_B = m
########################################################################################################################
# Figures
########################################################################################################################
Figure_object = plt.figure(1, figsize=(11.69, 8.27), dpi=400, edgecolor="black", linewidth=0.5)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)

plt.figtext(0.0200, 0.970, "a", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")
plt.figtext(0.5000, 0.970, "b", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")

gs_master = matplotlib.gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[1, 1])

Axes_obj_01 = Figure_object.add_subplot(gs_master[0])
Axes_obj_03 = Figure_object.add_subplot(gs_master[1])

############################################################################################################
# "(a) "
############################################################################################################
Axes_obj_01.set_title("From Philbrick et al. 2007, Systematic Review", size=11.4230, fontweight="normal")
Axes_obj_01.set_xlabel('')
Axes_obj_01.set_ylabel('')
Axes_obj_01.set_xlim(0.0, 28.0)
Axes_obj_01.set_ylim(-0.5, 16.5)
Axes_obj_01.tick_params(length=0.0)
Axes_obj_01.set_xticklabels([])
Axes_obj_01.set_yticklabels([])

Axes_obj_01.vlines(x=1.00, ymin=-0.25, ymax=16.25, color="red", linestyle='solid', linewidth=1.0)
Axes_obj_01.vlines(x=13.0, ymin=-0.25, ymax=16.25, color="red", linestyle='solid', linewidth=1.0)
Axes_obj_01.vlines(x=17.0, ymin=-0.25, ymax=16.25, color="blue", linestyle='solid', linewidth=1.0)
Axes_obj_01.vlines(x=25.0, ymin=-0.25, ymax=16.25, color="blue", linestyle='solid', linewidth=1.0)

Axes_obj_01.hlines(y=-0.25, xmin=1.00, xmax=13.0, color="red", linestyle="solid", linewidth=1.0)
Axes_obj_01.hlines(y=16.25, xmin=1.00, xmax=13.0, color="red", linestyle="solid", linewidth=1.0)
Axes_obj_01.hlines(y=-0.25, xmin=17.0, xmax=25.0, color="blue", linestyle="solid", linewidth=1.0)
Axes_obj_01.hlines(y=16.25, xmin=17.0, xmax=25.0, color="blue", linestyle="solid", linewidth=1.0)

Axes_obj_01.vlines(x=3.00, ymin=0.0, ymax=16.0, color="gray", linestyle='dashed', linewidth=1.0)
Axes_obj_01.vlines(x=9.00, ymin=0.0, ymax=16.0, color="gray", linestyle='dashed', linewidth=1.0)
Axes_obj_01.vlines(x=19.0, ymin=0.0, ymax=16.0, color="gray", linestyle='dashed', linewidth=1.0)
Axes_obj_01.vlines(x=21.0, ymin=0.0, ymax=16.0, color="gray", linestyle='dashed', linewidth=1.0)
Axes_obj_01.vlines(x=23.0, ymin=0.0, ymax=16.0, color="gray", linestyle='dashed', linewidth=1.0)

Axes_obj_01.hlines(y=0.0, xmin=1.00, xmax=13.0, color="gray", linestyle="dotted", linewidth=0.5)
Axes_obj_01.hlines(y=0.0, xmin=17.0, xmax=25.0, color="gray", linestyle="dotted", linewidth=0.5)

Axes_obj_01.annotate(text="PE Frequency (per million)",
                     xy=(15, 17/2), ha='center', va='center', color="red", size=10, weight="normal",
                     bbox=dict(boxstyle='square', edgecolor='white', fc="white"), rotation=90.0)

Axes_obj_01.annotate(text="DVT Frequency (%)",
                     xy=(27, 17/2), ha='center', va='center', color="blue", size=10, weight="normal",
                     bbox=dict(boxstyle='square', edgecolor='white', fc="white"), rotation=90.0)


Y_AXIS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
Y_AXIS_L = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16"]
for i in range(0, 17):
    Axes_obj_01.plot([13.0, 13.20], [Y_AXIS[i], Y_AXIS[i]], color="red")
    Axes_obj_01.text(13.35, Y_AXIS[i], Y_AXIS_L[i],
                     horizontalalignment='left', verticalalignment='center', fontsize=10.0, color="red")

X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
X1 = [2, 4, 5, 6, 7, 8, 10, 11, 12]
X2 = [18, 20, 22, 24]



for i in range(0, 9):
    Axes_obj_01.scatter(X1[i], Data_per_M_Systematic_Review["[Exact] Calc PM"][i], color="red")
    Axes_obj_01.scatter(X1[i], Data_per_M_Systematic_Review["[Exact] CI(U) PM"][i], color="red", marker='_')
    Axes_obj_01.scatter(X1[i], Data_per_M_Systematic_Review["[Exact] CI(L) PM"][i], color="red", marker='_')
    Axes_obj_01.vlines(x=X1[i],
                       ymin=Data_per_M_Systematic_Review["[Exact] CI(L) PM"][i],
                       ymax=Data_per_M_Systematic_Review["[Exact] CI(U) PM"][i], color="red", linestyle='solid', linewidth=1.0)
    Axes_obj_01.annotate(text=Data_per_M_Systematic_Review["Author"][i],
                         xy=(X1[i], 11.85), ha='center', va='top', color="red", size=10, weight="normal",
                         bbox=dict(boxstyle='square', edgecolor='white', fc="white"), rotation=90.0)
    Axes_obj_01.annotate(text=Data_per_M_Systematic_Review["Year"][i],
                         xy=(X1[i], 12.75), ha='center', va='center', color="red", size=10, weight="normal",
                         bbox=dict(boxstyle='square', edgecolor='white', fc="white"), rotation=90.0)
    Axes_obj_01.annotate(text=Data_per_M_Systematic_Review["Duration"][i],
                         xy=(X1[i], 14.75), ha='center', va='center', color="red", size=10, weight="normal",
                         bbox=dict(boxstyle='square', edgecolor='white', fc="white"), rotation=90.0)


for i in range(0, 4):
    Axes_obj_01.annotate(text=Data_per_C_Systematic_Review["Author"][i],
                         xy=(X2[i], 11.85), ha='center', va='top', color="blue", size=10, weight="normal",
                         bbox=dict(boxstyle='square', edgecolor='white', fc="white"), rotation=90.0)
    Axes_obj_01.annotate(text=Data_per_C_Systematic_Review["Year"][i],
                         xy=(X2[i], 12.75), ha='center', va='center', color="blue", size=10, weight="normal",
                         bbox=dict(boxstyle='square', edgecolor='white', fc="white"), rotation=90.0)
    Axes_obj_01.annotate(text=Data_per_C_Systematic_Review["Duration"][i],
                         xy=(X2[i], 14.75), ha='center', va='center', color="blue", size=10, weight="normal",
                         bbox=dict(boxstyle='square', edgecolor='white', fc="white"), rotation=90.0)

Axes_obj_02 = Axes_obj_01.twinx()
Axes_obj_02.set_ylim(-9.5/(16/0.5), 9.5+(9.5/(16/0.5)))
Axes_obj_02.tick_params(length=0.0)
Axes_obj_02.set_xticklabels([])
Axes_obj_02.set_yticklabels([])

for i in range(0, 4):
    Axes_obj_02.scatter(X2[i], Data_per_C_Systematic_Review["[Exact] Calc %"][i], color="blue")
    Axes_obj_02.scatter(X2[i], Data_per_C_Systematic_Review["[Exact] CI(U) %"][i], color="blue", marker='_')
    Axes_obj_02.scatter(X2[i], Data_per_C_Systematic_Review["[Exact] CI(L) %"][i], color="blue", marker='_')
    Axes_obj_02.vlines(x=X2[i],
                       ymin=Data_per_C_Systematic_Review["[Exact] CI(L) %"][i], ymax=Data_per_C_Systematic_Review["[Exact] CI(U) %"][i],
                       color="blue", linestyle='solid', linewidth=1.0)

Y_AXIS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
Y_AXIS_L = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
for i in range(0, 10):
    Axes_obj_02.plot([25.0, 25.20], [Y_AXIS[i], Y_AXIS[i]], color="blue")
    Axes_obj_02.text(25.35, Y_AXIS[i], Y_AXIS_L[i],
                     horizontalalignment='left', verticalalignment='center', fontsize=10.0, color="blue")

############################################################################################################
# "(b)
############################################################################################################
Axes_obj_03.set_title("Regression Analysis by Disease Type", size=11.4230,  fontweight="normal")
Axes_obj_03.set_xlabel('Duration, hr', fontweight="normal")
Axes_obj_03.set_ylabel('PE Frequency (per million)', fontweight="normal")
Axes_obj_03.set_xlim(-1.0, 21.0)
Axes_obj_03.set_ylim(-0.5, 16.5)
Axes_obj_03.set_xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24])
Axes_obj_03.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
Axes_obj_03.xaxis.set_minor_locator(ticker.MultipleLocator(1.0))
Axes_obj_03.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
Axes_obj_03.yaxis.label.set_color("red"),
Axes_obj_03.tick_params(axis="y", colors="red")

#for i in range(0, 25):
#    Axes_obj_03.vlines(x=i, ymin=0, ymax=8.5, color="gray", linestyle="dotted", linewidth=0.5)
#Axes_obj_03.hlines(y=8.5, xmin=0, xmax=20, color="gray", linestyle="dotted", linewidth=0.5)

Axes_obj_03.vlines(x=0.0, ymin=0.0, ymax=16.25, color="gray", linestyle="dotted", linewidth=0.5)
Axes_obj_03.vlines(x=20.0, ymin=0.0, ymax=16.25, color="gray", linestyle="dotted", linewidth=0.5)

Axes_obj_03.hlines(y=0.0, xmin=0, xmax=20, color="gray", linestyle="dotted", linewidth=0.5)
Axes_obj_03.hlines(y=16.25, xmin=0, xmax=20, color="gray", linestyle="dotted", linewidth=0.5)

Axes_obj_03.plot(X_A_S, Y_AS, color="red", linestyle="solid", linewidth=1.6318, zorder=10)

for i in range(0, len(Data_per_M["Mean Time"])):
    Axes_obj_03.vlines(x=Data_per_M["Mean Time"][i], ymin=0, ymax=8.5, color="gray", linestyle="dotted", linewidth=0.5)
    Axes_obj_03.scatter(Data_per_M["Mean Time"][i], 8.5, color="red", zorder=10)
    Axes_obj_03.scatter(Data_per_M["Mean Time"][i], Data_per_M["[Exact] Calc PM"][i],
                        color="red")
    Axes_obj_03.scatter(Data_per_M["Mean Time"][i], Data_per_M["[Exact] CI(U) PM"][i],
                        color="red", marker='_')
    Axes_obj_03.scatter(Data_per_M["Mean Time"][i], Data_per_M["[Exact] CI(L) PM"][i],
                        color="red", marker='_')
    Axes_obj_03.vlines(x=Data_per_M["Mean Time"][i],
                       ymin=Data_per_M["[Exact] CI(L) PM"][i], ymax=Data_per_M["[Exact] CI(U) PM"][i],
                       color="red", linestyle='solid', linewidth=1.0)

#Axes_obj_03.text(5.0, 8.0, "Time < 10h", size=10, color="black",ha='center', va='center',
#                 bbox=dict(boxstyle='round', edgecolor='black', fc='white'))
#Axes_obj_03.text(15.0, 8.0, "Time > 10h", size=10, color="black", ha='center', va='center',
#                 bbox=dict(boxstyle='round', edgecolor='black', fc='white'))

#Axes_obj_03.quiver(3.0, 8.0, -3.0, 0.0, scale_units='xy', angles='xy', scale=1, linestyle='solid', linewidth=0.25)
#Axes_obj_03.quiver(5.0, 8.0, 5.0, 0.0, scale_units='xy', angles='xy', scale=1, linestyle='solid', linewidth=0.25)
#Axes_obj_03.quiver(13.0, 8.0, -3.0, 0.0, scale_units='xy', angles='xy', scale=1, linestyle='solid', linewidth=0.25)
#Axes_obj_03.quiver(17.0, 8.0, 3.0, 0.0, scale_units='xy', angles='xy', scale=1, linestyle='solid', linewidth=0.25)

#Axes_obj_03.vlines(x=10.0, ymin=-0.25, ymax=16.25, color="black", linestyle='solid', linewidth=1.0)
#Axes_obj_03.vlines(x=0.0, ymin=8.0-0.25, ymax=8.0+0.25, color="black", linestyle='solid', linewidth=1.0)
#Axes_obj_03.vlines(x=20.0, ymin=8.0-0.25, ymax=8.0+0.25, color="black", linestyle='solid', linewidth=1.0)

Axes_obj_03.vlines(x=Point_X_A, ymin=-0.5, ymax=Point_Y_A, color="red", linestyle='dashed', linewidth=1.0)

Axes_obj_03.annotate(
 text='\n'.join(["PE", "9.2 h"]),
    xy=(Point_X_A, 5.0), xytext=(Point_X_A, 6.0), ha='center', va='center', size=10.0,
    bbox=dict(boxstyle='round', edgecolor='red', fc='white'),
    arrowprops=dict(
        facecolor='red', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0, edgecolor='red', shrink=0.1)
)
############################################################################################################
# "()
############################################################################################################
Axes_obj_04 = Axes_obj_03.twinx()

Axes_obj_04.set_ylabel('DVT Frequency (%)', fontweight="normal")
Axes_obj_04.set_xlim(-1.0, 21.0)
Axes_obj_04.set_ylim(-9.5/(16/0.5), 9.5+(9.5/(16/0.5)))
Axes_obj_04.yaxis.label.set_color("blue")
Axes_obj_04.tick_params(axis="y", colors="blue")
Axes_obj_04.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
Axes_obj_04.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))

Axes_obj_04.plot(X_B_S, Y_BS, color="blue", linestyle="solid", linewidth=1.6318, zorder=1)

X = [11.0+0.2, 12.0-0.2, 12.0+0.2, 12.5]
for i in range(0, len(Data_per_C["Mean Time"])):
    Axes_obj_04.vlines(x=X[i], ymin=0, ymax=8.5*(9.5/16), color="gray", linestyle="dotted", linewidth=0.5)
    Axes_obj_04.scatter(X[i], Data_per_C["[Exact] Calc %"][i],
                        color="blue")
    Axes_obj_04.scatter(X[i], Data_per_C["[Exact] CI(U) %"][i],
                        color="blue", marker='_')
    Axes_obj_04.scatter(X[i], Data_per_C["[Exact] CI(L) %"][i],
                        color="blue", marker='_')
    Axes_obj_04.vlines(x=X[i],
                       ymin=Data_per_C["[Exact] CI(L) %"][i], ymax=Data_per_C["[Exact] CI(U) %"][i],
                       color="blue", linestyle='solid', linewidth=1.0)

###########################################################################################################
    for i in range(0, 13):
        Axes_obj_03.plot([LabelDataSet["Point_x"][i], LabelDataSet["Point_x"][i]], [8.5, 8.7],
                         color=LabelDataSet["LColor"][i])
        Axes_obj_03.plot([LabelDataSet["Label_Point_x"][i], LabelDataSet["Label_Point_x"][i]], [9.0, 9.2],
                         color=LabelDataSet["LColor"][i])
        Axes_obj_03.plot([LabelDataSet["Point_x"][i], LabelDataSet["Label_Point_x"][i]], [8.7, 9.0],
                         color=LabelDataSet["LColor"][i])
        Axes_obj_03.text(LabelDataSet["Label_Point_x"][i], 9.4,
                         LabelDataSet["Label"][i], color=LabelDataSet["LColor"][i],
                         fontsize=10, fontweight="normal", horizontalalignment="center", verticalalignment="bottom",
                         rotation=90.0,
                         bbox=dict(boxstyle='square', edgecolor='black', fc="white"))

Axes_obj_04.vlines(x=Point_X_B, ymin=-0.5, ymax=Point_Y_B, color="blue", linestyle='dashed', linewidth=1.0)

Axes_obj_04.annotate(
 text='\n'.join(["DVT", "12.1 h"]),
    xy=(Point_X_B, 5.0*(9.5/16)), xytext=(Point_X_B, 6.0*(9.5/16)), ha='center', va='center', size=10.0,
    bbox=dict(boxstyle='round', edgecolor='blue', fc='white'),
    arrowprops=dict(
        facecolor='blue', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0, edgecolor='blue', shrink=0.1)
)
########################################################################################################################
Figure_object.tight_layout()
my_path = os.path.abspath("Figures")
Figure_object.savefig(os.path.join(my_path, "01_Kimoto_et_al_(2023)_[Fig_S_03].png"))

from PIL import Image # py -m pip install pillow
img = Image.open(os.path.join(my_path, "01_Kimoto_et_al_(2023)_[Fig_S_03].png"))
img_resize = img.resize(size=(2866, 2016)) #size in pixels, as a 2-tuple: (width, height)
my_path2 = os.path.abspath("Figures/Fixed_Version/B6")
img_resize.save(os.path.join(my_path2, "01_Kimoto_et_al_(2023)_[Fig_S_03]_B6.png"))

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