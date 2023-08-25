########################################################################################################################
# Fig. S4. Re-analysis of the data in Table 1 reported by Clérel & Caillard.(01_Kimoto_et_al_(2023)_[Fig_S_04].py)
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
Line_01 = ["Timeline Table 1: Years 1992–1997 (Recommendations of Hormone Replacement Therapy, HRT)                   "]
Line_02 = ["----------------------------------------------------------------------------------------------------------"]
Line_03 = ["Year Month Type of Article  Journal (Citation)                     Recommendation                    "]
Line_04 = ["----------------------------------------------------------------------------------------------------------"]
Line_05 = ["1992  Dec  ACP Guideline    Ann Intern Med. 1992;117(12):1038-41.  Consider for coronary disease          "]
Line_06 = ["1995  Jul  AHA Guideline    J Am Coll Cardiol. 1995;26(1):292-4.   Mentioned risk reduction interventions "]
Line_07 = ["1997  May  AHA Guideline    Circulation. 1997;95(9):2329-31.       Consider HRT in postmenopausal women   "]
Line_08 = ["1997  Jun  NHS Cohort       N Engl J Med. 1997;336(25):1769-75.    Mortality: HRT user < non-user         "]
Line_09 = ["----------------------------------------------------------------------------------------------------------"]
Line_10 = ["                                                                                                          "]
Line_11 = ["Timeline Table 2: Years 1998–2002 (Expressing Concerns on the Safety of Hormone Replacement Therapy, HRT) "]
Line_12 = ["----------------------------------------------------------------------------------------------------------"]
Line_13 = [" Year  Month  Type of Article                                        Journal (Citation)                   "]
Line_14 = ["----------------------------------------------------------------------------------------------------------"]
Line_15 = [" 1998   Aug   HERS Randomized Clinical Trial                         AMA. 1998;280(7):605-13.             "]
Line_16 = [" 1999   May   AHA/ACC scientific statement                           J Am Coll Cardiol. 1999;33(6):1751-5."]
Line_17 = [" 2001   Jul   AHA Guideline                                              Circulation. 2001;104(4):499-503."]
Line_18 = [" 2002   Jul   HERS Randomized Clinical Trial Follow-up (HERS II)         AMA. 2002;288(1):58-66.          "]
Line_19 = [" 2002   Jul   Women's Health Initiative (WHI) Randomized Clinical Trial  JAMA. 2002;288(3):321-33.        "]
Line_20 = ["----------------------------------------------------------------------------------------------------------"]
Line_21 = ["                                                                                                          "]
Line_22 = ["Abbreviations:                                                                                            "]
Line_23 = ["ACP: American College of Physicians, AHA: American Heart Association, NHS: Nurses' Health Study           "]
Line_24 = ["HERS: Heart and Estrogen/progestin Replacement Study, ACC: American College of Cardiology                 "]

TextDataSet = pd.DataFrame([Line_01, Line_02, Line_03, Line_04, Line_05, Line_06, Line_07, Line_08, Line_09, Line_10,
                            Line_11, Line_12, Line_13, Line_14, Line_15, Line_16, Line_17, Line_18, Line_19, Line_20,
                            Line_21, Line_22, Line_23, Line_24],
                           index=["Line_01", "Line_02", "Line_03", "Line_04", "Line_05",
                                  "Line_06", "Line_07", "Line_08", "Line_09", "Line_10",
                                  "Line_11", "Line_12", "Line_13", "Line_14", "Line_15",
                                  "Line_16", "Line_17", "Line_18", "Line_19", "Line_20",
                                  "Line_21", "Line_22", "Line_23", "Line_24"],
                           columns=["Text"])

print(TextDataSet)

Cite_00 = ["Bull Acad Natl Med. 1999;183(5):985-97; discussion 997-1001."]
Cite_01 = ["Clérel M, Caillard G."]
Cite_02 = ["Syndrome thrombo-embolique de la station assise prolongée et vols de longue durée:"]
Cite_03 = ["l'expérience du Service Médical d'Urgence d'Aéroports De Paris"]
Cite_04 = ["[Thromboembolic syndrome from prolonged sitting and flights of long duration:"]
Cite_05 = ["experience of the Emergency Medical Service of the Paris Airports]."]
Cite_06 = ["Bull Acad Natl Med. 1999;183(5):985-97; discussion 997-1001. French. PMID: 10465002."]


Passengers_1990 = [1990, 23.0063694267516, 23.0, 23, 3]
Passengers_1991 = [1991, 22.2929936305732, 22.3, 22, 0]
Passengers_1992 = [1992, 24.8789808917197, 24.9, 24, 1]
Passengers_1993 = [1993, 25.5031847133758, 25.5, 25, 5]
Passengers_1994 = [1994, 27.3757961783439, 27.3, 27, 5]
Passengers_1995 = [1995, 27.1974522292994, 27.2, 26, 11]
Passengers_1996 = [1996, 29.3375796178344, 29.3, 29, 12]
Passengers_1997 = [1997, 29.9617834394904, 30.0, 30, 12]
Passengers_1998 = [1998, 31.7452229299363, 31.7, 32, 15]

DataSet_Passengers = pd.DataFrame([Passengers_1990, Passengers_1991, Passengers_1992, Passengers_1993,
                                   Passengers_1994, Passengers_1995, Passengers_1996, Passengers_1997,
                                   Passengers_1998],
                                  index=["Passengers_1990", "Passengers_1991", "Passengers_1992", "Passengers_1993",
                                         "Passengers_1994", "Passengers_1995", "Passengers_1996", "Passengers_1997",
                                         "Passengers_1998"],
                                  columns=["Year", "Passengers1", "Passengers2", "Passengers3", "Thoromboembolism"])
print(DataSet_Passengers)
print(DataSet_Passengers.sum(axis=0))

Patient_001 = [1, 55, "F", "5"]
Patient_002 = [2, 65, "F", "23"]
Patient_003 = [3, 58, "H", "12"]
Patient_004 = [4, 52, "F", "13"]
Patient_005 = [5, 65, "F", "8"]
Patient_006 = [6, 27, "F", "18"]
Patient_007 = [7, 45, "H", "14"]
Patient_008 = [8, 79, "F", "12"]
Patient_009 = [9, 47, "H", "8"]
Patient_010 = [10, 71, "H", "14"]
Patient_011 = [11, 37, "F", "15"]
Patient_012 = [12, 68, "F", "20"]
Patient_013 = [13, 66, "F", "12"]
Patient_014 = [14, 45, "F", "12"]
Patient_015 = [15, 26, "H", "3"]
Patient_016 = [16, 63, "F", "12"]
Patient_017 = [17, 69, "F", "15"]
Patient_018 = [18, 61, "F", "15"]
Patient_019 = [19, 57, "H", "14"]
Patient_020 = [20, 66, "H", "6"]
Patient_021 = [21, 44, "F", "12"]
Patient_022 = [22, 66, "H", "13"]
Patient_023 = [23, 56, "F", "20"]
Patient_024 = [24, 68, "H", "12"]
Patient_025 = [25, 48, "F", "12"]
Patient_026 = [26, 64, "F", "13"]
Patient_027 = [27, 60, "F", "16"]
Patient_028 = [28, 58, "F", "13"]
Patient_029 = [29, 50, "F", "13"]
Patient_030 = [30, 44, "H", "13"]
Patient_031 = [31, 71, "F", "12"]
Patient_032 = [32, 48, "F", "13"]
Patient_033 = [33, 85, "F", "18"]
Patient_034 = [34, 64, "F", "12"]
Patient_035 = [35, 67, "F", "12"]
Patient_036 = [36, 61, "F", "13"]
Patient_037 = [37, 60, "F", "8"]
Patient_038 = [38, 69, "F", "20"]
Patient_039 = [39, 85, "F", "12"]
Patient_040 = [40, 51, "H", "12"]
Patient_041 = [41, 65, "F", "17"]
Patient_042 = [42, 60, "F", "10"]
Patient_043 = [43, 73, "F", "12"]
Patient_044 = [44, 67, "F", "13"]
Patient_045 = [45, 57, "F", "15"]
Patient_046 = [46, 53, "F", "18"]
Patient_047 = [47, 72, "F", "11"]
Patient_048 = [48, 66, "H", "14"]
Patient_049 = [49, 75, "F", "23"]
Patient_050 = [50, 59, "F", "11"]
Patient_051 = [51, 66, "F", "8"]
Patient_052 = [52, 53, "F", "10"]
Patient_053 = [53, 35, "F", "14"]
Patient_054 = [54, 59, "F", "12"]
Patient_055 = [55, 29, "H", "3"]
Patient_056 = [56, 55, "F", "13"]
Patient_057 = [57, 66, "H", "7"]
Patient_058 = [58, 77, "H", "8"]
Patient_059 = [59, 49, "F", "14"]
Patient_060 = [60, 44, "F", "12"]
Patient_061 = [61, 47, "F", "-"]
Patient_062 = [62, 67, "F", "15"]
Patient_063 = [63, 55, "F", "12"]
Patient_064 = [64, 53, "F", "14"]

DataSet_Patients = pd.DataFrame([Patient_001, Patient_002, Patient_003, Patient_004, Patient_005,
                                 Patient_006, Patient_007, Patient_008, Patient_009, Patient_010,
                                 Patient_011, Patient_012, Patient_013, Patient_014, Patient_015,
                                 Patient_016, Patient_017, Patient_018, Patient_019, Patient_020,
                                 Patient_021, Patient_022, Patient_023, Patient_024, Patient_025,
                                 Patient_026, Patient_027, Patient_028, Patient_029, Patient_030,
                                 Patient_031, Patient_032, Patient_033, Patient_034, Patient_035,
                                 Patient_036, Patient_037, Patient_038, Patient_039, Patient_040,
                                 Patient_041, Patient_042, Patient_043, Patient_044, Patient_045,
                                 Patient_046, Patient_047, Patient_048, Patient_049, Patient_050,
                                 Patient_051, Patient_052, Patient_053, Patient_054, Patient_055,
                                 Patient_056, Patient_057, Patient_058, Patient_059, Patient_060,
                                 Patient_061, Patient_062, Patient_063, Patient_064],
                                index=["Patient_001", "Patient_002", "Patient_003", "Patient_004",
                                       "Patient_005", "Patient_006", "Patient_007", "Patient_008",
                                       "Patient_009", "Patient_010", "Patient_011", "Patient_012",
                                       "Patient_013", "Patient_014", "Patient_015", "Patient_016",
                                       "Patient_017", "Patient_018", "Patient_019", "Patient_020",
                                       "Patient_021", "Patient_022", "Patient_023", "Patient_024",
                                       "Patient_025", "Patient_026", "Patient_027", "Patient_028",
                                       "Patient_029", "Patient_030", "Patient_031", "Patient_032",
                                       "Patient_033", "Patient_034", "Patient_035", "Patient_036",
                                       "Patient_037", "Patient_038", "Patient_039", "Patient_040",
                                       "Patient_041", "Patient_042", "Patient_043", "Patient_044",
                                       "Patient_045", "Patient_046", "Patient_047", "Patient_048",
                                       "Patient_049", "Patient_050", "Patient_051", "Patient_052",
                                       "Patient_053", "Patient_054", "Patient_055", "Patient_056",
                                       "Patient_057", "Patient_058", "Patient_059", "Patient_060",
                                       "Patient_061", "Patient_062", "Patient_063", "Patient_064"],
                                columns=["No.", "Age", "Sex", "Duration"])

DataSet_Male = DataSet_Patients[DataSet_Patients["Sex"] == "H"]
DataSet_Female = DataSet_Patients[DataSet_Patients["Sex"] == "F"]

########################################################################################################################
print("")
print("###############################################################################################################")
print("########## Clerel & Caillard ##########")
print("---------------------------------------------------------------------------------------------------------------")
print(DataSet_Patients)
print("---------------------------------------------------------------------------------------------------------------")
print("Age, Lange: ", np.min(DataSet_Patients["Age"]), " - ", np.max(DataSet_Patients["Age"]),
      " n = ", len(DataSet_Patients["Age"]))
print("---------------------------------------------------------------------------------------------------------------")
print("Age, Mean: ", np.mean(DataSet_Patients["Age"]), ", SD: ", np.std(DataSet_Patients["Age"], ddof=1))
print("Age, Median: ", np.median(DataSet_Patients["Age"]))
print("---------------------------------------------------------------------------------------------------------------")
print("---------------------------------------------------------------------------------------------------------------")

print("")
print("###############################################################################################################")
print("##########  Clerel & Caillard  (Male) ##########")
print("---------------------------------------------------------------------------------------------------------------")
print(DataSet_Male)
print("---------------------------------------------------------------------------------------------------------------")
print("Age, Lange: ", np.min(DataSet_Male["Age"]), " - ", np.max(DataSet_Male["Age"]),
      " n = ", len(DataSet_Male["Age"]))
print("---------------------------------------------------------------------------------------------------------------")
print("Age, Mean: ", np.mean(DataSet_Male["Age"]), ", SD: ", np.std(DataSet_Male["Age"], ddof=1))
print("Age, Median: ", np.median(DataSet_Male["Age"]))
print("---------------------------------------------------------------------------------------------------------------")

print("")
print("###############################################################################################################")
print("##########  Clerel & Caillard  (Female) ##########")
print("---------------------------------------------------------------------------------------------------------------")
print(DataSet_Female)
print("---------------------------------------------------------------------------------------------------------------")
print("Age, Lange: ", np.min(DataSet_Female["Age"]), " - ", np.max(DataSet_Female["Age"]),
      " n = ", len(DataSet_Female["Age"]))
print("---------------------------------------------------------------------------------------------------------------")
print("Age, Mean: ", np.mean(DataSet_Female["Age"]), ", SD: ", np.std(DataSet_Female["Age"], ddof=1))
print("Age, Median: ", np.median(DataSet_Female["Age"]))
print("---------------------------------------------------------------------------------------------------------------")
########################################################################################################################
print("")
print("###############################################################################################################")

DataSet_Patients_2 = DataSet_Patients.drop("Patient_061", axis=0)
DataSet_Male_2 = DataSet_Male.copy()
DataSet_Female_2 = DataSet_Female.drop("Patient_061", axis=0)

print(DataSet_Male_2)

DataSet_Patients_2["Duration_int"] = DataSet_Patients_2["Duration"].astype(int)
DataSet_Male_2["Duration_int"] = DataSet_Male_2.loc[:, "Duration"].astype(int)
DataSet_Female_2["Duration_int"] = DataSet_Female_2["Duration"].astype(int)

print(DataSet_Patients_2)
print(DataSet_Male_2)
print(DataSet_Female_2)

Time = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

Patients_00hr = len(DataSet_Patients_2[DataSet_Patients_2["Duration_int"] == 0])
Patients_01hr = len(DataSet_Patients_2[DataSet_Patients_2["Duration_int"] == 1])
Patients_02hr = len(DataSet_Patients_2[DataSet_Patients_2["Duration_int"] == 2])
Patients_03hr = len(DataSet_Patients_2[DataSet_Patients_2["Duration_int"] == 3])
Patients_04hr = len(DataSet_Patients_2[DataSet_Patients_2["Duration_int"] == 4])
Patients_05hr = len(DataSet_Patients_2[DataSet_Patients_2["Duration_int"] == 5])
Patients_06hr = len(DataSet_Patients_2[DataSet_Patients_2["Duration_int"] == 6])
Patients_07hr = len(DataSet_Patients_2[DataSet_Patients_2["Duration_int"] == 7])
Patients_08hr = len(DataSet_Patients_2[DataSet_Patients_2["Duration_int"] == 8])
Patients_09hr = len(DataSet_Patients_2[DataSet_Patients_2["Duration_int"] == 9])
Patients_10hr = len(DataSet_Patients_2[DataSet_Patients_2["Duration_int"] == 10])
Patients_11hr = len(DataSet_Patients_2[DataSet_Patients_2["Duration_int"] == 11])
Patients_12hr = len(DataSet_Patients_2[DataSet_Patients_2["Duration_int"] == 12])
Patients_13hr = len(DataSet_Patients_2[DataSet_Patients_2["Duration_int"] == 13])
Patients_14hr = len(DataSet_Patients_2[DataSet_Patients_2["Duration_int"] == 14])
Patients_15hr = len(DataSet_Patients_2[DataSet_Patients_2["Duration_int"] == 15])
Patients_16hr = len(DataSet_Patients_2[DataSet_Patients_2["Duration_int"] == 16])
Patients_17hr = len(DataSet_Patients_2[DataSet_Patients_2["Duration_int"] == 17])
Patients_18hr = len(DataSet_Patients_2[DataSet_Patients_2["Duration_int"] == 18])
Patients_19hr = len(DataSet_Patients_2[DataSet_Patients_2["Duration_int"] == 19])
Patients_20hr = len(DataSet_Patients_2[DataSet_Patients_2["Duration_int"] == 20])
Patients_21hr = len(DataSet_Patients_2[DataSet_Patients_2["Duration_int"] == 21])
Patients_22hr = len(DataSet_Patients_2[DataSet_Patients_2["Duration_int"] == 22])
Patients_23hr = len(DataSet_Patients_2[DataSet_Patients_2["Duration_int"] == 23])
Patients_24hr = len(DataSet_Patients_2[DataSet_Patients_2["Duration_int"] == 24])

PatientsFreq = [Patients_00hr, Patients_01hr, Patients_02hr, Patients_03hr, Patients_04hr, Patients_05hr,
                Patients_06hr, Patients_07hr, Patients_08hr, Patients_09hr, Patients_10hr, Patients_11hr,
                Patients_12hr, Patients_13hr, Patients_14hr, Patients_15hr, Patients_16hr, Patients_17hr,
                Patients_18hr, Patients_19hr, Patients_20hr, Patients_21hr, Patients_22hr, Patients_23hr,
                Patients_24hr]

Male_00hr = len(DataSet_Male_2[DataSet_Male_2["Duration_int"] == 0])
Male_01hr = len(DataSet_Male_2[DataSet_Male_2["Duration_int"] == 1])
Male_02hr = len(DataSet_Male_2[DataSet_Male_2["Duration_int"] == 2])
Male_03hr = len(DataSet_Male_2[DataSet_Male_2["Duration_int"] == 3])
Male_04hr = len(DataSet_Male_2[DataSet_Male_2["Duration_int"] == 4])
Male_05hr = len(DataSet_Male_2[DataSet_Male_2["Duration_int"] == 5])
Male_06hr = len(DataSet_Male_2[DataSet_Male_2["Duration_int"] == 6])
Male_07hr = len(DataSet_Male_2[DataSet_Male_2["Duration_int"] == 7])
Male_08hr = len(DataSet_Male_2[DataSet_Male_2["Duration_int"] == 8])
Male_09hr = len(DataSet_Male_2[DataSet_Male_2["Duration_int"] == 9])
Male_10hr = len(DataSet_Male_2[DataSet_Male_2["Duration_int"] == 10])
Male_11hr = len(DataSet_Male_2[DataSet_Male_2["Duration_int"] == 11])
Male_12hr = len(DataSet_Male_2[DataSet_Male_2["Duration_int"] == 12])
Male_13hr = len(DataSet_Male_2[DataSet_Male_2["Duration_int"] == 13])
Male_14hr = len(DataSet_Male_2[DataSet_Male_2["Duration_int"] == 14])
Male_15hr = len(DataSet_Male_2[DataSet_Male_2["Duration_int"] == 15])
Male_16hr = len(DataSet_Male_2[DataSet_Male_2["Duration_int"] == 16])
Male_17hr = len(DataSet_Male_2[DataSet_Male_2["Duration_int"] == 17])
Male_18hr = len(DataSet_Male_2[DataSet_Male_2["Duration_int"] == 18])
Male_19hr = len(DataSet_Male_2[DataSet_Male_2["Duration_int"] == 19])
Male_20hr = len(DataSet_Male_2[DataSet_Male_2["Duration_int"] == 20])
Male_21hr = len(DataSet_Male_2[DataSet_Male_2["Duration_int"] == 21])
Male_22hr = len(DataSet_Male_2[DataSet_Male_2["Duration_int"] == 22])
Male_23hr = len(DataSet_Male_2[DataSet_Male_2["Duration_int"] == 23])
Male_24hr = len(DataSet_Male_2[DataSet_Male_2["Duration_int"] == 24])

MaleFreq = [Male_00hr, Male_01hr, Male_02hr, Male_03hr, Male_04hr, Male_05hr,
            Male_06hr, Male_07hr, Male_08hr, Male_09hr, Male_10hr, Male_11hr,
            Male_12hr, Male_13hr, Male_14hr, Male_15hr, Male_16hr, Male_17hr,
            Male_18hr, Male_19hr, Male_20hr, Male_21hr, Male_22hr, Male_23hr, Male_24hr]

Female_00hr = len(DataSet_Female_2[DataSet_Female_2["Duration_int"] == 0])
Female_01hr = len(DataSet_Female_2[DataSet_Female_2["Duration_int"] == 1])
Female_02hr = len(DataSet_Female_2[DataSet_Female_2["Duration_int"] == 2])
Female_03hr = len(DataSet_Female_2[DataSet_Female_2["Duration_int"] == 3])
Female_04hr = len(DataSet_Female_2[DataSet_Female_2["Duration_int"] == 4])
Female_05hr = len(DataSet_Female_2[DataSet_Female_2["Duration_int"] == 5])
Female_06hr = len(DataSet_Female_2[DataSet_Female_2["Duration_int"] == 6])
Female_07hr = len(DataSet_Female_2[DataSet_Female_2["Duration_int"] == 7])
Female_08hr = len(DataSet_Female_2[DataSet_Female_2["Duration_int"] == 8])
Female_09hr = len(DataSet_Female_2[DataSet_Female_2["Duration_int"] == 9])
Female_10hr = len(DataSet_Female_2[DataSet_Female_2["Duration_int"] == 10])
Female_11hr = len(DataSet_Female_2[DataSet_Female_2["Duration_int"] == 11])
Female_12hr = len(DataSet_Female_2[DataSet_Female_2["Duration_int"] == 12])
Female_13hr = len(DataSet_Female_2[DataSet_Female_2["Duration_int"] == 13])
Female_14hr = len(DataSet_Female_2[DataSet_Female_2["Duration_int"] == 14])
Female_15hr = len(DataSet_Female_2[DataSet_Female_2["Duration_int"] == 15])
Female_16hr = len(DataSet_Female_2[DataSet_Female_2["Duration_int"] == 16])
Female_17hr = len(DataSet_Female_2[DataSet_Female_2["Duration_int"] == 17])
Female_18hr = len(DataSet_Female_2[DataSet_Female_2["Duration_int"] == 18])
Female_19hr = len(DataSet_Female_2[DataSet_Female_2["Duration_int"] == 19])
Female_20hr = len(DataSet_Female_2[DataSet_Female_2["Duration_int"] == 20])
Female_21hr = len(DataSet_Female_2[DataSet_Female_2["Duration_int"] == 21])
Female_22hr = len(DataSet_Female_2[DataSet_Female_2["Duration_int"] == 22])
Female_23hr = len(DataSet_Female_2[DataSet_Female_2["Duration_int"] == 23])
Female_24hr = len(DataSet_Female_2[DataSet_Female_2["Duration_int"] == 24])

FemaleFreq = [Female_00hr, Female_01hr, Female_02hr, Female_03hr, Female_04hr, Female_05hr,
              Female_06hr, Female_07hr, Female_08hr, Female_09hr, Female_10hr, Female_11hr,
              Female_12hr, Female_13hr, Female_14hr, Female_15hr, Female_16hr, Female_17hr,
              Female_18hr, Female_19hr, Female_20hr, Female_21hr, Female_22hr, Female_23hr,
              Female_24hr]

DataSet_Freq = pd.DataFrame({"Time": Time, "Male": MaleFreq, "Female": FemaleFreq, "Total": PatientsFreq},
                            index=["00hr", "01hr", "02hr", "03hr", "04hr", "05hr",
                                   "06hr", "07hr", "08hr", "09hr", "10hr", "11hr",
                                   "12hr", "13hr", "14hr", "15hr", "16hr", "17hr",
                                   "18hr", "19hr", "20hr", "21hr", "22hr", "23hr","24hr"])
print(DataSet_Freq)
print(DataSet_Freq.sum(axis=0))

########################################################################################################################
print("")
print("###############################################################################################################")

Time_Table = ["0 - 3", "3 - 6", "6 - 9", "9 - 12", "12 - 15", "15 - 18", "18 - 21", "21 - 24"]

Patients_00_03_hr = len(DataSet_Patients_2[(DataSet_Patients_2["Duration_int"] >=0) & (DataSet_Patients_2["Duration_int"] < 3)])
Patients_03_06_hr = len(DataSet_Patients_2[(DataSet_Patients_2["Duration_int"] >=3) & (DataSet_Patients_2["Duration_int"] < 6)])
Patients_06_09_hr = len(DataSet_Patients_2[(DataSet_Patients_2["Duration_int"] >=6) & (DataSet_Patients_2["Duration_int"] < 9)])
Patients_09_12_hr = len(DataSet_Patients_2[(DataSet_Patients_2["Duration_int"] >=9) & (DataSet_Patients_2["Duration_int"] < 12)])
Patients_12_15_hr = len(DataSet_Patients_2[(DataSet_Patients_2["Duration_int"] >=12) & (DataSet_Patients_2["Duration_int"] < 15)])
Patients_15_18_hr = len(DataSet_Patients_2[(DataSet_Patients_2["Duration_int"] >=15) & (DataSet_Patients_2["Duration_int"] < 18)])
Patients_18_21_hr = len(DataSet_Patients_2[(DataSet_Patients_2["Duration_int"] >=18) & (DataSet_Patients_2["Duration_int"] < 21)])
Patients_21_24_hr = len(DataSet_Patients_2[(DataSet_Patients_2["Duration_int"] >=21) & (DataSet_Patients_2["Duration_int"] < 24)])

PatientsFreq_Table = [Patients_00_03_hr, Patients_03_06_hr, Patients_06_09_hr, Patients_09_12_hr, Patients_12_15_hr, Patients_15_18_hr, Patients_18_21_hr, Patients_21_24_hr]

Male_00_03_hr = len(DataSet_Male_2[(DataSet_Male_2["Duration_int"] >= 0) & (DataSet_Male_2["Duration_int"] < 3)])
Male_03_06_hr = len(DataSet_Male_2[(DataSet_Male_2["Duration_int"] >= 3) & (DataSet_Male_2["Duration_int"] < 6)])
Male_06_09_hr = len(DataSet_Male_2[(DataSet_Male_2["Duration_int"] >= 6) & (DataSet_Male_2["Duration_int"] < 9)])
Male_09_12_hr = len(DataSet_Male_2[(DataSet_Male_2["Duration_int"] >= 9) & (DataSet_Male_2["Duration_int"] < 12)])
Male_12_15_hr = len(DataSet_Male_2[(DataSet_Male_2["Duration_int"] >= 12) & (DataSet_Male_2["Duration_int"] < 15)])
Male_15_18_hr = len(DataSet_Male_2[(DataSet_Male_2["Duration_int"] >= 15) & (DataSet_Male_2["Duration_int"] < 18)])
Male_18_21_hr = len(DataSet_Male_2[(DataSet_Male_2["Duration_int"] >= 18) & (DataSet_Male_2["Duration_int"] < 21)])
Male_21_24_hr = len(DataSet_Male_2[(DataSet_Male_2["Duration_int"] >= 21) & (DataSet_Male_2["Duration_int"] < 24)])

MaleFreq_Table = [Male_00_03_hr, Male_03_06_hr, Male_06_09_hr, Male_09_12_hr, Male_12_15_hr, Male_15_18_hr, Male_18_21_hr, Male_21_24_hr]

Female_00_03_hr = len(DataSet_Female_2[(DataSet_Female_2["Duration_int"] >= 0) & (DataSet_Female_2["Duration_int"] < 3)])
Female_03_06_hr = len(DataSet_Female_2[(DataSet_Female_2["Duration_int"] >= 3) & (DataSet_Female_2["Duration_int"] < 6)])
Female_06_09_hr = len(DataSet_Female_2[(DataSet_Female_2["Duration_int"] >= 6) & (DataSet_Female_2["Duration_int"] < 9)])
Female_09_12_hr = len(DataSet_Female_2[(DataSet_Female_2["Duration_int"] >= 9) & (DataSet_Female_2["Duration_int"] < 12)])
Female_12_15_hr = len(DataSet_Female_2[(DataSet_Female_2["Duration_int"] >= 12) & (DataSet_Female_2["Duration_int"] < 15)])
Female_15_18_hr = len(DataSet_Female_2[(DataSet_Female_2["Duration_int"] >= 15) & (DataSet_Female_2["Duration_int"] < 18)])
Female_18_21_hr = len(DataSet_Female_2[(DataSet_Female_2["Duration_int"] >= 18) & (DataSet_Female_2["Duration_int"] < 21)])
Female_21_24_hr = len(DataSet_Female_2[(DataSet_Female_2["Duration_int"] >= 21) & (DataSet_Female_2["Duration_int"] < 24)])

FemaleFreq_Table = [Female_00_03_hr, Female_03_06_hr, Female_06_09_hr, Female_09_12_hr, Female_12_15_hr, Female_15_18_hr, Female_18_21_hr, Female_21_24_hr]

DataSet_Freq_Table = pd.DataFrame({"Time": Time_Table, "Male": MaleFreq_Table, "Female": FemaleFreq_Table, "Total": PatientsFreq_Table},
                            index=["0h - 3h", "3h - 6h", "6h - 9h", "9h - 12h", "12h - 15h", "15h - 18h", "18h - 21r", "21h - 24h"])
print(DataSet_Freq_Table)
print(DataSet_Freq_Table.sum(axis=0))

########################################################################################################################
print("")
print("###############################################################################################################")
Total = [0, 0, 0, 2, 0, 1, 1, 1, 5, 0, 2, 2, 17, 10, 7, 5, 1, 1, 3, 0, 3, 0, 0, 2, 0, 0]

data_00_First = []
data_01_First = [99]
data_02_First = [99, 99]
data_03_First = [99, 99, 99]
data_04_First = [99, 99, 99, 99]
data_05_First = [99, 99, 99, 99, 99]
data_06_First = [99, 99, 99, 99, 99, 99]
data_07_First = [99, 99, 99, 99, 99, 99, 99]
data_08_First = [99, 99, 99, 99, 99, 99, 99, 99]
data_09_First = [99, 99, 99, 99, 99, 99, 99, 99, 99]
data_10_First = [99, 99, 99, 99, 99, 99, 99, 99, 99, 99]
data_11_First = [99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99]
data_12_First = [99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99]
data_13_First = [99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99]
data_14_First = [99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99]
data_15_First = [99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99]
data_16_First = [99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99]
data_17_First = [99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99]
data_18_First = [99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99]
data_19_First = [99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99]
data_20_First = [99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99]
data_21_First = [99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99]
data_22_First = [99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99]

data_00_Last = [99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99]
data_01_Last = [99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99]
data_02_Last = [99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99]
data_03_Last = [99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99]
data_04_Last = [99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99]
data_05_Last = [99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99]
data_06_Last = [99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99]
data_07_Last = [99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99]
data_08_Last = [99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99]
data_09_Last = [99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99]
data_10_Last = [99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99]
data_11_Last = [99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99]
data_12_Last = [99, 99, 99, 99, 99, 99, 99, 99, 99, 99]
data_13_Last = [99, 99, 99, 99, 99, 99, 99, 99, 99]
data_14_Last = [99, 99, 99, 99, 99, 99, 99, 99]
data_15_Last = [99, 99, 99, 99, 99, 99, 99]
data_16_Last = [99, 99, 99, 99, 99, 99]
data_17_Last = [99, 99, 99, 99, 99]
data_18_Last = [99, 99, 99, 99]
data_19_Last = [99, 99, 99]
data_20_Last = [99, 99]
data_21_Last = [99]
data_22_Last = []

data_00 = data_00_First + Total + data_00_Last
data_01 = data_01_First + Total + data_01_Last
data_02 = data_02_First + Total + data_02_Last
data_03 = data_03_First + Total + data_03_Last
data_04 = data_04_First + Total + data_04_Last
data_05 = data_05_First + Total + data_05_Last
data_06 = data_06_First + Total + data_06_Last
data_07 = data_07_First + Total + data_07_Last
data_08 = data_08_First + Total + data_08_Last
data_09 = data_09_First + Total + data_09_Last
data_10 = data_10_First + Total + data_10_Last
data_11 = data_11_First + Total + data_11_Last
data_12 = data_12_First + Total + data_12_Last
data_13 = data_13_First + Total + data_13_Last
data_14 = data_14_First + Total + data_14_Last
data_15 = data_15_First + Total + data_15_Last
data_16 = data_16_First + Total + data_16_Last
data_17 = data_17_First + Total + data_17_Last
data_18 = data_18_First + Total + data_18_Last
data_19 = data_19_First + Total + data_19_Last
data_20 = data_20_First + Total + data_20_Last
data_21 = data_21_First + Total + data_21_Last
data_22 = data_22_First + Total + data_22_Last

DataSet_Peri = pd.DataFrame(
    {"data_00": data_00, "data_01": data_01, "data_02": data_02, "data_03": data_03,"data_04": data_04,
     "data_05": data_05, "data_06": data_06, "data_07": data_07, "data_08": data_08, "data_09": data_09,
     "data_10": data_10, "data_11": data_11, "data_12": data_12, "data_13": data_13, "data_14": data_14,
     "data_15": data_15, "data_16": data_16, "data_17": data_17, "data_18": data_18, "data_19": data_19,
     "data_20": data_20, "data_21": data_21, "data_22": data_22})

DataSet_Peri.replace([99], np.nan, inplace=True)
print(DataSet_Peri)

data_00_data_01 = DataSet_Peri.loc[1:25, ["data_00", "data_01"]]
data_00_data_02 = DataSet_Peri.loc[2:25, ["data_00", "data_02"]]
data_00_data_03 = DataSet_Peri.loc[3:25, ["data_00", "data_03"]]
data_00_data_04 = DataSet_Peri.loc[4:25, ["data_00", "data_04"]]
data_00_data_05 = DataSet_Peri.loc[5:25, ["data_00", "data_05"]]
data_00_data_06 = DataSet_Peri.loc[6:25, ["data_00", "data_06"]]
data_00_data_07 = DataSet_Peri.loc[7:25, ["data_00", "data_07"]]
data_00_data_08 = DataSet_Peri.loc[8:25, ["data_00", "data_08"]]
data_00_data_09 = DataSet_Peri.loc[9:25, ["data_00", "data_09"]]
data_00_data_10 = DataSet_Peri.loc[10:25, ["data_00", "data_10"]]
data_00_data_11 = DataSet_Peri.loc[11:25, ["data_00", "data_11"]]
data_00_data_12 = DataSet_Peri.loc[12:25, ["data_00", "data_12"]]
data_00_data_13 = DataSet_Peri.loc[13:25, ["data_00", "data_13"]]
data_00_data_14 = DataSet_Peri.loc[14:25, ["data_00", "data_14"]]
data_00_data_15 = DataSet_Peri.loc[15:25, ["data_00", "data_15"]]
data_00_data_16 = DataSet_Peri.loc[16:25, ["data_00", "data_16"]]
data_00_data_17 = DataSet_Peri.loc[17:25, ["data_00", "data_17"]]
data_00_data_18 = DataSet_Peri.loc[18:25, ["data_00", "data_18"]]
data_00_data_19 = DataSet_Peri.loc[19:25, ["data_00", "data_19"]]
data_00_data_20 = DataSet_Peri.loc[20:25, ["data_00", "data_20"]]
data_00_data_21 = DataSet_Peri.loc[21:25, ["data_00", "data_21"]]
data_00_data_22 = DataSet_Peri.loc[22:25, ["data_00", "data_22"]]

R_01_p = data_00_data_01.corr(method="pearson").iloc[0, 1]
R_02_p = data_00_data_02.corr(method="pearson").iloc[0, 1]
R_03_p = data_00_data_03.corr(method="pearson").iloc[0, 1]
R_04_p = data_00_data_04.corr(method="pearson").iloc[0, 1]
R_05_p = data_00_data_05.corr(method="pearson").iloc[0, 1]
R_06_p = data_00_data_06.corr(method="pearson").iloc[0, 1]
R_07_p = data_00_data_07.corr(method="pearson").iloc[0, 1]
R_08_p = data_00_data_08.corr(method="pearson").iloc[0, 1]
R_09_p = data_00_data_09.corr(method="pearson").iloc[0, 1]
R_10_p = data_00_data_10.corr(method="pearson").iloc[0, 1]
R_11_p = data_00_data_11.corr(method="pearson").iloc[0, 1]
R_12_p = data_00_data_12.corr(method="pearson").iloc[0, 1]
R_13_p = data_00_data_13.corr(method="pearson").iloc[0, 1]
R_14_p = data_00_data_14.corr(method="pearson").iloc[0, 1]
R_15_p = data_00_data_15.corr(method="pearson").iloc[0, 1]
R_16_p = data_00_data_16.corr(method="pearson").iloc[0, 1]
R_17_p = data_00_data_17.corr(method="pearson").iloc[0, 1]
R_18_p = data_00_data_18.corr(method="pearson").iloc[0, 1]
R_19_p = data_00_data_19.corr(method="pearson").iloc[0, 1]
R_20_p = data_00_data_20.corr(method="pearson").iloc[0, 1]
R_21_p = data_00_data_21.corr(method="pearson").iloc[0, 1]
R_22_p = data_00_data_22.corr(method="pearson").iloc[0, 1]

R_01_s = data_00_data_01.corr(method="spearman").iloc[0, 1]
R_02_s = data_00_data_02.corr(method="spearman").iloc[0, 1]
R_03_s = data_00_data_03.corr(method="spearman").iloc[0, 1]
R_04_s = data_00_data_04.corr(method="spearman").iloc[0, 1]
R_05_s = data_00_data_05.corr(method="spearman").iloc[0, 1]
R_06_s = data_00_data_06.corr(method="spearman").iloc[0, 1]
R_07_s = data_00_data_07.corr(method="spearman").iloc[0, 1]
R_08_s = data_00_data_08.corr(method="spearman").iloc[0, 1]
R_09_s = data_00_data_09.corr(method="spearman").iloc[0, 1]
R_10_s = data_00_data_10.corr(method="spearman").iloc[0, 1]
R_11_s = data_00_data_11.corr(method="spearman").iloc[0, 1]
R_12_s = data_00_data_12.corr(method="spearman").iloc[0, 1]
R_13_s = data_00_data_13.corr(method="spearman").iloc[0, 1]
R_14_s = data_00_data_14.corr(method="spearman").iloc[0, 1]
R_15_s = data_00_data_15.corr(method="spearman").iloc[0, 1]
R_16_s = data_00_data_16.corr(method="spearman").iloc[0, 1]
R_17_s = data_00_data_17.corr(method="spearman").iloc[0, 1]
R_18_s = data_00_data_18.corr(method="spearman").iloc[0, 1]
R_19_s = data_00_data_19.corr(method="spearman").iloc[0, 1]
R_20_s = data_00_data_20.corr(method="spearman").iloc[0, 1]
R_21_s = data_00_data_21.corr(method="spearman").iloc[0, 1]
R_22_s = data_00_data_22.corr(method="spearman").iloc[0, 1]

R_01_k = data_00_data_01.corr(method="kendall").iloc[0, 1]
R_02_k = data_00_data_02.corr(method="kendall").iloc[0, 1]
R_03_k = data_00_data_03.corr(method="kendall").iloc[0, 1]
R_04_k = data_00_data_04.corr(method="kendall").iloc[0, 1]
R_05_k = data_00_data_05.corr(method="kendall").iloc[0, 1]
R_06_k = data_00_data_06.corr(method="kendall").iloc[0, 1]
R_07_k = data_00_data_07.corr(method="kendall").iloc[0, 1]
R_08_k = data_00_data_08.corr(method="kendall").iloc[0, 1]
R_09_k = data_00_data_09.corr(method="kendall").iloc[0, 1]
R_10_k = data_00_data_10.corr(method="kendall").iloc[0, 1]
R_11_k = data_00_data_11.corr(method="kendall").iloc[0, 1]
R_12_k = data_00_data_12.corr(method="kendall").iloc[0, 1]
R_13_k = data_00_data_13.corr(method="kendall").iloc[0, 1]
R_14_k = data_00_data_14.corr(method="kendall").iloc[0, 1]
R_15_k = data_00_data_15.corr(method="kendall").iloc[0, 1]
R_16_k = data_00_data_16.corr(method="kendall").iloc[0, 1]
R_17_k = data_00_data_17.corr(method="kendall").iloc[0, 1]
R_18_k = data_00_data_18.corr(method="kendall").iloc[0, 1]
R_19_k = data_00_data_19.corr(method="kendall").iloc[0, 1]
R_20_k = data_00_data_20.corr(method="kendall").iloc[0, 1]
R_21_k = data_00_data_21.corr(method="kendall").iloc[0, 1]
R_22_k = data_00_data_22.corr(method="kendall").iloc[0, 1]

R_coef_p = [R_01_p, R_02_p, R_03_p, R_04_p, R_05_p, R_06_p, R_07_p, R_08_p, R_09_p, R_10_p, R_11_p, R_12_p, R_13_p, R_14_p, R_15_p, R_16_p, R_17_p, R_18_p, R_19_p, R_20_p, R_21_p, R_22_p]
R_coef_s = [R_01_s, R_02_s, R_03_s, R_04_s, R_05_s, R_06_s, R_07_s, R_08_s, R_09_s, R_10_s, R_11_s, R_12_s, R_13_s, R_14_s, R_15_s, R_16_s, R_17_s, R_18_s, R_19_s, R_20_s, R_21_s, R_22_s]
R_coef_k = [R_01_k, R_02_k, R_03_k, R_04_k, R_05_k, R_06_k, R_07_k, R_08_k, R_09_k, R_10_k, R_11_k, R_12_k, R_13_k, R_14_k, R_15_k, R_16_k, R_17_k, R_18_k, R_19_k, R_20_k, R_21_k, R_22_k]
X_value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

def reg_func_01(parameter: ["n1", "a1", "b1", "n2", "a2", "b2"], x, y):
    n1 = parameter[0]
    a1 = parameter[1]
    b1 = parameter[2]
    n2 = parameter[3]
    a2 = parameter[4]
    b2 = parameter[5]

    _Output_ = y - ( n1 * (np.sin(a1 * (x - b1))) + n2 * (np.sin(a2 * (x - b2))))
    return _Output_
x = X_value; y = R_coef_s
#parameter_0 = [-1, 0.01, -10, 0.28, 0.35, -2.75]
parameter_0 = [1.5, 0.015, 1.0, -0.3, 0.015, 6.5]
RegResult_Original_W_01 = optimize.leastsq(reg_func_01, parameter_0, args=(x, y), full_output=True)
Result_W_01_0 = RegResult_Original_W_01[0]
Result_W_01 = [Result_W_01_0[0], Result_W_01_0[1], Result_W_01_0[2], Result_W_01_0[3], Result_W_01_0[4], Result_W_01_0[5]]
print(Result_W_01)

def reg_func_02(parameter: ["n1", "a1", "b1"], x, y):
    n1 = parameter[0]
    a1 = parameter[1]
    b1 = parameter[2]

    _Output_ = y - ( n1 * (np.sin(a1 * (x - b1))) )
    return _Output_
x = X_value; y = R_coef_s
parameter_0 = [0.45, -0.18, 5.00]
RegResult_Original_W_02 = optimize.leastsq(reg_func_02, parameter_0, args=(x, y), full_output=True)
Result_W_02_0 = RegResult_Original_W_02[0]
Result_W_02 = [Result_W_02_0[0], Result_W_02_0[1], Result_W_02_0[2]]
print(Result_W_02)

########################################################################################################################
# Values for Curves (Week)
########################################################################################################################
n1 = Result_W_01[0]
a1 = Result_W_01[1]
b1 = Result_W_01[2]
n2 = Result_W_01[3]
a2 = Result_W_01[4]
b2 = Result_W_01[5]

Time = np.linspace(0.0, 24, 1000)
Y_value = n1 * (np.sin(a1 * (Time - b1))) + n2 * (np.sin(a2 * (Time - b2)))

Period_a1 = 2*np.pi/a1
Period_round_a1 = Decimal(str(Period_a1)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)

Period_a2 = 2*np.pi/a2
Period_round_a2 = Decimal(str(Period_a2)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)

#Y = 8.18450943975949 * (sin(-0.003135287843114744 * (x - 1.8425813142702718))) -0.2836001722174021 * (sin(0.3424656810464765 * (x - 6.550891211403165)))
Maxima_solve = [1.183408138729218, 11.91797692381453, 19.53154120857992]
diff_Maxima_solve = [10.73456878508531, 7.61356428476539]
Average = 9.17406653492535
inflection_point_x = [6.550786954000879, 15.72465101405573, 24.89728609382136]

width_01 = Decimal(str(Maxima_solve[1] - Maxima_solve[0])).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
width_02 = Decimal(str(Maxima_solve[2] - Maxima_solve[1])).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
width_T = Decimal(str(Maxima_solve[2] - Maxima_solve[0])).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

width_01_h = ''.join([str(width_01), ' h'])
width_02_h = ''.join([str(width_02), ' h'])
width_T_h = ''.join([str(width_T), ' h'])

########################################################################################################################
# Figures
########################################################################################################################
Figure_object = plt.figure(1, figsize=(11.69, 8.27), dpi=400, edgecolor="black", linewidth=0.5)

plt.figtext(0.0200, 0.970, "a", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")
plt.figtext(0.5150, 0.970, "b", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")
plt.figtext(0.0200, 0.500, "c", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")
plt.figtext(0.5150, 0.500, "d", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")

gs_master = matplotlib.gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[1, 1])
gs_1 = matplotlib.gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=2, subplot_spec=gs_master[0])
gs_2 = matplotlib.gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=2, subplot_spec=gs_master[1])

Axes_obj_01 = Figure_object.add_subplot(gs_1[0])
Axes_obj_02 = Figure_object.add_subplot(gs_1[1])
Axes_obj_03 = Figure_object.add_subplot(gs_2[0])
Axes_obj_04 = Figure_object.add_subplot(gs_2[1])

########################################################################################################################
############################################################################################################
# (a)
############################################################################################################
Axes_obj_01.set_title("Clérel & Caillard 1999, Fig. 3", size=11.4230, fontweight="normal")
Axes_obj_01.set_xlabel('Duration, hr', fontweight="normal")
Axes_obj_01.set_ylabel('Frequency (Number of Patients)', fontweight="normal")
Axes_obj_01.tick_params(length=0.0)
Axes_obj_01.set_xticks([])
Axes_obj_01.set_yticks([])
#Axes_obj_01.xaxis.set_minor_locator(ticker.MultipleLocator(10))
#Axes_obj_01.yaxis.set_minor_locator(ticker.MultipleLocator(10))
Axes_obj_01.imshow(plt.imread("Clerel_Fig_03.jpg"))

############################################################################################################
# (b)
############################################################################################################
Axes_obj_02.set_title("Created from the raw data reported by Clérel & Caillard", size=11.4230, fontweight="normal")
Axes_obj_02.yaxis.set_visible(False)
Axes_obj_02.xaxis.set_visible(False)
Axes_obj_02.spines['right'].set_visible(False)
Axes_obj_02.spines['left'].set_visible(False)
Axes_obj_02.spines['top'].set_visible(False)
Axes_obj_02.spines['bottom'].set_visible(False)

#Axes_obj_02.set_xlabel('XXXX', fontweight="normal")
#Axes_obj_02.set_ylabel("XXXX")
Axes_obj_02.tick_params(length=0.0)
Axes_obj_02.set_xticks([])
Axes_obj_02.set_yticks([])
#Axes_obj_02.xaxis.set_minor_locator(ticker.MultipleLocator(5))
#Axes_obj_02.yaxis.set_minor_locator(ticker.MultipleLocator(5))

Axes_obj_02.set_xlim([0.0-2.5, 100+2.5])
Axes_obj_02.set_ylim([0.0-2.5, 100+2.5])

Axes_obj_02.hlines(y=0 + (100/9)*9, xmin=0.00, xmax=100, color="black", linestyle="solid", linewidth=1.0, zorder=9)
Axes_obj_02.hlines(y=0 + (100/9)*8, xmin=0.00, xmax=100, color="black", linestyle="solid", linewidth=1.0, zorder=9)
#Axes_obj_02.hlines(y=0 + (100/9)*7, xmin=0.00, xmax=100, color="black", linestyle="solid", linewidth=1.0, zorder=9)
#Axes_obj_02.hlines(y=0 + (100/9)*6, xmin=0.00, xmax=100, color="black", linestyle="solid", linewidth=1.0, zorder=9)
#Axes_obj_02.hlines(y=0 + (100/9)*5, xmin=0.00, xmax=100, color="black", linestyle="solid", linewidth=1.0, zorder=9)
#Axes_obj_02.hlines(y=0 + (100/9)*4, xmin=0.00, xmax=100, color="black", linestyle="solid", linewidth=1.0, zorder=9)
#Axes_obj_02.hlines(y=0 + (100/9)*3, xmin=0.00, xmax=100, color="black", linestyle="solid", linewidth=1.0, zorder=9)
#Axes_obj_02.hlines(y=0 + (100/9)*2, xmin=0.00, xmax=100, color="black", linestyle="solid", linewidth=1.0, zorder=9)
#Axes_obj_02.hlines(y=0 + (100/9)*1, xmin=0.00, xmax=100, color="black", linestyle="solid", linewidth=1.0, zorder=9)
Axes_obj_02.hlines(y=0 + (100/9)*0, xmin=0.00, xmax=100, color="black", linestyle="solid", linewidth=1.0, zorder=9)

#Axes_obj_02.hlines(y=0 + (100/9)*8 + (100/18), xmin=0.00, xmax=100, color="black", linestyle="dashed", linewidth=0.5, zorder=9)
#Axes_obj_02.hlines(y=0 + (100/9)*7 + (100/18), xmin=0.00, xmax=100, color="black", linestyle="dashed", linewidth=0.5, zorder=9)
#Axes_obj_02.hlines(y=0 + (100/9)*6 + (100/18), xmin=0.00, xmax=100, color="black", linestyle="dashed", linewidth=0.5, zorder=9)
#Axes_obj_02.hlines(y=0 + (100/9)*5 + (100/18), xmin=0.00, xmax=100, color="black", linestyle="dashed", linewidth=0.5, zorder=9)
#Axes_obj_02.hlines(y=0 + (100/9)*4 + (100/18), xmin=0.00, xmax=100, color="black", linestyle="dashed", linewidth=0.5, zorder=9)
#Axes_obj_02.hlines(y=0 + (100/9)*3 + (100/18), xmin=0.00, xmax=100, color="black", linestyle="dashed", linewidth=0.5, zorder=9)
#Axes_obj_02.hlines(y=0 + (100/9)*2 + (100/18), xmin=0.00, xmax=100, color="black", linestyle="dashed", linewidth=0.5, zorder=9)
#Axes_obj_02.hlines(y=0 + (100/9)*1 + (100/18), xmin=0.00, xmax=100, color="black", linestyle="dashed", linewidth=0.5, zorder=9)
#Axes_obj_02.hlines(y=0 + (100/9)*0 + (100/18), xmin=0.00, xmax=100, color="black", linestyle="dashed", linewidth=0.5, zorder=9)

Axes_obj_02.text(10, 0 + (100/9)*7 + (100/18), "0h - 3h", size=10.0, color="black", ha='center', va='center', fontweight="bold", zorder=10)
Axes_obj_02.text(10, 0 + (100/9)*6 + (100/18), "3h - 6h", size=10.0, color="black", ha='center', va='center', fontweight="bold", zorder=10)
Axes_obj_02.text(10, 0 + (100/9)*5 + (100/18), "6h - 9h", size=10.0, color="black", ha='center', va='center', fontweight="bold", zorder=10)
Axes_obj_02.text(10, 0 + (100/9)*4 + (100/18), "9h - 12h", size=10.0, color="black", ha='center', va='center', fontweight="bold", zorder=10)
Axes_obj_02.text(10, 0 + (100/9)*3 + (100/18), "12h - 15h", size=10.0, color="black", ha='center', va='center', fontweight="bold", zorder=10)
Axes_obj_02.text(10, 0 + (100/9)*2 + (100/18), "15h - 18h", size=10.0, color="black", ha='center', va='center', fontweight="bold", zorder=10)
Axes_obj_02.text(10, 0 + (100/9)*1 + (100/18), "18h - 21h", size=10.0, color="black", ha='center', va='center', fontweight="bold", zorder=10)
Axes_obj_02.text(10, 0 + (100/9)*0 + (100/18), "21h - 24h", size=10.0, color="black", ha='center', va='center', fontweight="bold", zorder=10)

Axes_obj_02.text(10, 0 + (100/9)*8 + (100/18), "Time", size=10.0, color="black", ha='center', va='center', fontweight="bold", zorder=10)
Axes_obj_02.text(30, 0 + (100/9)*8 + (100/18), "Male", size=10.0, color="black", ha='center', va='center', fontweight="bold", zorder=10)
Axes_obj_02.text(50, 0 + (100/9)*8 + (100/18), "Female", size=10.0, color="black", ha='center', va='center', fontweight="bold", zorder=10)
Axes_obj_02.text(70, 0 + (100/9)*8 + (100/18), "Total", size=10.0, color="black", ha='center', va='center', fontweight="bold", zorder=10)
Axes_obj_02.text(90, 0 + (100/9)*8 + (100/18), "Peak", size=10.0, color="black", ha='center', va='center', fontweight="bold", zorder=10)

Axes_obj_02.vlines(x=0.0+1.25, ymin=0 + (100/9)*0+1.25, ymax=0 + (100/9)*4-1.25, color="red", linestyle="dashed", linewidth=1.0, zorder=9)
Axes_obj_02.vlines(x=20.0-1.25, ymin=0 + (100/9)*0+1.25, ymax=0 + (100/9)*4-1.25, color="red", linestyle="dashed", linewidth=1.0, zorder=9)
Axes_obj_02.hlines(y=0 + (100/9)*0+1.25, xmin=0.00+1.25, xmax=20.0-1.25, color="red", linestyle="dashed", linewidth=1.0, zorder=9)
Axes_obj_02.hlines(y=0 + (100/9)*4-1.25, xmin=0.00+1.25, xmax=20.0-1.25, color="red", linestyle="dashed", linewidth=1.0, zorder=9)

Axes_obj_02.quiver(95, 0 + (100/9)*5 + (100/18), -10, 0.0, scale_units='xy', angles='xy', scale=1, linestyle='solid', color="red", width=0.006, zorder=10)
Axes_obj_02.quiver(95, 0 + (100/9)*3 + (100/18), -10, 0.0, scale_units='xy', angles='xy', scale=1, linestyle='solid', color="red", width=0.006, zorder=10)


Axes_obj_02.text(30, 0 + (100/9)*7 + (100/18), DataSet_Freq_Table["Male"][0], size=10.0, color="black", ha='center', va='center', fontweight="normal", zorder=10)
Axes_obj_02.text(30, 0 + (100/9)*6 + (100/18), DataSet_Freq_Table["Male"][1], size=10.0, color="black", ha='center', va='center', fontweight="normal", zorder=10)
Axes_obj_02.text(30, 0 + (100/9)*5 + (100/18), DataSet_Freq_Table["Male"][2], size=10.0, color="black", ha='center', va='center', fontweight="normal", zorder=10)
Axes_obj_02.text(30, 0 + (100/9)*4 + (100/18), DataSet_Freq_Table["Male"][3], size=10.0, color="black", ha='center', va='center', fontweight="normal", zorder=10)
Axes_obj_02.text(30, 0 + (100/9)*3 + (100/18), DataSet_Freq_Table["Male"][4], size=10.0, color="black", ha='center', va='center', fontweight="normal", zorder=10)
Axes_obj_02.text(30, 0 + (100/9)*2 + (100/18), DataSet_Freq_Table["Male"][5], size=10.0, color="black", ha='center', va='center', fontweight="normal", zorder=10)
Axes_obj_02.text(30, 0 + (100/9)*1 + (100/18), DataSet_Freq_Table["Male"][6], size=10.0, color="black", ha='center', va='center', fontweight="normal", zorder=10)
Axes_obj_02.text(30, 0 + (100/9)*0 + (100/18), DataSet_Freq_Table["Male"][7], size=10.0, color="black", ha='center', va='center', fontweight="normal", zorder=10)

Axes_obj_02.text(50, 0 + (100/9)*7 + (100/18), DataSet_Freq_Table["Female"][0], size=10.0, color="black", ha='center', va='center', fontweight="normal", zorder=10)
Axes_obj_02.text(50, 0 + (100/9)*6 + (100/18), DataSet_Freq_Table["Female"][1], size=10.0, color="black", ha='center', va='center', fontweight="normal", zorder=10)
Axes_obj_02.text(50, 0 + (100/9)*5 + (100/18), DataSet_Freq_Table["Female"][2], size=10.0, color="black", ha='center', va='center', fontweight="normal", zorder=10)
Axes_obj_02.text(50, 0 + (100/9)*4 + (100/18), DataSet_Freq_Table["Female"][3], size=10.0, color="black", ha='center', va='center', fontweight="normal", zorder=10)
Axes_obj_02.text(50, 0 + (100/9)*3 + (100/18), DataSet_Freq_Table["Female"][4], size=10.0, color="black", ha='center', va='center', fontweight="normal", zorder=10)
Axes_obj_02.text(50, 0 + (100/9)*2 + (100/18), DataSet_Freq_Table["Female"][5], size=10.0, color="black", ha='center', va='center', fontweight="normal", zorder=10)
Axes_obj_02.text(50, 0 + (100/9)*1 + (100/18), DataSet_Freq_Table["Female"][6], size=10.0, color="black", ha='center', va='center', fontweight="normal", zorder=10)
Axes_obj_02.text(50, 0 + (100/9)*0 + (100/18), DataSet_Freq_Table["Female"][7], size=10.0, color="black", ha='center', va='center', fontweight="normal", zorder=10)

Axes_obj_02.text(70, 0 + (100/9)*7 + (100/18), DataSet_Freq_Table["Total"][0], size=10.0, color="black", ha='center', va='center', fontweight="normal", zorder=10)
Axes_obj_02.text(70, 0 + (100/9)*6 + (100/18), DataSet_Freq_Table["Total"][1], size=10.0, color="black", ha='center', va='center', fontweight="normal", zorder=10)
Axes_obj_02.text(70, 0 + (100/9)*5 + (100/18), DataSet_Freq_Table["Total"][2], size=10.0, color="black", ha='center', va='center', fontweight="bold", zorder=10)
Axes_obj_02.text(70, 0 + (100/9)*4 + (100/18), DataSet_Freq_Table["Total"][3], size=10.0, color="black", ha='center', va='center', fontweight="normal", zorder=10)
Axes_obj_02.text(70, 0 + (100/9)*3 + (100/18), DataSet_Freq_Table["Total"][4], size=10.0, color="black", ha='center', va='center', fontweight="bold", zorder=10)
Axes_obj_02.text(70, 0 + (100/9)*2 + (100/18), DataSet_Freq_Table["Total"][5], size=10.0, color="black", ha='center', va='center', fontweight="normal", zorder=10)
Axes_obj_02.text(70, 0 + (100/9)*1 + (100/18), DataSet_Freq_Table["Total"][6], size=10.0, color="black", ha='center', va='center', fontweight="normal", zorder=10)
Axes_obj_02.text(70, 0 + (100/9)*0 + (100/18), DataSet_Freq_Table["Total"][7], size=10.0, color="black", ha='center', va='center', fontweight="normal", zorder=10)

############################################################################################################
# (c)
############################################################################################################
Axes_obj_03.set_title("Correlogram (autocorrelations in the time series data)",  size=11.4230, fontweight="normal")
Axes_obj_03.set_xlabel("Time, hr")
Axes_obj_03.set_ylabel("Correlation coefficient, r (Spearman)")
#Axes_obj_03.tick_params(length=0.0)
#Axes_obj_03.set_xticklabels([])
#Axes_obj_03.set_yticklabels([])
Axes_obj_03.set_xlim([0-(24*0.05), 24+(24*0.05)])
Axes_obj_03.set_ylim([-1.0-(1.0*0.05), 1.0+(1.0*0.05)])

Axes_obj_03.set_xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24])
Axes_obj_03.xaxis.set_minor_locator(ticker.MultipleLocator(1))

Axes_obj_03.set_yticks([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
Axes_obj_03.yaxis.set_minor_locator(ticker.MultipleLocator(0.125))

#Axes_obj_03.plot(X_value, R_coef_k, color="red", linestyle='solid', linewidth=0.5)
Axes_obj_03.scatter(X_value, R_coef_s, color="black", zorder=10)

Axes_obj_03.plot(Time, Y_value, color="red", linestyle='solid', linewidth=1.0, zorder=10)
#Figure_object.tight_layout()

for i in range(0, 13):
    Axes_obj_03.vlines(x=2.0*i, ymin=-1.0, ymax=1.0, color="gray", linestyle="dotted", linewidth=0.5, zorder=9)
for i in range(0, 9):
    Axes_obj_03.hlines(y=-1.0 + 0.25*i, xmin=0.0, xmax=24, color="gray", linestyle="dotted", linewidth=0.5, zorder=9)
Axes_obj_03.hlines(y=0, xmin=0.0, xmax=24, color="black", linestyle="solid", linewidth=0.5, zorder=9)

Axes_obj_03.vlines(x=Maxima_solve[0], ymin=-1.0, ymax=1.0, color="blue", linestyle="solid", linewidth=0.5, zorder=9)
Axes_obj_03.vlines(x=Maxima_solve[1], ymin=-1.0, ymax=0.75, color="blue", linestyle="solid", linewidth=0.5, zorder=9)
Axes_obj_03.vlines(x=Maxima_solve[2], ymin=-1.0, ymax=1.0, color="blue", linestyle="solid", linewidth=0.5, zorder=9)

Axes_obj_03.text((Maxima_solve[0] + Maxima_solve[1])/2, 0.725, width_01_h, size=10.0, color="black", ha='center', va='center',
                 bbox=dict(boxstyle='round', edgecolor='blue', fc='white', alpha=1.0), zorder=10)
Axes_obj_03.quiver((Maxima_solve[0] + Maxima_solve[1])/2, 0.725, -((Maxima_solve[0] + Maxima_solve[1])/2 - Maxima_solve[0]), 0.0, scale_units='xy', angles='xy', scale=1, linestyle='solid', color="blue", width=0.006, zorder=9)
Axes_obj_03.quiver((Maxima_solve[0] + Maxima_solve[1])/2, 0.725, ((Maxima_solve[0] + Maxima_solve[1])/2 - Maxima_solve[0]), 0.0, scale_units='xy', angles='xy', scale=1, linestyle='solid', color="blue", width=0.006, zorder=9)

Axes_obj_03.text((Maxima_solve[1] + Maxima_solve[2])/2, 0.725, width_02_h, size=10.0, color="black", ha='center', va='center',
                 bbox=dict(boxstyle='round', edgecolor='blue', fc='white', alpha=1.0), zorder=10)
Axes_obj_03.quiver((Maxima_solve[1] + Maxima_solve[2])/2, 0.725, -((Maxima_solve[1] + Maxima_solve[2])/2 - Maxima_solve[1]), 0.0, scale_units='xy', angles='xy', scale=1, linestyle='solid', color="blue", width=0.006, zorder=9)
Axes_obj_03.quiver((Maxima_solve[1] + Maxima_solve[2])/2, 0.725, ((Maxima_solve[1] + Maxima_solve[2])/2 - Maxima_solve[1]), 0.0, scale_units='xy', angles='xy', scale=1, linestyle='solid', color="blue", width=0.006, zorder=9)

Axes_obj_03.text((Maxima_solve[0] + Maxima_solve[2])/2, 0.875, width_T_h, size=10.0, color="black", ha='center', va='center',
                 bbox=dict(boxstyle='round', edgecolor='blue', fc='white', alpha=1.0), zorder=10)
Axes_obj_03.quiver((Maxima_solve[0] + Maxima_solve[2])/2, 0.875, -((Maxima_solve[0] + Maxima_solve[2])/2 - Maxima_solve[0]), 0.0, scale_units='xy', angles='xy', scale=1, linestyle='solid', color="blue", width=0.006, zorder=9)
Axes_obj_03.quiver((Maxima_solve[0] + Maxima_solve[2])/2, 0.875, ((Maxima_solve[0] + Maxima_solve[2])/2 - Maxima_solve[0]), 0.0, scale_units='xy', angles='xy', scale=1, linestyle='solid', color="blue", width=0.006, zorder=9)

Axes_obj_03.text(3.0, 0.5,
                 r"$f(x) = n_{1}\sin{\{a_{1}\left(x-b_{1}\right)\}} + n_{2}\sin{\{a_{2}\left(x-b_{2}\right)\}}$",
                 horizontalalignment='left', fontsize=14.6848*0.85, color="black",
                 bbox=dict(boxstyle='round', edgecolor='red', fc='white', alpha=1.0), zorder=10)

############################################################################################################
# (d)
############################################################################################################
Axes_obj_04.set_title("Distribution of Age (by Sex)", size=11.4230, fontweight="normal")
Axes_obj_04.set_xlabel('Age', fontweight="normal")
Axes_obj_04.set_ylabel('Frequency (Number of Patients)', fontweight="normal")

Axes_obj_04.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
Axes_obj_04.xaxis.set_minor_locator(ticker.MultipleLocator(5))
Axes_obj_04.set_yticks([0, 2, 4, 6, 8, 10, 12])
Axes_obj_04.yaxis.set_minor_locator(ticker.MultipleLocator(1))

Axes_obj_04.hist(DataSet_Male["Age"],
                 range=(0.0+0.3, 100+0.3), bins=20, edgecolor="blue", color="blue", alpha=0.4,
                 orientation="vertical", linestyle="solid", linewidth=1.0, zorder=9)

Axes_obj_04.hist(DataSet_Female["Age"],
                 range=(0.0-0.3, 100-0.3), bins=20, edgecolor="red", color="red", alpha=0.4,
                 orientation="vertical", linestyle="solid", linewidth=1.0, zorder=8)

from scipy.stats import gaussian_kde
kde_model = gaussian_kde(DataSet_Male["Age"])
x = np.linspace(0.0, 100, num=100)
y = kde_model(x)
Axes_obj_04.plot(x, y*(5/np.max(y)), color="blue", label="Male", zorder=10)

from scipy.stats import gaussian_kde
kde_model = gaussian_kde(DataSet_Female["Age"])
x = np.linspace(0.0, 100, num=100)
y = kde_model(x)
Axes_obj_04.plot(x, y*(11/np.max(y)), color="red", label="Female", zorder=10)

Axes_obj_04.legend(title="KDE", title_fontsize=11.4230, edgecolor="black", loc=(0.01, 0.775))

Axes_obj_04.annotate(
 text='\n'.join(["Male", "N=15", "Mean: 55.8", "SD: 15.2", "Median: 58.0"]),
    xy=(7.25, 7.0), xytext=(7.25, 7.0), ha='center', va='center', size=10.0,
    bbox=dict(boxstyle='round', edgecolor='blue', fc='white'),
    arrowprops=dict(
        facecolor='blue', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0, edgecolor='blue', shrink=0.1)
)

Axes_obj_04.annotate(
 text='\n'.join(["Female", "N=49", "Mean: 59.3", "SD: 11.9", "Median: 60.0"]),
    xy=(7.25, 3.75), xytext=(7.25, 3.75), ha='center', va='center', size=10.0,
    bbox=dict(boxstyle='round', edgecolor='red', fc='white'),
    arrowprops=dict(
        facecolor='red', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0, edgecolor='red', shrink=0.1)
)


########################################################################################################################
Figure_object.tight_layout()
my_path = os.path.abspath("Figures")
Figure_object.savefig(os.path.join(my_path, "01_Kimoto_et_al_(2023)_[Fig_S_04].png"))

from PIL import Image # py -m pip install pillow
img = Image.open(os.path.join(my_path, "01_Kimoto_et_al_(2023)_[Fig_S_04].png"))
img_resize = img.resize(size=(2866, 2016)) #size in pixels, as a 2-tuple: (width, height)
my_path2 = os.path.abspath("Figures/Fixed_Version/B6")
img_resize.save(os.path.join(my_path2, "01_Kimoto_et_al_(2023)_[Fig_S_04]_B6.png"))

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












