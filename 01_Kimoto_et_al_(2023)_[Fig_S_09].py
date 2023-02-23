########################################################################################################################
# Fig. S9. Relationship between age and travel duration in the data reported by Clérel & Caillard (1999).
# (01_Kimoto_et_al_(2023)_[Fig_S_09].py)
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

# U.S. BUREAU OF LABOR STATISTICS
# Postal Square Building 2 Massachusetts Avenue NE Washington, DC 20212-0001
# Telephone:1-202-691-5200 Telecommunications Relay Service:7-1-1
# www.bls.gov
# Men's_earnings_at_peak_at_age_45-54_JULY_23,_1999
# Median usual weekly earnings of full-time male workers, by age, second quarter 1999 averages
# https://www.bls.gov/opub/ted/1999/jul/wk3/art05.htm

Age_16_to_24_E = ["16 to 24", 20, 344]
Age_25_to_34_E = ["25 to 34", 29.5, 573]
Age_35_to_44_E = ["35 to 44", 39.5, 706]
Age_44_to_54_E = ["44 to 54", 49.5, 757]
Age_55_to_64_E = ["55 to 64", 59.5, 715]
Age_65_and_over_E = ["65 and over", 69.5, 421]

DataSet_Men_earnings = pd.DataFrame(
    [Age_16_to_24_E, Age_25_to_34_E, Age_35_to_44_E, Age_44_to_54_E, Age_55_to_64_E, Age_65_and_over_E],
    index=["16 to 24", "25 to 34", "35 to 44", "44 to 54", "55 to 64", "65 and over"],
    columns=["Age label", "Age", "Earning"])

print(DataSet_Men_earnings)

########################################################################################################################

########################################################################################################################
# Figures
########################################################################################################################
Figure_object = plt.figure(1, figsize=(11.69, 8.27), dpi=400, edgecolor="black", linewidth=0.5)
plt.subplots_adjust(left=0.125, bottom=0.11, right=0.9, top=0.88, wspace=0.2, hspace=0.2)

gs_master = matplotlib.gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[1, 1])
gs_1 = matplotlib.gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=2, subplot_spec=gs_master[0])
gs_2 = matplotlib.gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=2, subplot_spec=gs_master[1])

#Axes_obj_01 = Figure_object.add_subplot(gs_1[0])
#Axes_obj_02 = Figure_object.add_subplot(gs_1[1])
#Axes_obj_03 = Figure_object.add_subplot(gs_2[0])
#Axes_obj_04 = Figure_object.add_subplot(gs_2[1])

Axes_obj_01 = plt.axes([0.0525794, 0.560191, 0.424813, 0.39648])
Axes_obj_02 = plt.axes([0.562355, 0.560191, 0.424813, 0.39648])
Axes_obj_03 = plt.axes([0.0525794, 0.0692597, 0.424813, 0.39648])
Axes_obj_04 = plt.axes([0.562355, 0.0692597, 0.424813, 0.39648])

plt.figtext(0.0200, 0.970, "a", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")
plt.figtext(0.5150, 0.970, "b", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")
plt.figtext(0.0200, 0.500, "c", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")
plt.figtext(0.5150, 0.500, "d", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")

########################################################################################################################

############################################################################################################
# (a) Thrombosis onset, Paris airport, Men (Clérel & Caillard, 1999)
############################################################################################################
Axes_obj_01.set_title("Thrombosis onset, Paris airport, Men (Clérel & Caillard, 1999)", size=11.4230, fontweight="normal")
Axes_obj_01.set_xlabel('Age', fontweight="normal")
Axes_obj_01.set_ylabel('Travel duration, hr', fontweight="normal")
#Axes_obj_01.tick_params(length=0.0)
Axes_obj_01.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
#Axes_obj_01.set_yticks([])
Axes_obj_01.xaxis.set_minor_locator(ticker.MultipleLocator(5))
Axes_obj_01.yaxis.set_minor_locator(ticker.MultipleLocator(1))
#Axes_obj_01.imshow(plt.imread("Clerel_Fig_03.jpg"))

Axes_obj_01.set_xlim([0-(100*0.05), 100+(100*0.05)])
Axes_obj_01.set_ylim([0.0-(24*0.05), 24+(24*0.05)])

for i in range(0, 11):
    Axes_obj_01.vlines(x=10.0*i, ymin=0.0, ymax=24, color="gray", linestyle="dotted", linewidth=0.5, zorder=8)
for i in range(0, 25):
    Axes_obj_01.hlines(y=1.0*i, xmin=0.0, xmax=100, color="gray", linestyle="dotted", linewidth=0.5, zorder=8)

Axes_obj_01.scatter(DataSet_Male_2["Age"], DataSet_Male_2["Duration_int"],
                    marker='o', s=60, color="blue", alpha=0.6, edgecolors="blue", zorder=10)

Corrcoef_Men = np.corrcoef(DataSet_Male_2["Age"], DataSet_Male_2["Duration_int"])
print(Corrcoef_Men)

from scipy.stats import pearsonr
res = pearsonr(DataSet_Male_2["Age"], DataSet_Male_2["Duration_int"])
r_value = res[0]
p_value = res[1]

print('r：', r_value)
print('p-value：', p_value)

####################################################################
# Fitted simple one peak curve
####################################################################

def reg_func_01(parameter: ["L", "a", "d"], x, y):
    L = parameter[0]
    a = parameter[1]
    d = parameter[2]

    _Output_ = y - ( L*np.exp(-a*(x-d)**(2)) )

    return _Output_
x = DataSet_Male_2["Age"]; y = DataSet_Male_2["Duration_int"]
parameter_0 = [17, 0.01, 45]
RegResult_Original_W_01 = optimize.leastsq(reg_func_01, parameter_0, args=(x, y), full_output=True)
Result_W_01_0 = RegResult_Original_W_01[0]
Result_W_01 = [Result_W_01_0[0], Result_W_01_0[1], Result_W_01_0[2]]
print(Result_W_01)

X_line_1_1 = np.linspace(0.0, 80, 1000)
Y_line_1_1 = Result_W_01[0]*np.exp(-Result_W_01[1]*(X_line_1_1-Result_W_01[2])**(2))
Axes_obj_01.plot(X_line_1_1, Y_line_1_1, color="blue", linestyle="solid", linewidth=2.0, zorder=10,
                 label="Fitted simple one peak curve")

X_line_1_1_b = np.linspace(80, 100, 1000)
Y_line_1_1_b = Result_W_01[0]*np.exp(-Result_W_01[1]*(X_line_1_1_b-Result_W_01[2])**(2))
Axes_obj_01.plot(X_line_1_1_b, Y_line_1_1_b, color="blue", linestyle="dotted", linewidth=1.0, zorder=10,
                 )

Axes_obj_01.vlines(x=Result_W_01[2],
                   ymin=0.0, ymax=Result_W_01[0]*np.exp(-Result_W_01[1]*(Result_W_01[2]-Result_W_01[2])**(2)),
                   color="blue", linestyle="solid", linewidth=0.75, zorder=10)

Peak_x = Decimal(str(Result_W_01[2])).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)
Axes_obj_01.annotate(text=Peak_x,
                     bbox= dict(boxstyle='round', edgecolor='blue', fc='white'),
                       xy=(Result_W_01[2], 2.0), ha='center', va='center', zorder=10)

####################################################################
# Fitted combined double S curves
####################################################################

def reg_func_01(parameter: ["L", "a1", "a2", "d1", "d2"], x, y):
    L = parameter[0]
    a1 = parameter[1]
    a2 = parameter[2]
    d1 = parameter[3]
    d2 = parameter[4]

    _Output_ = y - ( ((L)/(1+np.exp(-a1*(x-d1))))+((L)/(1+np.exp(a2*(x-d2))))-L )

    return _Output_
x = DataSet_Male_2["Age"]; y = DataSet_Male_2["Duration_int"]
parameter_0 = [11.6, 0.32, 0.08, 31.1, 45.0]
RegResult_Original_W_01 = optimize.leastsq(reg_func_01, parameter_0, args=(x, y), full_output=True)
Result_W_01_0 = RegResult_Original_W_01[0]
Result_W_01 = [Result_W_01_0[0], Result_W_01_0[1], Result_W_01_0[2], Result_W_01_0[3], Result_W_01_0[4]]
print(Result_W_01)

X_line_1_2 = np.linspace(0.0, 100, 1000)
Y_line_1_2 = ((Result_W_01[0])/(1+np.exp(-Result_W_01[1]*(X_line_1_2-Result_W_01[3]))))+((Result_W_01[0])/(1+np.exp(Result_W_01[2]*(X_line_1_2-Result_W_01[4]))))-Result_W_01[0]
Axes_obj_01.plot(X_line_1_2, Y_line_1_2, color="blue", linestyle="dashed", linewidth=1.0, zorder=10,
                 label="Fitted combined double S curves")

####################################################################
# Spurious regression line
####################################################################

def reg_func_01(parameter: ["a", "b"], x, y):
    a = parameter[0]
    b = parameter[1]

    _Output_ = y - ( a*x + b )

    return _Output_
x = DataSet_Male_2["Age"]; y = DataSet_Male_2["Duration_int"]
parameter_0 = [0.5, 5.0]
RegResult_Original_W_01 = optimize.leastsq(reg_func_01, parameter_0, args=(x, y), full_output=True)
Result_W_01_0 = RegResult_Original_W_01[0]
Result_W_01 = [Result_W_01_0[0], Result_W_01_0[1]]
print(Result_W_01)

X_line_1_3 = np.linspace(10, 80, 1000)
Y_line_1_3 = Result_W_01[0]*X_line_1_3 + Result_W_01[1]
Axes_obj_01.plot(X_line_1_3, Y_line_1_3, color="black", linestyle="solid", linewidth=1.0, zorder=10,
                 label="Spurious regression line")


Axes_obj_01.legend(facecolor='white', edgecolor="white", loc=(0.05, 0.75))

#def reg_func_01(parameter: ["L", "a", "b", "c", "d1", "d2"], x, y):
#    L = parameter[0]
#    a = parameter[1]
#    b = parameter[2]
#    c = parameter[3]
#    d1 = parameter[4]
#    d2 = parameter[5]
#
#    _Output_ = y - ( ((L)/(1+np.exp(-a*(x-d1)))) -b**(c*(x-d2)) )
#
#    return _Output_
#x = DataSet_Male_2["Age"]; y = DataSet_Male_2["Duration_int"]
#
#parameter_0 = [11.6, 0.32, 2, 0.25, 31.1, 65]
#RegResult_Original_W_01 = optimize.leastsq(reg_func_01, parameter_0, args=(x, y), full_output=True)
#Result_W_01_0 = RegResult_Original_W_01[0]
#Result_W_01 = [Result_W_01_0[0], Result_W_01_0[1], Result_W_01_0[2], Result_W_01_0[3], Result_W_01_0[4], Result_W_01_0[5]]
#print(Result_W_01)

#X_line = np.linspace(0.0, 80, 1000)
#Y_line = ((Result_W_01[0])/(1+np.exp(-Result_W_01[1]*(X_line-Result_W_01[4])))) -Result_W_01[2]**(Result_W_01[3]*(X_line-Result_W_01[5]))
#Axes_obj_01.plot(X_line, Y_line, color="red", linestyle="solid", linewidth=1.0, zorder=10)


Axes_obj_01.text(40.0, 15.0, "Correlation due to differences in the balance of data",
                 size=10.0, color="black", ha='center', va='center', fontweight="normal", zorder=10, rotation=18.5)

Axes_obj_01.text(
    90, 20,
    "No data", size=10.0, color="black", ha='center', va='center',
    fontweight="normal", zorder=10)

Axes_obj_01.fill_between(np.linspace(80, 100, 1000), 24*np.ones(1000), 0.00*np.ones(1000),
                         facecolor='gray', alpha=0.3)

############################################################################################################
# (b) Men's earnings (U.S. Bureau of Labor Statistics, 1999)
############################################################################################################
Axes_obj_02.set_title("Men's earnings (U.S. Bureau of Labor Statistics, 1999)", size=11.4230, fontweight="normal")
Axes_obj_02.set_xlabel('Age', fontweight="normal")
Axes_obj_02.set_ylabel('Earnings, $ (= responsibility, pressure)', fontweight="normal")
#Axes_obj_02.tick_params(length=0.0)
Axes_obj_02.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
#Axes_obj_02.set_yticks([])
Axes_obj_02.xaxis.set_minor_locator(ticker.MultipleLocator(5))
Axes_obj_02.yaxis.set_minor_locator(ticker.MultipleLocator(100))
#Axes_obj_02.imshow(plt.imread("Clerel_Fig_03.jpg"))

Axes_obj_02.set_xlim([0-(100*0.05), 100+(100*0.05)])
Axes_obj_02.set_ylim([0.0-(1300*0.05), 1300+(1300*0.05)])

for i in range(0, 11):
    Axes_obj_02.vlines(x=10.0*i, ymin=0.0, ymax=1300, color="gray", linestyle="dotted", linewidth=0.5, zorder=9)
for i in range(0, 14):
    Axes_obj_02.hlines(y=100.0*i, xmin=0.0, xmax=100, color="gray", linestyle="dotted", linewidth=0.5, zorder=9)

Axes_obj_02.scatter(DataSet_Men_earnings["Age"], DataSet_Men_earnings["Earning"],
                    marker='o', s=60, color="blue", alpha=0.6, edgecolors="blue", zorder=10)

Corrcoef_Men_E = np.corrcoef(DataSet_Men_earnings["Age"], DataSet_Men_earnings["Earning"])
print(Corrcoef_Men_E)

####################################################################
# Fitted simple one peak curve
####################################################################

def reg_func_02(parameter: ["L", "a", "d"], x, y):
    L = parameter[0]
    a = parameter[1]
    d = parameter[2]

    _Output_ = y - ( L*np.exp(-a*(x-d)**(2)) )

    return _Output_
x = DataSet_Men_earnings["Age"]; y = DataSet_Men_earnings["Earning"]
parameter_0 = [750, 0.01, 45]
RegResult_Original_W_02 = optimize.leastsq(reg_func_02, parameter_0, args=(x, y), full_output=True)
Result_W_02_0 = RegResult_Original_W_02[0]
Result_W_02 = [Result_W_02_0[0], Result_W_02_0[1], Result_W_02_0[2]]
print(Result_W_02)

X_line_2_1 = np.linspace(0.0, 100, 1000)
Y_line_2_1 = Result_W_02[0]*np.exp(-Result_W_02[1]*(X_line_2_1-Result_W_02[2])**(2))
Axes_obj_02.plot(X_line_2_1, Y_line_2_1, color="blue", linestyle="solid", linewidth=2.0, zorder=10,
                 label="Fitted simple one peak curve")

Axes_obj_02.vlines(x=Result_W_02[2],
                   ymin=0.0, ymax=Result_W_02[0]*np.exp(-Result_W_02[1]*(Result_W_02[2]-Result_W_02[2])**(2)),
                   color="blue", linestyle="solid", linewidth=0.75, zorder=10)

Peak_x = Decimal(str(Result_W_02[2])).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)
Axes_obj_02.annotate(text=Peak_x,
                     bbox= dict(boxstyle='round', edgecolor='blue', fc='white'),
                       xy=(Result_W_02[2], 100), ha='center', va='center', zorder=10)


####################################################################
# Fitted combined double S curves
####################################################################

def reg_func_02(parameter: ["L", "a1", "a2", "d1", "d2"], x, y):
    L = parameter[0]
    a1 = parameter[1]
    a2 = parameter[2]
    d1 = parameter[3]
    d2 = parameter[4]

    _Output_ = y - ( ((L)/(1+np.exp(-a1*(x-d1))))+((L)/(1+np.exp(a2*(x-d2))))-L )

    return _Output_
x = DataSet_Men_earnings["Age"]; y = DataSet_Men_earnings["Earning"]
parameter_0 = [800, 0.32, 0.08, 31.1, 45.0]
RegResult_Original_W_02 = optimize.leastsq(reg_func_02, parameter_0, args=(x, y), full_output=True)
Result_W_02_0 = RegResult_Original_W_02[0]
Result_W_02 = [Result_W_02_0[0], Result_W_02_0[1], Result_W_02_0[2], Result_W_02_0[3], Result_W_02_0[4]]
print(Result_W_02)

X_line_2_2 = np.linspace(0.0, 100, 1000)
Y_line_2_2 = ((Result_W_02[0])/(1+np.exp(-Result_W_02[1]*(X_line_2_2-Result_W_02[3]))))+((Result_W_02[0])/(1+np.exp(Result_W_02[2]*(X_line_2_2-Result_W_02[4]))))-Result_W_02[0]
Axes_obj_02.plot(X_line_2_2, Y_line_2_2, color="blue", linestyle="dashed", linewidth=1.0, zorder=10,
                 label="Fitted combined double S curves")

####################################################################
# Spurious regression line
####################################################################

def reg_func_02(parameter: ["a", "b"], x, y):
    a = parameter[0]
    b = parameter[1]

    _Output_ = y - ( a*x + b )

    return _Output_
x = DataSet_Men_earnings["Age"]; y = DataSet_Men_earnings["Earning"]
parameter_0 = [6.0, 400]
RegResult_Original_W_02 = optimize.leastsq(reg_func_02, parameter_0, args=(x, y), full_output=True)
Result_W_02_0 = RegResult_Original_W_02[0]
Result_W_02 = [Result_W_02_0[0], Result_W_02_0[1]]
print(Result_W_02)

X_line_2_3 = np.linspace(10, 90, 1000)
Y_line_2_3 = Result_W_02[0]*X_line_2_3 + Result_W_02[1]
Axes_obj_02.plot(X_line_2_3, Y_line_2_3, color="black", linestyle="solid", linewidth=1.0, zorder=10,
                 label="Spurious regression line")


Axes_obj_02.legend(facecolor='white', edgecolor="white", loc=(0.05, 0.75))

Axes_obj_02.text(
    DataSet_Men_earnings["Age"].iat[0]-10, DataSet_Men_earnings["Earning"].iat[0],
    DataSet_Men_earnings["Age label"].iat[0], size=10.0, color="black", ha='center', va='center',
    fontweight="normal", zorder=10)

Axes_obj_02.text(
    DataSet_Men_earnings["Age"].iat[1]-10, DataSet_Men_earnings["Earning"].iat[1]+100,
    DataSet_Men_earnings["Age label"].iat[1], size=10.0, color="black", ha='center', va='center',
    fontweight="normal", zorder=10)

Axes_obj_02.text(
    DataSet_Men_earnings["Age"].iat[2]-5, DataSet_Men_earnings["Earning"].iat[2]+100,
    DataSet_Men_earnings["Age label"].iat[2], size=10.0, color="black", ha='center', va='center',
    fontweight="normal", zorder=10)

Axes_obj_02.text(
    DataSet_Men_earnings["Age"].iat[3], DataSet_Men_earnings["Earning"].iat[3]+150,
    DataSet_Men_earnings["Age label"].iat[3], size=10.0, color="black", ha='center', va='center',
    fontweight="normal", zorder=10)

Axes_obj_02.text(
    DataSet_Men_earnings["Age"].iat[4]+5, DataSet_Men_earnings["Earning"].iat[4]+100,
    DataSet_Men_earnings["Age label"].iat[4], size=10.0, color="black", ha='center', va='center',
    fontweight="normal", zorder=10)

Axes_obj_02.text(
    DataSet_Men_earnings["Age"].iat[5]+15, DataSet_Men_earnings["Earning"].iat[5],
    DataSet_Men_earnings["Age label"].iat[5], size=10.0, color="black", ha='center', va='center',
    fontweight="normal", zorder=10)


Axes_obj_02.fill_between(np.linspace(65, 100, 1000), 1300*np.ones(1000), 0.00*np.ones(1000),
                         facecolor='gray', alpha=0.3)

Axes_obj_02.text(
    82.5, 1100,
    "Retirement age", size=10.0, color="black", ha='center', va='center',
    fontweight="normal", zorder=10)

############################################################################################################
# (c) Thrombosis onset, Paris airport, Women (Clérel & Caillard, 1999)
############################################################################################################
Axes_obj_03.set_title("Thrombosis onset, Paris airport, Women (Clérel & Caillard, 1999)",
                      size=11.4230, fontweight="normal")
Axes_obj_03.set_xlabel('Age', fontweight="normal")
Axes_obj_03.set_ylabel('Travel duration, hr', fontweight="normal")
#Axes_obj_03.tick_params(length=0.0)
Axes_obj_03.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
#Axes_obj_03.set_yticks([])
Axes_obj_03.xaxis.set_minor_locator(ticker.MultipleLocator(5))
Axes_obj_03.yaxis.set_minor_locator(ticker.MultipleLocator(1))
#Axes_obj_03.imshow(plt.imread("Clerel_Fig_03.jpg"))

Axes_obj_03.set_xlim([0-(100*0.05), 100+(100*0.05)])
Axes_obj_03.set_ylim([0.0-(24*0.05), 24+(24*0.05)])

for i in range(0, 11):
    Axes_obj_03.vlines(x=10.0*i, ymin=0.0, ymax=24, color="gray", linestyle="dotted", linewidth=0.5, zorder=9)
for i in range(0, 25):
    Axes_obj_03.hlines(y=1.0*i, xmin=0.0, xmax=100, color="gray", linestyle="dotted", linewidth=0.5, zorder=9)

Axes_obj_03.scatter(DataSet_Female_2["Age"], DataSet_Female_2["Duration_int"],
                    marker='o', s=60, color="red", alpha=0.6, edgecolors="red", zorder=10)

Corrcoef_Women = np.corrcoef(DataSet_Female_2["Age"], DataSet_Female_2["Duration_int"])
print(Corrcoef_Women)

from scipy.stats import pearsonr
res = pearsonr(DataSet_Female_2["Age"], DataSet_Female_2["Duration_int"])
r_value = res[0]
p_value = res[1]

print('r：', r_value)
print('p-value：', p_value)

####################################################################
# Fitted simple one peak curve
####################################################################

def reg_func_03(parameter: ["L", "a", "d"], x, y):
    L = parameter[0]
    a = parameter[1]
    d = parameter[2]
    L2 = parameter[3]

    _Output_ = y - ( L2-L*np.exp(-a*(x-d)**(2)) )

    return _Output_
x = DataSet_Female_2["Age"]; y = DataSet_Female_2["Duration_int"]
parameter_0 = [17, 0.01, 45, 30]
RegResult_Original_W_03 = optimize.leastsq(reg_func_03, parameter_0, args=(x, y), full_output=True)
Result_W_03_0 = RegResult_Original_W_03[0]
Result_W_03 = [Result_W_03_0[0], Result_W_03_0[1], Result_W_03_0[2], Result_W_03_0[3]]
print(Result_W_03)

X_line_3_1 = np.linspace(0.0, 100, 1000)
Y_line_3_1 = Result_W_03[3] - Result_W_03[0]*np.exp(-Result_W_03[1]*(X_line_3_1-Result_W_03[2])**(2))
Axes_obj_03.plot(X_line_3_1, Y_line_3_1, color="red", linestyle="solid", linewidth=2.0, zorder=10,
                 label="Fitted simple one peak curve")
Axes_obj_03.vlines(
    x=Result_W_03[2], ymin=0.0,
    ymax=Result_W_03[3] - Result_W_03[0]*np.exp(-Result_W_03[1]*(Result_W_03[2]-Result_W_03[2])**(2)),
                   color="red", linestyle="solid", linewidth=0.75, zorder=10)

Peak_x = Decimal(str(Result_W_03[2])).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)
Axes_obj_03.annotate(text=Peak_x,
                     bbox= dict(boxstyle='round', edgecolor='red', fc='white'),
                       xy=(Result_W_03[2], 2.0), ha='center', va='center', zorder=10)

####################################################################
# Fitted combined double S curves
####################################################################

def reg_func_03(parameter: ["L", "a1", "a2", "d1", "d2", "L2"], x, y):
    L = parameter[0]
    a1 = parameter[1]
    a2 = parameter[2]
    d1 = parameter[3]
    d2 = parameter[4]
    L2 = parameter[5]

    _Output_ = y - ( ((L)/(1+np.exp(-a1*(x-d1))))+((L)/(1+np.exp(a2*(x-d2))))-L2 )

    return _Output_
x = DataSet_Female_2["Age"]; y = DataSet_Female_2["Duration_int"]
parameter_0 = [10, 0.25, 0.4, 65, 30, 10]
RegResult_Original_W_03 = optimize.leastsq(reg_func_03, parameter_0, args=(x, y), full_output=True)
Result_W_03_0 = RegResult_Original_W_03[0]
Result_W_03 = [Result_W_03_0[0], Result_W_03_0[1], Result_W_03_0[2], Result_W_03_0[3], Result_W_03_0[4], Result_W_03_0[5]]
print(Result_W_03)

X_line_3_2 = np.linspace(0.0, 100, 1000)
Y_line_3_2 = ((Result_W_03[0])/(1+np.exp(-Result_W_03[1]*(X_line_3_2-Result_W_03[3]))))+((Result_W_03[0])/(1+np.exp(Result_W_03[2]*(X_line_3_2-Result_W_03[4]))))-Result_W_03[5]
Axes_obj_03.plot(X_line_3_2, Y_line_3_2, color="red", linestyle="dashed", linewidth=1.0, zorder=10,
                 label="Fitted combined double S curves")

####################################################################
# Spurious regression line
####################################################################

def reg_func_03(parameter: ["a", "b"], x, y):
    a = parameter[0]
    b = parameter[1]

    _Output_ = y - ( a*x + b )

    return _Output_
x = DataSet_Female_2["Age"]; y = DataSet_Female_2["Duration_int"]
parameter_0 = [6.0, 400]
RegResult_Original_W_03 = optimize.leastsq(reg_func_03, parameter_0, args=(x, y), full_output=True)
Result_W_03_0 = RegResult_Original_W_03[0]
Result_W_03 = [Result_W_03_0[0], Result_W_03_0[1]]
print(Result_W_03)

X_line_3_3 = np.linspace(10, 90, 1000)
Y_line_3_3 = Result_W_03[0]*X_line_3_3 + Result_W_03[1]
Axes_obj_03.plot(X_line_3_3, Y_line_3_3, color="black", linestyle="solid", linewidth=1.0, zorder=10,
                 label="Spurious regression line")


Axes_obj_03.legend(facecolor='white', edgecolor="white", loc=(0.05, 0.75))


Axes_obj_03.fill_between(np.linspace(0.0, 50, 1000), 24*np.ones(1000), 0.00*np.ones(1000),
                         facecolor='gray', alpha=0.3)

Axes_obj_03.text(
    25, 5,
    "Small amount of data\nN=9", size=10.0, color="black", ha='center', va='center',
    fontweight="normal", zorder=10)

Axes_obj_03.text(
    75+2.5, 5,
    "Large amount of data\nN=40", size=10.0, color="black", ha='center', va='center',
    fontweight="normal", zorder=10)

############################################################################################################
# (d) Spurious correlation between duration and age (thrombosis risk)
############################################################################################################
Axes_obj_04.set_title("Spurious correlation between duration and age (thrombosis risk)",
                      size=11.4230, fontweight="normal")
Axes_obj_04.set_xlabel('Age (healthcare risk = thrombosis onset risk)', fontweight="normal")
Axes_obj_04.set_ylabel('Travel duration & social situation', fontweight="normal")
#Axes_obj_04.tick_params(length=0.0)
Axes_obj_04.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
#Axes_obj_04.set_yticks([])
Axes_obj_04.xaxis.set_minor_locator(ticker.MultipleLocator(5))
Axes_obj_04.yaxis.set_minor_locator(ticker.MultipleLocator(1))
#Axes_obj_04.imshow(plt.imread("Clerel_Fig_03.jpg"))

Axes_obj_04.set_xlim([0-(100*0.05), 100+(100*0.05)])
Axes_obj_04.set_ylim([0.0-(24*0.05), 24+(24*0.05)])

for i in range(0, 11):
    Axes_obj_04.vlines(x=10.0*i, ymin=0.0, ymax=24, color="gray", linestyle="dotted", linewidth=0.5, zorder=9)
for i in range(0, 25):
    Axes_obj_04.hlines(y=1.0*i, xmin=0.0, xmax=100, color="gray", linestyle="dotted", linewidth=0.5, zorder=9)

Axes_obj_04.plot(X_line_1_1, Y_line_1_1, color="blue", linestyle="solid", linewidth=1.0, zorder=10)
Axes_obj_04.plot(X_line_3_1, Y_line_3_1, color="red", linestyle="solid", linewidth=1.0, zorder=10)

Axes_obj_04.plot(X_line_1_3, Y_line_1_3, color="blue", linestyle="dashed", linewidth=0.75, zorder=10,
                 label="Spurious regression line")
Axes_obj_04.plot(X_line_3_3, Y_line_3_3, color="red", linestyle="dashed", linewidth=0.75, zorder=10,
                 label="Spurious regression line")

Axes_obj_04.quiver(10, 8.5, 85, 6.5, scale_units='xy', angles='xy', scale=1, color="black", width=0.0075)

Axes_obj_04.text(55, 10.75, "Spurious \"linear\" increasing of travel duration",
                 size=10.0, color="black", ha='center', va='center', fontweight="bold", zorder=10, rotation=10.75)


Axes_obj_04.text(17.5, 2.5, "You are young!\nnot intercontinental\nbusiness trip\n(low pressure)",
                 size=10.0, color="blue", ha='center', va='center', fontweight="bold", zorder=10)
Axes_obj_04.text(57, 5.5, "Go to international\nconference as president\nor something finest person\n(high pressure)",
                 size=10.0, color="blue", ha='center', va='center', fontweight="bold", zorder=10)
Axes_obj_04.text(85, 0.5, "Retirement\n(I've done enough..)",
                 size=10.0, color="blue", ha='center', va='center', fontweight="bold", zorder=10)


Axes_obj_04.text(15, 20, "Enjoy vacation!\nas Honeymoon\n(OC user?)",
                 size=10.0, color="red", ha='center', va='center', fontweight="bold", zorder=10)
Axes_obj_04.text(55, 17.5, "Busy!!\nMy husband’s presence is...",
                 size=10.0, color="red", ha='center', va='center', fontweight="bold", zorder=10)
Axes_obj_04.text(85, 20, "Enjoy vacation!\nwith friends?\n(HRT user?)",
                 size=10.0, color="red", ha='center', va='center', fontweight="bold", zorder=10)

########################################################################################################################
#Figure_object.tight_layout()
my_path = os.path.abspath("Figures")
Figure_object.savefig(os.path.join(my_path, "01_Kimoto_et_al_(2023)_[Fig_S_09].png"))

from PIL import Image # py -m pip install pillow
img = Image.open(os.path.join(my_path, "01_Kimoto_et_al_(2023)_[Fig_S_09].png"))
img_resize = img.resize(size=(2866, 2016)) #size in pixels, as a 2-tuple: (width, height)
my_path2 = os.path.abspath("Figures/Fixed_Version/B6")
img_resize.save(os.path.join(my_path2, "01_Kimoto_et_al_(2023)_[Fig_S_09]_B6.png"))

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