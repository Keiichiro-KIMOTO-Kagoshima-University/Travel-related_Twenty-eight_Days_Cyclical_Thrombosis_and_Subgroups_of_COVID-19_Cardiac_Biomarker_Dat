########################################################################################################################
# Fig. S5. Thrombosis reported by Clérel & Caillard & our novel annotations.(01_Kimoto_et_al_(2023)_[Fig_S_05].py)
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

Line_01 = ["Timeline Table 1: Years 1992–1997 (Recommendations of Hormone Replacement Therapy, HRT)                   "]
Line_02 = ["----------------------------------------------------------------------------------------------------------"]
Line_03 = [" Year Month Article Type   Journal (Citation)                    Contents or Recommendations              "]
Line_04 = ["----------------------------------------------------------------------------------------------------------"]
Line_05 = [" 1992  Dec  ACP Guideline  Ann Intern Med. 1992;117(12):1038-41. Consider for coronary disease            "]
Line_06 = [" 1995  Jul  AHA Guideline  J Am Coll Cardiol. 1995;26(1):292-4.  Mentioned risk reduction interventions   "]
Line_07 = [" 1997  May  AHA Guideline  Circulation. 1997;95(9):2329-31.      Consider HRT in postmenopausal women     "]
Line_08 = [" 1997  Jun  NHS Cohort     N Engl J Med. 1997;336(25):1769-75.   Mortality: HRT user < non-user           "]
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

########################################################################################################################
# Figures
########################################################################################################################
Figure_object = plt.figure(1, figsize=(11.69, 8.27), dpi=400, edgecolor="black", linewidth=0.5)

plt.figtext(0.0200, 0.970, "a", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")
#plt.figtext(0.5150, 0.970, "b", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")
plt.figtext(0.0200, 0.475, "c", horizontalalignment='center', fontsize=13.0549, fontweight="bold", color="black")

gs_master = matplotlib.gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[1, 1])
gs_1 = matplotlib.gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=2, subplot_spec=gs_master[0])
gs_2 = matplotlib.gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[1])

Axes_obj_01 = Figure_object.add_subplot(gs_1[0])
#Axes_obj_02 = Figure_object.add_subplot(gs_1[1])
Axes_obj_03 = Figure_object.add_subplot(gs_2[0])

########################################################################################################################
############################################################################################################
# (a)
############################################################################################################
Axes_obj_01.set_title("Clérel & Caillard 1999, Fig. 1.", size=11.4230,  fontweight="normal")
Axes_obj_01.set_xlabel("Year", fontweight="normal", size=10.0)
#Axes_obj_01.set_ylabel("No. of Pulmonary Thromboembolism", fontweight="normal", size=10.0)
Axes_obj_01.yaxis.label.set_color("blue")
Axes_obj_01.tick_params(length=0.0)
Axes_obj_01.set_xticks([])
Axes_obj_01.set_yticks([])

Axes_obj_01.imshow(plt.imread("Clerel_Fig_01.jpg"))

Axes_obj_01_2 = Axes_obj_01.twinx()
#Axes_obj_01_2.set_ylabel("No. of Passengers ( x10^6)", fontweight="normal", size=10.0)
Axes_obj_01_2.yaxis.label.set_color("red")
Axes_obj_01_2.tick_params(axis="y", colors="red")
Axes_obj_01_2.tick_params(length=0.0)
Axes_obj_01_2.set_xticks([])
Axes_obj_01_2.set_yticks([])

############################################################################################################
# (b)
############################################################################################################
#Axes_obj_02.set_title("Clérel & Caillard, Fig. 1. (Redraw & Annotated)", size=11.4230,  fontweight="normal")
#Axes_obj_02.set_xlabel("Year", fontweight="normal", size=10.0)
#Axes_obj_02.set_ylabel("No. of Pulmonary Thromboembolism", fontweight="normal", size=10.0)
#Axes_obj_02.set_xlim(1989, 1999)
#Axes_obj_02.set_ylim(0.0, 18)
#Axes_obj_02.set_xticks([1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998])
#Axes_obj_02.set_yticks([0, 2, 4, 6, 8, 10, 12, 14, 16])
#Axes_obj_02.yaxis.label.set_color("blue"),
#Axes_obj_02.tick_params(axis="y", colors="blue")

#Axes_obj_02.plot([1990-0.75, 1992+0.5], [2.00, 2.00], color='black', linestyle='solid', linewidth=1.6318, zorder=10)
#Axes_obj_02.plot([1993-0.5, 1994+0.5], [5.00, 5.00], color='black', linestyle='solid', linewidth=1.6318, zorder=10)
#Axes_obj_02.plot([1995-0.5, 1997+0.5], [11.5, 11.5], color='black', linestyle='solid', linewidth=1.6318, zorder=10)
#Axes_obj_02.plot([1998-0.5, 1998+0.75], [15.0, 15.0], color='black', linestyle='solid', linewidth=1.6318, zorder=10)

#Axes_obj_02.plot([1992+0.5, 1992+0.5], [2.00, 5.00], color='black', linestyle='solid', linewidth=1.6318, zorder=10)
#Axes_obj_02.plot([1994+0.5, 1994+0.5], [5.00, 11.5], color='black', linestyle='solid', linewidth=1.6318, zorder=10)
#Axes_obj_02.plot([1997+0.5, 1997+0.5], [11.5, 15.0], color='black', linestyle='solid', linewidth=1.6318, zorder=10)

#for i in range(0, 9):
#    Axes_obj_02.hlines(y=2.0*i, xmin=1989, xmax=1999, color="gray", linestyle="solid", linewidth=0.5, zorder=1)

#Axes_obj_02.bar(DataSet_Passengers["Year"], DataSet_Passengers["Thoromboembolism"], color='white', width=0.5, alpha=1.0, zorder=8)
#Axes_obj_02.bar(DataSet_Passengers["Year"], DataSet_Passengers["Thoromboembolism"], color='blue', width=0.5, alpha=0.3, zorder=9)
#Axes_obj_02.scatter(DataSet_Passengers["Year"], DataSet_Passengers["Thoromboembolism"], color='black', zorder=9)
#for i in range(0, len(DataSet_Passengers["Thoromboembolism"])):
#    Axes_obj_02.text(DataSet_Passengers["Year"][i], DataSet_Passengers["Thoromboembolism"][i]+0.5,
#                     DataSet_Passengers["Thoromboembolism"][i], size=10.0, color="black", zorder=9)

#Axes_obj_02_2 = Axes_obj_02.twinx()
#Axes_obj_02_2.set_ylabel("No. of Passengers ( x10^6)", fontweight="normal", size=10.0)
#Axes_obj_02_2.set_ylim(0.0, 35+(35/16)*2)
#Axes_obj_02_2.set_yticks([0, 5, 10, 15, 20, 25, 30, 35])
#Axes_obj_02_2.plot(DataSet_Passengers["Year"], DataSet_Passengers["Passengers2"], color='red', linestyle='solid', linewidth=1.6318, zorder=9)
#Axes_obj_02_2.scatter(DataSet_Passengers["Year"], DataSet_Passengers["Passengers2"], color='red', linestyle='solid', zorder=9)
#Axes_obj_02_2.yaxis.label.set_color("red"),
#Axes_obj_02_2.tick_params(axis="y", colors="red")


#LModel = clf.fit(DataSet_Passengers.loc[:, ["Year"]], DataSet_Passengers.loc[:, ["Passengers2"]])
#Coefficient = clf.coef_[0][0]
#Intercept = clf.intercept_[0]
#x1 = np.arange(1990-0.75, 1998+0.75, 1.0 * (10 ** (-3)))
#y1 = Coefficient*x1 + Intercept
#Axes_obj_02_2.plot(x1, y1, color="red", linestyle="solid", linewidth=1.0, zorder=9)

#for i in range(0, len(DataSet_Passengers["Thoromboembolism"])):
#    Axes_obj_02_2.text(DataSet_Passengers["Year"][i],
#                       DataSet_Passengers["Passengers2"][i] + 0.5,
#                       DataSet_Passengers["Passengers2"][i], size=10.0, color="red", ha='center', va='center', zorder=9)


#Axes_obj_02.annotate(
# text='\n'.join(["ACP", "Guideline"]),
#    xy=(1992.5, 6.75), xytext=(1992.5, 6.75+1.75), ha='center', va='center', size=10.0,
#    bbox=dict(boxstyle='round', edgecolor='blue', fc='white'),
#    arrowprops=dict(
#        facecolor='blue', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0, edgecolor='blue', shrink=0.1)
#)

#Axes_obj_02.annotate(
# text='\n'.join(["AHA", "Guideline"]),
#    xy=(1995, 13.5), xytext=(1995, 13.5+1.75), ha='center', va='center', size=10.0,
#    bbox=dict(boxstyle='round', edgecolor='blue', fc='white'),
#    arrowprops=dict(
#        facecolor='blue', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0, edgecolor='blue', shrink=0.1)
#)

#Axes_obj_02.annotate(
# text='\n'.join(["AHA", "NHS"]),
#    xy=(1997, 14.5), xytext=(1997, 14.5+1.75), ha='center', va='center', size=10.0,
#    bbox=dict(boxstyle='round', edgecolor='blue', fc='white'),
#    arrowprops=dict(
#        facecolor='blue', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0, edgecolor='blue', shrink=0.1)
#)

#Axes_obj_02.annotate(
# text='\n'.join(["HERS"]),
#    xy=(1998, 16), xytext=(1998, 16+1.25), ha='center', va='center', size=10.0,
#    bbox=dict(boxstyle='round', edgecolor='blue', fc='yellow'),
#    arrowprops=dict(
#        facecolor='blue', width=1.0, headwidth=7.5, headlength=10, linewidth=1.0, edgecolor='blue', shrink=0.1)
#)

############################################################################################################
# (c)
############################################################################################################
Axes_obj_03.set_title("History of Hormone Replacement Therapy (Recommendations to Concerns of Safety)",
                      size=11.4230, fontweight="normal")
Axes_obj_03.yaxis.set_visible(False)
Axes_obj_03.xaxis.set_visible(False)

Axes_obj_03.set_xlim(0.0, 1.0)
Axes_obj_03.set_ylim(0.0, 1.0)
Axes_obj_03.plot([0.01, 0.99], [0.6, 0.6], color='black', linestyle='solid', linewidth=1.0)
Axes_obj_03.quiver(0.04, 0.8, 0.0, -0.4, scale_units='xy', angles='xy', scale=1, linestyle='solid', linewidth=0.25)

Line_01 = "Timeline Table 1: Years 1992–1997 (Recommendations of Hormone Replacement Therapy, HRT)                   "
Line_02 = "----------------------------------------------------------------------------------------------------------"
Line_03 = " Year Month Type of Article  Journal (Citation)                     Recommendation                         "
Line_04 = "----------------------------------------------------------------------------------------------------------"
Line_05 = " 1992  Dec   ACP Guideline   Ann Intern Med. 1992;117(12):1038-41.  Consider for coronary disease         "
Line_06 = " 1995  Jul   AHA Guideline   J Am Coll Cardiol. 1995;26(1):292-4.   Mentioned risk reduction interventions"
Line_07 = " 1997  May   AHA Guideline   Circulation. 1997;95(9):2329-31.       Consider HRT in postmenopausal women  "
Line_08 = " 1997  Jun   NHS Cohort      N Engl J Med. 1997;336(25):1769-75.    Mortality: HRT user < non-user        "
Line_09 = "----------------------------------------------------------------------------------------------------------"
Line_10 = "                                                                                                          "
Line_11 = "Timeline Table 2: Years 1998–2002 (Expressing Concerns on the Safety of Hormone Replacement Therapy, HRT) "
Line_12 = "----------------------------------------------------------------------------------------------------------"
Line_13 = " Year  Month  Type of Article                                        Journal (Citation)                   "
Line_14 = "----------------------------------------------------------------------------------------------------------"
Line_15 = " 1998   Aug   HERS Randomized Clinical Trial                         AMA. 1998;280(7):605-13.             "
Line_16 = " 1999   May   AHA/ACC scientific statement                           J Am Coll Cardiol. 1999;33(6):1751-5."
Line_17 = " 2001   Jul   AHA Guideline                                              Circulation. 2001;104(4):499-503."
Line_18 = " 2002   Jul   HERS Randomized Clinical Trial Follow-up (HERS II)         AMA. 2002;288(1):58-66.          "
Line_19 = " 2002   Jul   Women's Health Initiative (WHI) Randomized Clinical Trial  JAMA. 2002;288(3):321-33.        "
Line_20 = "----------------------------------------------------------------------------------------------------------"
Line_21 = "                                                                                                          "
Line_22 = "Abbreviations:                                                                                            "
Line_23 = "ACP: American College of Physicians, AHA: American Heart Association, NHS: Nurses' Health Study           "
Line_24 = "HERS: Heart and Estrogen/progestin Replacement Study, ACC: American College of Cardiology                 "

#text01 = matplotlib.offsetbox.TextArea(Line_01, textprops=dict(fontname="monospace", color="black", fontsize=10.0, fontweight="normal"))
text01 = matplotlib.offsetbox.TextArea(Line_01, textprops=dict(fontname="monospace", color="blue", fontsize=10.0, fontweight="bold"))
text02 = matplotlib.offsetbox.TextArea(Line_02, textprops=dict(fontname="monospace", color="black", fontsize=10.0))
text03 = matplotlib.offsetbox.TextArea(Line_03, textprops=dict(fontname="monospace", color="black", fontsize=10.0, fontweight="bold"))
text04 = matplotlib.offsetbox.TextArea(Line_04, textprops=dict(fontname="monospace", color="black", fontsize=10.0))
text05 = matplotlib.offsetbox.TextArea(Line_05, textprops=dict(fontname="monospace", color="black", fontsize=10.0))
text06 = matplotlib.offsetbox.TextArea(Line_06, textprops=dict(fontname="monospace", color="black", fontsize=10.0))
text07 = matplotlib.offsetbox.TextArea(Line_07, textprops=dict(fontname="monospace", color="black", fontsize=10.0))
text08 = matplotlib.offsetbox.TextArea(Line_08, textprops=dict(fontname="monospace", color="black", fontsize=10.0))

text09 = matplotlib.offsetbox.TextArea(Line_09, textprops=dict(fontname="monospace", color="black", fontsize=10.0))

text10 = matplotlib.offsetbox.TextArea(Line_10, textprops=dict(fontname="monospace", color="black", fontsize=10.0))
text11 = matplotlib.offsetbox.TextArea(Line_11, textprops=dict(fontname="monospace", color="red", fontsize=10.0, fontweight="bold"))
text12 = matplotlib.offsetbox.TextArea(Line_12, textprops=dict(fontname="monospace", color="black", fontsize=10.0))
text13 = matplotlib.offsetbox.TextArea(Line_13, textprops=dict(fontname="monospace", color="black", fontsize=10.0, fontweight="bold"))
text14 = matplotlib.offsetbox.TextArea(Line_14, textprops=dict(fontname="monospace", color="black", fontsize=10.0))
text15 = matplotlib.offsetbox.TextArea(Line_15, textprops=dict(fontname="monospace", color="black", fontsize=10.0))
text16 = matplotlib.offsetbox.TextArea(Line_16, textprops=dict(fontname="monospace", color="black", fontsize=10.0))
text17 = matplotlib.offsetbox.TextArea(Line_17, textprops=dict(fontname="monospace", color="black", fontsize=10.0))
text18 = matplotlib.offsetbox.TextArea(Line_18, textprops=dict(fontname="monospace", color="black", fontsize=10.0))
text19 = matplotlib.offsetbox.TextArea(Line_19, textprops=dict(fontname="monospace", color="black", fontsize=10.0))
text20 = matplotlib.offsetbox.TextArea(Line_20, textprops=dict(fontname="monospace", color="black", fontsize=10.0))

text21 = matplotlib.offsetbox.TextArea(Line_21, textprops=dict(fontname="monospace", color="black", fontsize=10.0))

text22 = matplotlib.offsetbox.TextArea(Line_22, textprops=dict(fontname="monospace", color="black", fontsize=10.0, fontweight="bold"))
text23 = matplotlib.offsetbox.TextArea(Line_23, textprops=dict(fontname="monospace", color="black", fontsize=10.0))
text24 = matplotlib.offsetbox.TextArea(Line_24, textprops=dict(fontname="monospace", color="black", fontsize=10.0))

############################################################################################################
texts_vbox_1 = matplotlib.offsetbox.VPacker(children=[text01, text02, text03, text04, text05, text06, text07, text08],
                                            align="left", mode="fixed", pad=0.0, sep=0.1)
ann_1 = matplotlib.offsetbox.AnnotationBbox(texts_vbox_1, box_alignment=(0.5, 0.95), xycoords="data", xy=(0.5, 0.95),
                                          bboxprops=dict(boxstyle='round', edgecolor='black', fc='oldlace'),
                                          frameon=True)
Axes_obj_03.add_artist(ann_1)
############################################################################################################
texts_vbox_2 = matplotlib.offsetbox.VPacker(children=[text11, text12, text13, text14,
                                                      text15, text16, text17, text18, text19],
                                            align="left", mode="fixed", pad=0.0, sep=0.1)
ann_2 = matplotlib.offsetbox.AnnotationBbox(texts_vbox_2, box_alignment=(0.5, 0.3), xycoords="data", xy=(0.5, 0.3),
                                          bboxprops=dict(boxstyle='round', edgecolor='black', fc='oldlace'),
                                          frameon=True)
Axes_obj_03.add_artist(ann_2)
############################################################################################################
texts_vbox_3 = matplotlib.offsetbox.VPacker(children=[text22, text23, text24],
                                            align="left", mode="fixed", pad=0.0, sep=0.1)
ann_3 = matplotlib.offsetbox.AnnotationBbox(texts_vbox_3, box_alignment=(0.5, 0.025), xycoords="data", xy=(0.5, 0.025),
                                          bboxprops=dict(boxstyle='round', edgecolor='white', fc='white'),
                                          frameon=True)
Axes_obj_03.add_artist(ann_3)


########################################################################################################################
Figure_object.tight_layout()
my_path = os.path.abspath("Figures")
Figure_object.savefig(os.path.join(my_path, "01_Kimoto_et_al_(2023)_[Fig_S_05].png"))

from PIL import Image # py -m pip install pillow
img = Image.open(os.path.join(my_path, "01_Kimoto_et_al_(2023)_[Fig_S_05].png"))
img_resize = img.resize(size=(2866, 2016)) #size in pixels, as a 2-tuple: (width, height)
my_path2 = os.path.abspath("Figures/Fixed_Version/B6")
img_resize.save(os.path.join(my_path2, "01_Kimoto_et_al_(2023)_[Fig_S_05]_B6.png"))

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





