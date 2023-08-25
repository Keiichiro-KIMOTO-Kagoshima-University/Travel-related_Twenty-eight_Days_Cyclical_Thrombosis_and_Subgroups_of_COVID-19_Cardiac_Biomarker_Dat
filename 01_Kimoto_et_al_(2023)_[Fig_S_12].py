########################################################################################################################
# Fig. S12. Data cut-off dates in each study cited by Matsushita et al.
# (01_Kimoto_et_al_(2023)_[Fig_S_12].py)
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
import datetime

from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN
############################
import os
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option("display.width", 300)
import matplotlib.ticker as ticker
################################
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
print("")
print("###############################################################################################################")
print("### References #")
print("###############################################################################################################")

Guan_W_et_al = [1, 14,  "21", "I", "No.08", "Guan Y et al. ", "NEJM",
               "Non-severe, 45 [34-57]", "Severe, 52 [40-65]",
               "30 provinces", "China", "2019-12-11", "2020-01-31",
               "<No.08> Guan W et al. 30 provinces China", "normal", "red", "red", "left", "center",
                "(Severe, 52 [40-65] vs Non-severe, 45 [34-57])"]

Zhou_F_et_al_1 = [2, 12, "37", "I", "No.25", "Zhou F et al.", "Lancet",
               "", "",
               "Jinyintan Hosp.",
               "Dongxihu", "2019-12-29", "2020-01-31",
                "<No.25> Zhou F et al. Jinyintan Hosp. Dongxihu", "normal", "red", "red", "left", "center",
                  ""]

Zhou_F_et_al_2 = [3, 11, "37", "I", "No.25", "Zhou F et al.", "Lancet",
               "Survivor, 52 [45-58]", "Non-survivor, 69 [63-76]",
               "Pulmonary Hosp.",
               "Hongshan", "2019-12-29", "2020-01-31",
                "                      Pulmonary Hosp. Hongshan", "normal", "red", "red", "left", "center",
                "(Non-survivor, 69 [63-76] v.s. Survivor, 52 [45-58])"]

Wang_D_et_al = [4, 9, "32", "I", "No.19", "Wang D et al.", "JAMA",
               "Non-ICU, 51 [37-62]", "ICU, 66 [57-78]",
               "Zhongnan Hosp.", "Wuchang", "2020-01-01", "2020-02-03",
               "<No.19> Wang D et al. Zhongnan Hosp. Wuchang", "normal", "red", "red", "left", "center",
                "(ICU, 66 [57-78] vs Non-ICU, 51 [37-62])"]

Wu_C_et_al = [5, 7,  "34", "I", "No.21", "Wu C et al.", "JAMA Intern. Med",
             "Non-ARDS,48 [40-54]", "ARDS, 58.5 [50-69]",
             "Jinyintan Hosp.", "Dongxihu", "2019-12-25", "2020-02-13",
             "<No.21> Wu C et al.   Jinyintan Hosp. Dongxihu", "normal", "red", "red", "left", "center",
              "(ARDS, 58.5 [50-69] vs Non-ARDS,48 [40-54])"]

Cao_J_et_al_1 = [6, 5, "15", "II", "No.02", "Cao J et al.", "Intensive Care Med",
              "Non-severe, 31 [35-62]", "Severe, 66 [54-76]",
              "Zhongnan Hosp.",
              "Wuchang", "2020-01-03", "2020-02-15",
               "<No.02> Cao J et al.  Zhongnan Hosp. Wuchang", "normal", "blue", "blue", "left", "center",
                "(ICU, 66 [54-76] vs Non-ICU, 31 [35-62])"]

Deng_Y_et_al_1 = [7, 3, "20", "II", "No.07", "Deng Y et al.", "Chin. Med. J.",
                 "", "",
                 "Tongji Hosp. Caidian Br.",
                 "Caidian", "2020-01-01", "2020-02-21",
                 "<No.07> Deng Y et al. Tongji Hosp. Caidian Br. Caidian", "normal", "blue", "blue", "right", "center",
                  ""]

Deng_Y_et_al_2 = [8, 2, "20", "II", "No.07", "Deng Y et al.", "Chin. Med. J.",
                 "", "",
                 "Tongji Hosp. Hankou Br.",
                 "Qiaokou", "2020-01-01", "2020-02-21",
                 "                      Tongji Hosp.  Hankou Br. Qiaokou", "normal", "blue", "blue", "right", "center",
                  ""]

Deng_Y_et_al_3 = [9, 1, "20", "II", "No.07", "Deng Y et al.", "Chin. Med. J.",
                 "Recovered, 40 [33-57]", "Death, 69 [62-74]",
                 "Central Hosp. Hankou Br.",
                 "Qiaokou", "2020-01-01", "2020-02-21",
                 "                      Central Hosp. Hankou Br. Qiaokou", "normal", "blue", "blue", "right", "center",
                  "(Death, 69 [62-74] vs. Recovered, 40 [33-57])"]

#Yuan_M_et_al = [1, 19, "35", "I", "No.23", "Yuan M et al.", "PLOS ONE",
#               "Survival, 55 [35-60]", "Mortality, 68 [63-73]",
#               "Central Hosp.", "Jiang'an", "2020-01-01", "2020-01-25",
#               "<No.23> Yuan M et al. Central Hosp. Jiang'an", "normal", "red", "red", "left", "center"]

#Wang_L_et_al = [7, 8, "33", "I", "No.20", "Wang L. et al.", "J. Infect.",
#               "Survival, 68 [64-74]", "Dead, 76 [70-83]",
#               "Renmin Hosp.", "Wuchang", "2020-01-01", "2020-02-06",
#               "<No.20> Wang L. et al. Renmin Hosp. Wuchang", "normal", "red", "red", "left", "center"]

#Cao_J_et_al_1 = [8, 6, "15", "II", "No.02", "Cao J et al.", "Intensive Care Med",
#              "Non-severe, 31 [35-62]", "Severe, 66 [54-76]",
#              "Jinyintan Hosp.",
#              "Dongxihu", "2020-01-01", "2020-01-20",
#               "<No.02> Cao J et al. Jinyintan Hosp. Dongxihu (Chen N et al.)", "normal", "blue", "blue", "left", "center"]

#Cao_J_et_al_2 = [9, 5, "15", "II", "No.02", "Cao J et al.", "Intensive Care Med",
#              "Non-severe, 31 [35-62]", "Severe, 66 [54-76]",
#              "Zhongnan Hosp.",
#              "Wuchang", "2020-01-01", "2020-01-28",
#               "                     Zhongnan Hosp.  Wuchang (Wang D et al.)", "normal", "blue", "purple", "left", "center"]

Reference = pd.DataFrame(
    [Guan_W_et_al, Wu_C_et_al, Wang_D_et_al, Zhou_F_et_al_1, Zhou_F_et_al_2,
     Cao_J_et_al_1, Deng_Y_et_al_1, Deng_Y_et_al_2, Deng_Y_et_al_3],
    index=["Guan_W_et_al", "Wu_C_et_al", "Wang_D_et_al", "Zhou_F_et_al_1", "Zhou_F_et_al_2",
           "Cao_J_et_al_1", "Deng_Y_et_al_1", "Deng_Y_et_al_2", "Deng_Y_et_al_3"],
    columns=["Sq", "Sq2", "Ref.", "Group", "No.", "Author", "Jounal", "Non-severe Group",
             "Severe Group", "Medical Facility", "District", "Start", "End", "Label", "font_weight",
             "color1", "color2", "horizon", "vertical", "vs"])

Reference["Start2"] = pd.to_datetime(Reference["Start"], format='%Y/%m/%d')
Reference["End2"] = pd.to_datetime(Reference["End"], format='%Y/%m/%d')
#Reference.dtypes
print(Reference)

Duration = Reference.loc[:, ["Sq", "Sq2", "Label", "Start2", "End2", "font_weight", "color1", "color2",
                             "horizon", "vertical", "Severe Group", "Non-severe Group", "vs"]]
print(Duration)


Figure_object = plt.figure(1, figsize=(11.69, 8.27), dpi=400, edgecolor="black", linewidth=0.5)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)
gs_master = matplotlib.gridspec.GridSpec(nrows=1, ncols=1)
Axes_obj_01 = Figure_object.add_subplot(gs_master[0])

for i in range(1, len(Duration)+1):
    x1 = Duration[Duration["Sq"] == i].iat[0, Duration.columns.get_loc("Start2")]
    x2 = Duration[Duration["Sq"] == i].iat[0, Duration.columns.get_loc("End2")]
    y1 = Duration[Duration["Sq"] == i].iat[0, Duration.columns.get_loc("Sq2")]
    y2 = Duration[Duration["Sq"] == i].iat[0, Duration.columns.get_loc("Sq2")]
    label = Duration[Duration["Sq"] == i].iat[0, Duration.columns.get_loc("Label")]
    Start_label = Duration[Duration["Sq"] == i].iat[0, Duration.columns.get_loc("Start2")]
    End_label = Duration[Duration["Sq"] == i].iat[0, Duration.columns.get_loc("End2")]
    color1 = Duration[Duration["Sq"] == i].iat[0, Duration.columns.get_loc("color1")]
    color2 = Duration[Duration["Sq"] == i].iat[0, Duration.columns.get_loc("color2")]
    font_weight = Duration[Duration["Sq"] == i].iat[0, Duration.columns.get_loc("font_weight")]
    horizon = Duration[Duration["Sq"] == i].iat[0, Duration.columns.get_loc("horizon")]
    vertical = Duration[Duration["Sq"] == i].iat[0, Duration.columns.get_loc("vertical")]


    vs = Duration[Duration["Sq"] == i].iat[0, Duration.columns.get_loc("vs")]



    Axes_obj_01.set_title("Comparison of Durations on Each Studies", size=11.4230, fontweight="normal")
    Axes_obj_01.set_xlabel("Day")
    Axes_obj_01.set_ylabel("")
    Axes_obj_01.tick_params(length=0.0);
    Axes_obj_01.set_yticklabels([])
    Axes_obj_01.set_xlim(datetime.datetime.strptime("2019-10-01", '%Y-%m-%d'),
                         datetime.datetime.strptime("2020-03-01", '%Y-%m-%d'))
    Axes_obj_01.set_ylim(0, 15)

    Axes_obj_01.plot([x1, x2], [y1, y2], color=color2, linestyle="solid", linewidth=15, zorder=10)


    Axes_obj_01.text(datetime.datetime.strptime("2019-10-02", '%Y-%m-%d'), y2, label,size=11.4320,
                     fontweight="normal", color=color1, fontname="monospace")
    Axes_obj_01.text(datetime.datetime.strptime("2019-10-30", '%Y-%m-%d'), y2-0.75, vs, size=10.0,
                     fontweight="normal", color="black", fontname="monospace")


    Axes_obj_01.text(x1, y1+0.35, datetime.datetime.strftime(Start_label, '%Y-%m-%d'), color="black", zorder=9,
                     bbox={"facecolor": "white", "edgecolor": "white", "boxstyle": "round", "linewidth": 0.5})
    Axes_obj_01.text(x2, y2+0.35, datetime.datetime.strftime(End_label, '%Y-%m-%d'), color="black", zorder=9,
                     ha=horizon,
                     bbox={"facecolor": "white", "edgecolor": "white", "boxstyle": "round", "linewidth": 0.5})


#Axes_obj_01.plot([datetime.datetime.strptime("2019-10-03", '%Y-%m-%d'),
#                  datetime.datetime.strptime("2020-02-27", '%Y-%m-%d')],
#                 [15, 15], color="gray", linestyle="solid", linewidth=0.5)
#Axes_obj_01.plot([datetime.datetime.strptime("2019-10-03", '%Y-%m-%d'),
#                  datetime.datetime.strptime("2020-02-27", '%Y-%m-%d')],
#                 [14, 14], color="gray", linestyle="solid", linewidth=0.5)
Axes_obj_01.plot([datetime.datetime.strptime("2019-10-03", '%Y-%m-%d'),
                  datetime.datetime.strptime("2020-02-27", '%Y-%m-%d')],
                 [13, 13], color="gray", linestyle="solid", linewidth=0.5)
Axes_obj_01.plot([datetime.datetime.strptime("2019-10-03", '%Y-%m-%d'),
                  datetime.datetime.strptime("2020-02-27", '%Y-%m-%d')],
                 [10, 10], color="gray", linestyle="solid", linewidth=0.5)
Axes_obj_01.plot([datetime.datetime.strptime("2019-10-03", '%Y-%m-%d'),
                  datetime.datetime.strptime("2020-02-27", '%Y-%m-%d')],
                 [8, 8], color="gray", linestyle="solid", linewidth=0.5)
Axes_obj_01.plot([datetime.datetime.strptime("2019-10-01", '%Y-%m-%d'),
                  datetime.datetime.strptime("2020-03-01", '%Y-%m-%d')],
                 [6, 6], color="gray", linestyle="solid", linewidth=1.5)
Axes_obj_01.plot([datetime.datetime.strptime("2019-10-03", '%Y-%m-%d'),
                  datetime.datetime.strptime("2020-02-27", '%Y-%m-%d')],
                 [4, 4], color="gray", linestyle="solid", linewidth=0.5)

Axes_obj_01.plot([datetime.datetime.strptime("2019-12-31", '%Y-%m-%d'),
                  datetime.datetime.strptime("2019-12-31", '%Y-%m-%d')],
                 [0, 20], color="red", linestyle="dashed", linewidth=1.6318)

Axes_obj_01.plot([datetime.datetime.strptime("2020-02-01", '%Y-%m-%d'),
                  datetime.datetime.strptime("2020-02-01", '%Y-%m-%d')],
                 [0, 20], color="red", linestyle="dashed", linewidth=1.6318)

#Axes_obj_01.plot([datetime.datetime.strptime("2019-12-31", '%Y-%m-%d'),
#                  datetime.datetime.strptime("2019-12-31", '%Y-%m-%d')],
#                 [5, 15], color="black", linestyle="dashed", linewidth=0.5)
#Axes_obj_01.plot([datetime.datetime.strptime("2020-01-29", '%Y-%m-%d'),
#                  datetime.datetime.strptime("2020-01-29", '%Y-%m-%d')],
#                 [5, 15], color="black", linestyle="dashed", linewidth=0.5)

############################################################################################################

Figure_object.tight_layout()
my_path = os.path.abspath("Figures")
plt.savefig(os.path.join(my_path, "01_Kimoto_et_al_(2023)_[Fig_S_12].png"))

from PIL import Image # py -m pip install pillow
img = Image.open(os.path.join(my_path, "01_Kimoto_et_al_(2023)_[Fig_S_12].png"))
img_resize = img.resize(size=(2866, 2016))
my_path2 = os.path.abspath("Figures/Fixed_Version/B6")
img_resize.save(os.path.join(my_path2, "01_Kimoto_et_al_(2023)_[Fig_S_12]_B6.png"))

#plt.show()
########################################################################################################################


