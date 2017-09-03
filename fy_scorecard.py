# -*- coding: utf-8 -*-
"""
Spyder Editor
created: 2017-09-01
After RNG losed the LPL Final
fy_scorecard.
"""
import pandas as pd
import numpy as np
from pandas import ExcelWriter

class fs_scorecard:
    def __init__(self,x,y,event=1,workpath = "./"):
        self.x = x
        self.y = y
        self.event = event
        self.workpath = workpath
        self.max_woe = 10
        self.min_woe = -10

    def get_woe_iv(self,bins = 5):
        bin_interval = 1.0/bins
        x = self.x
        y = self.y
        event = self.event
        y_column = y.name
        
        xy_concat = pd.concat([x,y],axis = 1).copy()
        list_q = np.arange(0,1,bin_interval)
        
        xy_concat.loc[xy_concat[y_column]==event,y_column] = 1
        xy_concat.loc[xy_concat[y_column]<>event,y_column] = 0
        total_count = xy_concat.shape[0]
        total_pos_count = xy_concat[y_column].sum()
        pos_rt = (total_pos_count + 0.0)/total_count
        
        columns_bin_dict = {}
        woe_t = pd.DataFrame()
        for i in x.columns:
            nunique_i = x[i].nunique()
            type_i = str(x[i].dtype)
            
            if nunique_i <= 1:
                print i," : 0 or 1 level, no woe/iv calculated."
                continue
            elif nunique_i <= 10:
                #value_counts
                xy_concat[i] = xy_concat[i].fillna(-9999)
                df_temp = xy_concat.groupby(i)[y_column].agg([pd.Series.sum,pd.Series.count]).reset_index()\
                            .rename(columns = {"sum":"pos_count","count":"cat_total_count",i:"var_cat"})                    
                df_temp["var_name"] = i
                df_temp["neg_count"] = df_temp["cat_total_count"] - df_temp["pos_count"]
                df_temp["p_ni"] = df_temp["cat_total_count"] / total_count
                df_temp["p_yi"] = df_temp["pos_count"] / total_pos_count
                df_temp["woe"] = np.log(df_temp["pos_count"]/total_pos_count / \
                                df_temp["p_ni"])
                woe_t = woe_t.append(df_temp)
                continue
            else:
                if type_i == "object":
                    print i,": too many values for discrete variables."
                else:
                    var_name = i
                    var_name_bin = str(i) + "_bin"
                    list_q2 = list(xy_concat[i].dropna().quantile(list_q).unique()) + [np.inf]
                    list_q2[0] = -np.inf
                    if len(list_q2) == 2:
                        list_q2 = list(pd.Series(xy_concat[i].dropna().unique()) \
                            .quantile(list_q)) + [np.inf]
                        list_q2[0] = -np.inf
                        xy_concat[var_name_bin] = pd.cut(xy_concat[i], list_q2).astype("string")
                    else: 
                        xy_concat[var_name_bin] = pd.cut(x[var_name],list_q2).astype("string")
                    df_temp = xy_concat.groupby(var_name_bin)[y_column].agg([pd.Series.sum,pd.Series.count]).reset_index()\
                            .rename(columns = {"sum":"pos_count","count":"cat_total_count",var_name_bin:"var_cat"})
                    df_temp["var_name"] = i
                    df_temp["neg_count"] = df_temp["cat_total_count"] - df_temp["pos_count"]
                    df_temp["p_ni"] = df_temp["cat_total_count"] / total_count
                    df_temp["p_yi"] = df_temp["pos_count"] / total_pos_count
                    df_temp["woe"] = np.log(df_temp["pos_count"]/total_pos_count / \
                                    df_temp["p_ni"])
                    woe_t = woe_t.append(df_temp)
        woe_t.loc[woe_t["woe"] >= self.max_woe,"woe"] = self.max_woe
        woe_t.loc[woe_t["woe"] <= self.min_woe,"woe"] = self.min_woe
        woe_t["iv_i"] = (woe_t["p_yi"] - woe_t["p_ni"])*woe_t["woe"]
        woe_t["p_y_total"] = pos_rt
        woe_show_cols = ['var_name','var_cat','cat_total_count',
                         'pos_count','neg_count','p_ni',	"p_y_total",
                         'p_yi','woe','iv_i']
        woe_t = woe_t[woe_show_cols]
        iv_t = woe_t.groupby("var_name")["iv_i"].sum().rename(columns = {"iv_i":"iv"}).reset_index()
        filePath = self.workpath+"woe_t.xlsx"
        excel_writer = ExcelWriter("woe_t.xlsx")
        woe_t.to_excel(excel_writer,sheet_name="woe",index =False)
        iv_t.to_excel(excel_writer,sheet_name="iv",index =False)
        excel_writer.save()
        self.woe_t = woe_t
        self.iv_t = iv_t
        print filePath," generated;\n <name>.woe_t, <name>.iv_t available"


print "FY Scorecard ready!"