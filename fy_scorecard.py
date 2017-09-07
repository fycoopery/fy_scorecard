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
import statsmodels.api as sm


class fs_scorecard:
    def __init__(self,x,y,event=1,workpath = "./"):
        self.x = pd.DataFrame(x)
        self.y = pd.Series(y)
        self.event = event
        self.workpath = workpath
        self.max_woe = 10
        self.min_woe = -10
        self.columns_bin_dict = {}

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
        total_neg_count = total_count-total_pos_count
        pos_rt = (total_pos_count + 0.0)/total_count
        
        columns_bin_dict = self.columns_bin_dict
        woe_t = pd.DataFrame()
        for i in x.columns:
            nunique_i = x[i].nunique()
            type_i = str(x[i].dtype)
            
            if nunique_i <= 1:
                print i," : 0 or 1 level, no woe/iv calculated."
                columns_bin_dict[i] = []
                continue
            elif nunique_i <= 10:
                #value_counts
                xy_concat[i] = xy_concat[i].fillna("nan")
                df_temp = xy_concat.groupby(i)[y_column].agg([pd.Series.sum,pd.Series.count]).reset_index()\
                            .rename(columns = {"sum":"pos_count","count":"cat_total_count",i:"var_cat"})
                columns_bin_dict[i] = []
            else:
                if type_i == "object":
                    print i,": too many values for discrete variables."
                    continue
                else:
                    var_name = i
                    var_name_bin = str(i) + "_bin"
                    if i not in columns_bin_dict.keys():
                        list_q2 = list(xy_concat[i].dropna().quantile(list_q).unique()) + [np.inf]
                        list_q2[0] = -np.inf
                        if len(list_q2) == 2:
                            list_q2 = list(pd.Series(xy_concat[i].dropna().unique()) \
                                .quantile(list_q)) + [np.inf]
                            list_q2[0] = -np.inf
                    else: 
                        list_q2 = columns_bin_dict[i]
                    xy_concat[var_name_bin] = pd.cut(x[var_name],list_q2).astype("string")
                    df_temp = xy_concat.groupby(var_name_bin)[y_column].agg([pd.Series.sum,pd.Series.count]).reset_index()\
                            .rename(columns = {"sum":"pos_count","count":"cat_total_count",var_name_bin:"var_cat"})
                            
                    columns_bin_dict[i] = list_q2
                    
            df_temp["var_name"] = i
            df_temp["neg_count"] = df_temp["cat_total_count"] - df_temp["pos_count"]
            df_temp["p_ni"] = df_temp["neg_count"] / total_neg_count
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
        woe_t["var_cat"]=woe_t["var_cat"].astype("string")
        iv_t = woe_t.groupby("var_name")["iv_i"].sum().reset_index().rename(columns = { "iv_i" :"iv"})
        filePath = self.workpath+"woe_t.xlsx"
        excel_writer = ExcelWriter(filePath)
        woe_t.to_excel(excel_writer,sheet_name="woe",index =False)
        iv_t.to_excel(excel_writer,sheet_name="iv",index =False)
        excel_writer.save()

        self.woe_t = woe_t
        self.iv_t = iv_t
        print filePath," generated;\n <name>.woe_t, <name>.iv_t available"


    def get_woe_replaced_df(self):
        x = self.x
        woe_t = self.woe_t
        df_binned = pd.DataFrame()
        df_woe_replaced = pd.DataFrame()
        columns_bin_dict = self.columns_bin_dict
        for i in columns_bin_dict.keys():
            if len(columns_bin_dict[i])>0:
                df_binned[i] = pd.cut(x[i],columns_bin_dict[i]).astype("string")
            else:
                df_binned[i] = x[i].astype("string")
        for i in df_binned.columns:
            df_woe_value = woe_t[woe_t["var_name"]==i][["var_cat","woe"]]
            df_woe_replaced[i] = df_binned.merge(df_woe_value,how="left",left_on=i,right_on="var_cat")["woe"]
        
        self.df_woe_replaced = df_woe_replaced
        self.df_binned = df_binned
        print "<name>.df_woe_replaced, <name>.df_binned available"

    def genmodel(self, excluded_columns = []):
        x = self.df_woe_replaced.drop(excluded_columns,axis = 1)
        y = self.y.copy()
        y[y==self.event] = 1
        y[y<>self.event] = 0
        self.model = sm.Logit(endog=y , exog=x)
        self.result = self.model.fit()
        print self.result.summary()
        

print "FY Scorecard ready!"