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


    def get_woe_replaced_df(self , input_x = None):
        if input_x is None:
            x = self.x
        else:
            x = input_x

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
        if input_x is None:
            print "<name>.df_woe_replaced, <name>.df_binned available"
        else:
            return df_binned,df_woe_replaced


    def gen_model(self, iv_lower_bound = 0.02,iv_upper_bound = 20,excluded_columns = []):
        columns_iv = self.iv_t[(self.iv_t["iv"]>=iv_lower_bound) & (self.iv_t["iv"]<=iv_upper_bound)]["var_name"]
        x = self.df_woe_replaced[columns_iv].drop(excluded_columns,axis = 1,errors = "ignore").reset_index(drop = True)
        y = self.y.copy().reset_index(drop = True)
        y[y==self.event] = 1
        y[y<>self.event] = 0
        self.y_event = y
        self.model = sm.Logit(endog=y , exog=x)
        self.model_result = self.model.fit()
        print self.model_result.summary()

    def gen_score(self, input_x = None, score_base = 600, odds_change_rt = 20):
        base_score = score_base
        base_rt = odds_change_rt
        
        y_event = self.y_event
        base_odds = (y_event.sum()+0.0)/(y_event.count()-y_event.sum())
        reverse_base_odds = 1/base_odds
        
        p = base_rt/np.log(2)
        q = base_score - base_rt*np.log(reverse_base_odds)/np.log(2)
        
        print "base_odds: ",base_odds
        print "reverse_base_odds: ",reverse_base_odds
        print "base_rt: ",base_rt
        print "base_score: ",base_score
        print "p: ",p
        print "q: ",q

        #生成df_scored
        
        params = self.model_result.params
        df_params = params.reset_index().rename(columns={"index":"var_name",0:"params"})
        
        self.woe_t_scored = self.woe_t.merge(df_params,on="var_name",how="left")
        woe_t2 = self.woe_t_scored
        woe_t2["score"] = - woe_t2["params"] * woe_t2["woe"]*p
        
        #woe_t2 为得分概览
        df_scored = pd.DataFrame()

        if input_x is not None:
            df_binned,df_woe_replaced = self.get_woe_replaced_df(input_x)
        else:
            df_binned = self.df_binned

        for i in self.model.exog_names:
        	print i 
        	df_score_value = woe_t2[woe_t2["var_name"]==i][["var_cat","score"]]
        	df_scored[i] = df_binned.merge(df_score_value,how="left",left_on=i,right_on="var_cat")["score"]

        df_scored["final_score"] = df_scored.sum(axis = 1) + base_score
        print df_scored["final_score"].describe()
        if input_x is not None:
            return df_scored
        else:
            self.df_scored = df_scored
            print "<name>.df_scored, <name>.woe_t_scored available"

        


print "FY Scorecard ready!"