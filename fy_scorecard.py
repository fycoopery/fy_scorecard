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
import matplotlib.pyplot as plt
import math as mt
import sklearn.metrics as skmetric
import matplotlib.pyplot as plt

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
        woe_t["odds"] = woe_t["pos_count"]/woe_t["neg_count"]
        woe_t.loc[woe_t["woe"] >= self.max_woe,"woe"] = self.max_woe
        woe_t.loc[woe_t["woe"] <= self.min_woe,"woe"] = self.min_woe
        woe_t["iv_i"] = (woe_t["p_yi"] - woe_t["p_ni"])*woe_t["woe"]
        woe_t["p_y_total"] = pos_rt
        woe_show_cols = ['var_name','var_cat','cat_total_count',
                         'pos_count','neg_count','p_ni',"p_y_total","odds",
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
        if input_x is None:
            self.df_woe_replaced = df_woe_replaced
            self.df_binned = df_binned
            print "<name>.df_woe_replaced, <name>.df_binned available"
        else:
            return df_binned,df_woe_replaced


    def gen_woe_iv_plot(self):
        woe_t = self.woe_t
        iv_t = self.iv_t

        var_cnt = iv_t.shape[0]
        self.woe_iv_plot,axes = plt.subplots(nrows = var_cnt,ncols = 1)
        self.woe_iv_plot.set_size_inches(15,10 * var_cnt)
        ax_cnt = 0

        for i in iv_t.sort_values(by = "iv" ,ascending=False)["var_name"]:
            plt.subplot(var_cnt , 1 , ax_cnt+1)
            woe_t_sample = woe_t[woe_t["var_name"]==i]
            iv_value = iv_t[iv_t["var_name"]==i]["iv"].round(3).iloc[0]

            ind = np.arange(woe_t_sample.shape[0])    # the x locations for the groups
            width = 0.35       # the width of the bars: can also be len(x) sequence
            
            p1 = plt.bar(ind, woe_t_sample["pos_count"], width, color='#d62728')
            p2 = plt.bar(ind, woe_t_sample["neg_count"], width, bottom=woe_t_sample["pos_count"])
            plt.twinx()
            p3 = plt.plot(ind,woe_t_sample["pos_count"]/woe_t_sample["cat_total_count"],'o-',color = "darkorange")
            plt.ylabel('Obs cnt')
            plt.title("Var - %s, iv = %.3f" % (i,iv_value))
            xticks = woe_t_sample["var_cat"]+" \n woe: " + woe_t_sample["woe"].round(2).astype("string")
            plt.xticks(ind, xticks)
            plt.legend((p1[0], p2[0],p3[0]), ('pos_count', 'neg_count','pos_rate'))
            re = plt.setp(axes[ax_cnt].get_xticklabels(),rotation = 30 ,horizontalalignment = "right")
            ax_cnt = ax_cnt+1

            
    def gen_model(self, iv_lower_bound = 0.02,iv_upper_bound = 20,excluded_columns = []):
        columns_iv = self.iv_t[(self.iv_t["iv"]>=iv_lower_bound) & (self.iv_t["iv"]<=iv_upper_bound)]["var_name"]
        x = self.df_woe_replaced[columns_iv].drop(excluded_columns,axis = 1,errors = "ignore").reset_index(drop = True).copy()
        x["__intercept"] = 1
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
            if i<>"__intercept":
                df_scored[i] = df_binned.merge(df_score_value,how="left",left_on=i,right_on="var_cat")["score"]

        df_scored["final_score"] = df_scored.sum(axis = 1) + base_score
        print df_scored["final_score"].describe()
        if input_x is not None:
            return df_scored
        else:
            self.df_scored = df_scored
            print "<name>.df_scored, <name>.woe_t_scored available"

    def model_evaluate(self,test_x = None,test_y = None):
    
        if test_x is not None:
            df_binned_test,df_woe_replaced_test = self.get_woe_replaced_df(test_x)
            model_columns = self.model.exog_names
            df_woe_replaced_test["__intercept"] = 1
            y_test_predict = self.model_result.predict(df_woe_replaced_test[model_columns])
            y = self.y_event
            predict_true_test = pd.DataFrame()
            predict_true_test["true"] = test_y.reset_index(drop= True)
            predict_true_test["predict"] = y_test_predict.reset_index(drop= True)
            fpr_test ,tpr_test ,thresholds_test = skmetric.roc_curve(predict_true_test["true"],predict_true_test["predict"])
            auc_test =  skmetric.auc(fpr_test, tpr_test)
            print 'ROC_TEST-(AUC = %0.2f)' % auc_test
        
        predict_true = pd.DataFrame()
        predict_true["true"] = self.y_event
        predict_true["predict"] = self.model_result.fittedvalues
        fpr ,tpr ,thresholds = skmetric.roc_curve(predict_true["true"],predict_true["predict"])
        auc =  skmetric.auc(fpr, tpr)  
        print 'ROC-(AUC = %0.2f)' % auc

        #roc_curve
        
        self.roc_plot = plt.figure()
        self.roc_plot.set_size_inches(15,10)
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve')
        if test_x is not None:
            plt.plot(fpr_test, tpr_test, color='darkblue',
                 lw=lw, label='Test ROC curve')
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic-(AUC = %0.2f)' % auc)
        plt.legend(loc="lower right")
        
        #ks_curve
        
        event_total = self.y_event.sum()
        non_event_total = self.y_event.count() - self.y_event.sum()
        
        predict_true["score_rk"] = predict_true["predict"].rank(ascending = False
                                                        ,method = "first")
        group_size = round(predict_true.shape[0]/20)
        predict_true["rk_group"] = (predict_true["score_rk"]/group_size).apply(np.ceil)

        test_grp = predict_true.groupby("rk_group")["true"].agg([np.sum,pd.Series.count])\
                    .rename(columns = {"sum":"event_cnt","count":"total_cnt"})\
                    .reset_index()
        test_grp["non_event_cnt"] = test_grp["total_cnt"] - test_grp["event_cnt"]
        test_grp["event_cnt_cum"] = test_grp["event_cnt"].cumsum()
        test_grp["non_event_cnt_cum"] = test_grp["non_event_cnt"].cumsum()

        test_grp["event_rt"] = test_grp["event_cnt_cum"] / event_total
        test_grp["none_event_rt"] = test_grp["non_event_cnt_cum"] / non_event_total

        test_grp["ks_value"] = test_grp["event_rt"] - test_grp["none_event_rt"]

        ks_value = test_grp["ks_value"].max()
        ks_position = test_grp[test_grp["ks_value"]==ks_value]["rk_group"].iloc[0]
        print "ks_value: %0.2f" % ks_value
        print "ks_position",ks_position

        test_grp_2 = pd.DataFrame({
            "rk_group":[0],
            "event_rt":[0],
            "none_event_rt":[0]}
        ).append(test_grp)

        self.ks_plot = plt.figure()
        self.ks_plot.set_size_inches(15,10)
        plt.plot(test_grp_2["rk_group"],test_grp_2["event_rt"],color = "darkorange")
        plt.plot(test_grp_2["rk_group"],test_grp_2["none_event_rt"],color = "navy")
        plt.plot([ks_position,ks_position],[0,1],color="red",linestyle = "--",
                label = "KS_position")
        plt.xlim([0,20])
        plt.ylim([0,1.05])
        plt.xlabel("grp")
        plt.ylabel("rt")
        plt.title("KS Curve - ks value %0.2f" % ks_value)
        plt.legend(loc = "lower right")

        
        #lift chart
        y = self.y_event
        df_scored= self.df_scored.copy()
        df_scored["score_rank"] = (df_scored["final_score"].rank(pct=True,ascending = True)*10).apply(mt.ceil)
        df_test = pd.concat([df_scored["score_rank"],y.reset_index(drop=True),df_scored["final_score"]],axis = 1)
        df_test_grp = df_test.groupby("score_rank")[y.name].agg([pd.Series.sum,pd.Series.count])
        df_test_grp["pos_rt"] = df_test_grp["sum"] / df_test_grp["count"]
        df_test_grp_2 = df_test.groupby("score_rank")["final_score"].agg([pd.Series.max,pd.Series.min])
        df_test_grp_all = df_test_grp.merge(df_test_grp_2,left_index=True,
                                            right_index=True).rename(columns = {
            "sum":"pos_cnt",
            "count": "total_cnt",
            "max": "score_max",
            "min": "score_min"
        })
        df_test_grp_all["cum_pos_cnt"] = df_test_grp_all["pos_cnt"].cumsum()
        df_test_grp_all["cum_total_cnt"] = df_test_grp_all["total_cnt"].cumsum()
        df_test_grp_all["cum_pos_rt"] = df_test_grp_all["cum_pos_cnt"]/df_test_grp_all["total_cnt"].cumsum()
        total_pos_rt = (df_test_grp_all["pos_cnt"].sum()+0.0)/df_test_grp_all["total_cnt"].sum()

        df_test_grp_all["decile_lift"] = df_test_grp_all["pos_rt"]/total_pos_rt
        df_test_grp_all["cum_lift"] = df_test_grp_all["cum_pos_rt"]/total_pos_rt

        fig,ax = plt.subplots()
        plt.subplot(1,1,1)
        fig.set_size_inches(15,10)
        x_tick = df_test_grp_all.index
        plt.plot(x_tick,df_test_grp_all["cum_lift"])
        plt.plot(x_tick,df_test_grp_all["decile_lift"])
        plt.plot(x_tick,[1]*df_test_grp_all.shape[0],label = "base_line")

        for x,y in zip(x_tick,df_test_grp_all["cum_lift"]):
            plt.text(x,y,"%.2f" % y)

        for x,y in zip(x_tick,df_test_grp_all["decile_lift"]):
            plt.text(x,y,"%.2f" % y)
            
        plt.title("Lift Chart")
        plt.xticks(range(1,11),size='small')

        plt.xlabel("Lift")
        plt.ylabel("Decile")
        self.lift_chart = fig
        self.lift_t = df_test_grp_all
        
        #Score Distribution
        self.score_dist_chart, ax = plt.subplots(2,1)
        self.score_dist_chart.set_size_inches(15,20)

        plt.subplot(2,1,1)
        plt.title("Score Distribution")
        h1 = plt.hist(self.df_scored[self.y_event == 0 ]["final_score"]
                ,histtype="stepfilled", bins=50, alpha=0.5, label = "neg")
        plt.legend(loc = "upper left")
        ax2 = plt.twinx()
        h2 = plt.hist(self.df_scored[self.y_event == 1 ]["final_score"]
                ,histtype="stepfilled", color = "r", bins=50, alpha=0.5, label = "pos")

        plt.xlabel("Score")
        plt.xlabel("Obs cnt")
        ax2.legend(loc = "upper right")

        plt.subplot(2,1,2)
        plt.title("Score Distribution - All")
        plt.hist(self.df_scored["final_score"]
                ,histtype="stepfilled", bins=50, alpha=0.5, label = "total")
                
        print "<name>.roc_plot/ks_plot/lift_chart/score_dist_chart/lift_t available"


print "FY Scorecard ready!"