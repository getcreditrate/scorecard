# Copyright 2026 Get Credit Rate Foundation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
#
# Builds the matured (resolved) loan set from the public LendingClub accepted-loan release. Published under the file name cited in the Model Validation Report, section 8; working name was extract.py. Logic unchanged; the input path is now relative.
import pandas as pd
import numpy as np
P='data/accepted_2007_to_2018Q4.csv'  # place the public release here; not redistributed
ORIG=['loan_amnt','term','annual_inc','dti','delinq_2yrs','earliest_cr_line','fico_range_low','fico_range_high','inq_last_6mths','mths_since_last_delinq','mths_since_last_record','open_acc','pub_rec','revol_bal','revol_util','total_acc','home_ownership','verification_status','purpose','emp_length','application_type','acc_now_delinq','tot_coll_amt','tot_cur_bal','total_rev_hi_lim','acc_open_past_24mths','avg_cur_bal','bc_open_to_buy','bc_util','chargeoff_within_12_mths','delinq_amnt','mo_sin_old_il_acct','mo_sin_old_rev_tl_op','mo_sin_rcnt_rev_tl_op','mo_sin_rcnt_tl','mort_acc','mths_since_recent_bc','mths_since_recent_inq','num_accts_ever_120_pd','num_actv_bc_tl','num_actv_rev_tl','num_bc_sats','num_bc_tl','num_il_tl','num_op_rev_tl','num_rev_accts','num_rev_tl_bal_gt_0','num_sats','num_tl_30dpd','num_tl_90g_dpd_24m','num_tl_op_past_12m','pct_tl_nvr_dlq','percent_bc_gt_75','pub_rec_bankruptcies','tax_liens','tot_hi_cred_lim','total_bal_ex_mort','total_bc_limit','total_il_high_credit_limit','collections_12_mths_ex_med','mths_since_last_major_derog']
KEEP=ORIG+['loan_status','issue_d','grade','sub_grade','int_rate']
MAT=['Fully Paid','Charged Off','Default']
out=[]
n=0
for ch in pd.read_csv(P,usecols=KEEP,chunksize=200000,low_memory=False):
    n+=len(ch)
    ch=ch[ch['loan_status'].isin(MAT)].copy()
    if len(ch)==0: continue
    ch['default']=(ch['loan_status']!='Fully Paid').astype('int8')
    ch['vintage']=pd.to_datetime(ch['issue_d'],format='%b-%Y',errors='coerce').dt.year
    ch['cr_hist_yrs']=ch['vintage']-pd.to_datetime(ch['earliest_cr_line'],format='%b-%Y',errors='coerce').dt.year
    ch['fico']=(ch['fico_range_low']+ch['fico_range_high'])/2.0
    ch=ch.drop(columns=['earliest_cr_line','fico_range_low','fico_range_high','issue_d'])
    out.append(ch)
    print('scanned',n,'kept',sum(len(x) for x in out),flush=True)
d=pd.concat(out,ignore_index=True)
d=d[d['vintage'].notna()]
d['vintage']=d['vintage'].astype(int)
for c in d.select_dtypes(include=['float64']).columns: d[c]=d[c].astype('float32')
d.to_parquet('matured.parquet',index=False)
print()
print('FINAL SHAPE',d.shape)
print('DEFAULT RATE',round(100*d['default'].mean(),2))
print()
print('DEFAULT RATE BY VINTAGE:')
g=d.groupby('vintage')['default'].agg(['size','mean'])
for v,r in g.iterrows(): print('  '+str(v)+'  n='+str(int(r['size'])).rjust(7)+'  dflt='+str(round(100*r['mean'],2))+'%')
