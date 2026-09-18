# Copyright 2026 Get Credit Rate Foundation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
#
# Fits and compares feature-set Variants B, C, D and E against the platform benchmark. Published under the file name cited in the Model Validation Report, section 8; working name was termtest.py. Logic unchanged.
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
SPEC=pickle.load(open('woe_spec.pkl','rb'))
d=pd.read_parquet('matured.parquet')
print('ALL MATURED',len(d),'defaults',int(d["default"].sum()))
w=d[(d.vintage>=2012)&(d.vintage<=2017)].copy()
print('WINDOW 2012-2017',len(w),'defaults',int(w["default"].sum()))
print('EXCLUDED',len(d)-len(w),'defaults',int(d["default"].sum())-int(w["default"].sum()))
tr=w[w.vintage<=2015].copy()
te=w[w.vintage>=2016].copy()
print('train',len(tr),'test',len(te),'sum',len(tr)+len(te))
FICO=['fico']
DEEP=['num_accts_ever_120_pd','num_tl_90g_dpd_24m','mths_since_last_delinq','mths_since_last_major_derog','mths_since_last_record','delinq_2yrs','pct_tl_nvr_dlq','num_rev_accts','total_acc','cr_hist_yrs','mo_sin_old_rev_tl_op','mo_sin_old_il_acct','num_sats','num_bc_tl','num_il_tl','num_op_rev_tl','num_rev_tl_bal_gt_0','num_actv_rev_tl','num_actv_bc_tl','num_bc_sats']
BUREAU=['bc_open_to_buy','total_bc_limit','revol_util','revol_bal','mort_acc','inq_last_6mths','pub_rec','pub_rec_bankruptcies','tax_liens','tot_coll_amt','percent_bc_gt_75','mths_since_recent_bc','mths_since_recent_inq','bc_util','open_acc','tot_cur_bal','tot_hi_cred_lim','total_bal_ex_mort','total_rev_hi_lim','total_il_high_credit_limit','avg_cur_bal','acc_open_past_24mths','num_tl_op_past_12m','mo_sin_rcnt_tl','mo_sin_rcnt_rev_tl_op','acc_now_delinq','collections_12_mths_ex_med','chargeoff_within_12_mths','delinq_amnt','num_tl_30dpd']
VB=[c for c in SPEC if c not in FICO and c not in DEEP]
VC=[c for c in VB if c!='term']
VD=[c for c in VB if c not in BUREAU]
VE=[c for c in VD if c!='term']
def bl(df,c):
    k,a,m,iv=SPEC[c]
    if k=='n': return pd.cut(pd.to_numeric(df[c],errors='coerce').astype(float),a).astype(str)
    s=df[c].fillna('MISS').astype(str)
    return s.where(s.isin(a),'OTHER')
def fit(cols,tag):
    X=pd.DataFrame({c:bl(tr,c).map(SPEC[c][2]).astype(float).fillna(0.0) for c in cols},index=tr.index)
    Y=pd.DataFrame({c:bl(te,c).map(SPEC[c][2]).astype(float).fillna(0.0) for c in cols},index=te.index)
    m=LogisticRegression(max_iter=1000).fit(X,tr['default'])
    a=roc_auc_score(te['default'],m.predict_proba(Y)[:,1])
    print(tag.ljust(52)+'k='+str(len(cols)).rjust(3)+'  AUC='+str(round(a,4)))
    return a
sg=te['sub_grade'].astype(str)
om={v:i for i,v in enumerate(sorted(sg.unique()))}
ab=roc_auc_score(te['default'],sg.map(om).astype(float))
print()
print('INCUMBENT LendingClub sub_grade'.ljust(52)+'      AUC='+str(round(ab,4)))
b=fit(VB,'B  no FICO, no deep history')
c=fit(VC,'C  = B minus term')
dd=fit(VD,'D  = B minus all bureau fields')
e=fit(VE,'E  = D minus term')
print()
print('B - incumbent  '+str(round(b-ab,4)))
print('C - incumbent  '+str(round(c-ab,4))+'   PARITY HOLDS' if c>=ab else 'C - incumbent  '+str(round(c-ab,4))+'   PARITY FAILS')
print('D - incumbent  '+str(round(dd-ab,4)))
print('E - incumbent  '+str(round(e-ab,4)))
print()
print('VD features ('+str(len(VD))+'): '+str(sorted(VD)))
