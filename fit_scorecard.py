# Copyright 2026 Get Credit Rate Foundation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
#
# Fits Weight of Evidence bins on training vintages and estimates Variants A and B; writes the validation report text and woe_spec.pkl. Published under the file name cited in the Model Validation Report, section 8; working name was scorecard.py. Logic unchanged.
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
d=pd.read_parquet('matured.parquet')
d=d[(d.vintage>=2012)&(d.vintage<=2017)].copy()
tr=d[d.vintage<=2015].copy()
te=d[d.vintage>=2016].copy()
DROP=['loan_status','default','vintage','grade','sub_grade','int_rate']
ALL=[c for c in d.columns if c not in DROP]
NUM=[c for c in ALL if pd.api.types.is_numeric_dtype(d[c])]
CAT=[c for c in ALL if c not in NUM]
FICO=['fico']
DEEP=['num_accts_ever_120_pd','num_tl_90g_dpd_24m','mths_since_last_delinq','mths_since_last_major_derog','mths_since_last_record','delinq_2yrs','pct_tl_nvr_dlq','num_rev_accts','total_acc','cr_hist_yrs','mo_sin_old_rev_tl_op','mo_sin_old_il_acct','num_sats','num_bc_tl','num_il_tl','num_op_rev_tl','num_rev_tl_bal_gt_0','num_actv_rev_tl','num_actv_bc_tl','num_bc_sats']
MINB=1000
def qb(s,nb):
    v=pd.to_numeric(s,errors='coerce').astype(float)
    q=np.unique(np.nanquantile(v,np.linspace(0,1,nb+1)))
    if len(q)<3: return None
    q=q.astype(float); q[0]=-1e18; q[-1]=1e18
    return q
def woe(bs,y,minb):
    g=pd.DataFrame({'b':bs,'y':np.asarray(y)}).groupby('b')['y'].agg(['size','sum'])
    g=g[g['size']>=minb]
    if len(g)<2: return None,0.0
    B=g['sum']+0.5; G=g['size']-g['sum']+0.5
    pb=B/B.sum(); pg=G/G.sum(); w=np.log(pg/pb)
    return w.to_dict(),float(((pg-pb)*w).sum())
SPEC={}
for c in NUM:
    q=qb(tr[c],8)
    if q is None: continue
    bs=pd.cut(pd.to_numeric(tr[c],errors='coerce').astype(float),q).astype(str)
    m,iv=woe(bs,tr['default'],MINB)
    if m is None: continue
    SPEC[c]=('n',q,m,iv)
for c in CAT:
    s=tr[c].fillna('MISS').astype(str)
    vc=s.value_counts(); keep=set(vc[vc>=MINB].index)
    bs=s.where(s.isin(keep),'OTHER')
    m,iv=woe(bs,tr['default'],MINB)
    if m is None: continue
    SPEC[c]=('c',keep,m,iv)
def tx(df,cols):
    X=pd.DataFrame(index=df.index)
    for c in cols:
        k,a,m,iv=SPEC[c]
        if k=='n':
            bs=pd.cut(pd.to_numeric(df[c],errors='coerce').astype(float),a).astype(str)
        else:
            s=df[c].fillna('MISS').astype(str); bs=s.where(s.isin(a),'OTHER')
        X[c]=bs.map(m).astype(float).fillna(0.0)
    return X
def ks(y,p):
    o=np.argsort(-np.asarray(p)); yy=np.asarray(y)[o]
    cb=np.cumsum(yy)/max(yy.sum(),1); cg=np.cumsum(1-yy)/max((1-yy).sum(),1)
    return float(np.max(np.abs(cb-cg)))
VA=[c for c in SPEC]
VB=[c for c in SPEC if c not in FICO and c not in DEEP]
OUT=[]
OUT.append('='*74)
OUT.append('GCR FRAMEWORK - PHASE 1 MODEL VALIDATION REPORT')
OUT.append('WoE + logistic regression scorecard | LendingClub 2007-2018Q4 accepted loans')
OUT.append('='*74)
OUT.append('')
OUT.append('POPULATION')
OUT.append('  source rows                2,260,701')
OUT.append('  matured (resolved) loans   1,345,350')
OUT.append('  modelling window           vintages 2012-2017')
OUT.append('  TRAIN  vintages 2012-2015  n='+str(len(tr))+'  default='+str(round(100*tr['default'].mean(),2))+'%')
OUT.append('  TEST   vintages 2016-2017  n='+str(len(te))+'  default='+str(round(100*te['default'].mean(),2))+'%')
OUT.append('  split is OUT-OF-TIME (later vintages held out entirely)')
OUT.append('')
OUT.append('LEAKAGE CONTROL - excluded from all feature sets')
OUT.append('  post-origination payment, recovery, settlement, hardship and last_fico fields')
OUT.append('  LendingClub grade, sub_grade and int_rate (incumbent assessment, used only as benchmark)')
OUT.append('')
OUT.append('FEATURES BINNED: '+str(len(SPEC)))
OUT.append('')
OUT.append('TOP 15 BY INFORMATION VALUE')
for c,(k,a,m,iv) in sorted(SPEC.items(),key=lambda x:-x[1][3])[:15]:
    OUT.append('  '+c.ljust(32)+'IV='+str(round(iv,4)))
OUT.append('')
RES={}
for tag,cols,desc in [('A',VA,'FULL-FILE (includes FICO and deep bureau history)'),('B',VB,'THIN-FILE PROXY (FICO and deep bureau history REMOVED)')]:
    Xtr=tx(tr,cols); Xte=tx(te,cols)
    m=LogisticRegression(max_iter=1000,C=1.0).fit(Xtr,tr['default'])
    p=m.predict_proba(Xte)[:,1]
    a=roc_auc_score(te['default'],p); k=ks(te['default'],p)
    RES[tag]=(a,k,len(cols),m,cols,p)
    OUT.append('-'*74)
    OUT.append('VARIANT '+tag+' - '+desc)
    OUT.append('  predictors        '+str(len(cols)))
    OUT.append('  out-of-time AUC   '+str(round(a,4)))
    OUT.append('  Gini              '+str(round(2*a-1,4)))
    OUT.append('  KS                '+str(round(k,4)))
sg=te['sub_grade'].astype(str)
ord_map={v:i for i,v in enumerate(sorted(sg.unique()))}
bench=sg.map(ord_map).astype(float)
ab=roc_auc_score(te['default'],bench)
OUT.append('-'*74)
OUT.append('INCUMBENT BENCHMARK - LendingClub sub_grade, same test set')
OUT.append('  out-of-time AUC   '+str(round(ab,4)))
OUT.append('  Gini              '+str(round(2*ab-1,4)))
OUT.append('')
OUT.append('COMPARISON')
OUT.append('  Variant A vs incumbent  '+str(round(RES['A'][0]-ab,4))+' AUC')
OUT.append('  Variant B vs incumbent  '+str(round(RES['B'][0]-ab,4))+' AUC')
OUT.append('  A minus B (value of bureau score + deep history) '+str(round(RES['A'][0]-RES['B'][0],4))+' AUC')
OUT.append('')
OUT.append('CALIBRATION - VARIANT B, out-of-time deciles')
pb=RES['B'][5]
df=pd.DataFrame({'p':pb,'y':te['default'].values})
df['dec']=pd.qcut(df['p'],10,labels=False,duplicates='drop')
g=df.groupby('dec').agg(n=('y','size'),pred=('p','mean'),obs=('y','mean'))
for i,r in g.iterrows():
    OUT.append('  decile '+str(int(i)+1).rjust(2)+'  n='+str(int(r['n'])).rjust(6)+'  predicted='+str(round(100*r['pred'],2)).rjust(6)+'%  observed='+str(round(100*r['obs'],2)).rjust(6)+'%')
OUT.append('')
OUT.append('STATED LIMITATIONS')
OUT.append('  1 No cash-flow, rental, utility or telecom variables exist in this source.')
OUT.append('    Alternative-data feature validation is deferred to the retrospective phase.')
OUT.append('  2 2018 vintages excluded: term length exceeds observation window, so resolved')
OUT.append('    2018 loans are biased toward early payoff and early default.')
OUT.append('  3 2016-2017 test vintages are not fully matured for 60-month terms.')
OUT.append('  4 LendingClub applicants are not a CDFI applicant population. Coefficients')
OUT.append('    must be refit on a partner institution portfolio before any live decision.')
OUT.append('  5 This population is credit-visible. Variant B removes the bureau score and')
OUT.append('    deep history to approximate a thin-file applicant; it does not replicate one.')
r=chr(10).join(OUT)
open('VALIDATION_REPORT.txt','w').write(r)
pickle.dump(SPEC,open('woe_spec.pkl','wb'))
print(r)
