# Copyright 2026 Get Credit Rate Foundation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
#
# Builds the scorecard from woe_spec.pkl and produces Appendices A through D (bin tables, scorecard, PSI, reason codes). Published under the file name cited in the Model Validation Report, section 8; working name was build4.py. Logic unchanged.
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
SPEC=pickle.load(open('woe_spec.pkl','rb'))
d=pd.read_parquet('matured.parquet')
d=d[(d.vintage>=2012)&(d.vintage<=2017)].copy()
tr=d[d.vintage<=2015].copy()
te=d[d.vintage>=2016].copy()
FICO=['fico']
DEEP=['num_accts_ever_120_pd','num_tl_90g_dpd_24m','mths_since_last_delinq','mths_since_last_major_derog','mths_since_last_record','delinq_2yrs','pct_tl_nvr_dlq','num_rev_accts','total_acc','cr_hist_yrs','mo_sin_old_rev_tl_op','mo_sin_old_il_acct','num_sats','num_bc_tl','num_il_tl','num_op_rev_tl','num_rev_tl_bal_gt_0','num_actv_rev_tl','num_actv_bc_tl','num_bc_sats']
VB=[c for c in SPEC if c not in FICO and c not in DEEP]
def binlab(df,c):
    k,a,m,iv=SPEC[c]
    if k=='n':
        return pd.cut(pd.to_numeric(df[c],errors='coerce').astype(float),a).astype(str)
    s=df[c].fillna('MISS').astype(str)
    return s.where(s.isin(a),'OTHER')
LTR={c:binlab(tr,c) for c in VB}
LTE={c:binlab(te,c) for c in VB}
Xtr=pd.DataFrame({c:LTR[c].map(SPEC[c][2]).astype(float).fillna(0.0) for c in VB},index=tr.index)
Xte=pd.DataFrame({c:LTE[c].map(SPEC[c][2]).astype(float).fillna(0.0) for c in VB},index=te.index)
mod=LogisticRegression(max_iter=1000).fit(Xtr,tr['default'])
pte=mod.predict_proba(Xte)[:,1]
ptr=mod.predict_proba(Xtr)[:,1]
AUC=roc_auc_score(te['default'],pte)
beta=pd.Series(mod.coef_[0],index=VB)
alpha=float(mod.intercept_[0])
NC=len(VB)
PDO=20.0
FACT=PDO/np.log(2.0)
BASE=600.0
BODDS=20.0
OFF=BASE-FACT*np.log(BODDS)
PTS={}
for c in VB:
    m=SPEC[c][2]
    PTS[c]={b:float(-(w*beta[c]+alpha/NC)*FACT+OFF/NC) for b,w in m.items()}
def sc(L):
    t=None
    for c in VB:
        v=L[c].map(PTS[c]).astype(float)
        v=v.fillna(float(np.mean(list(PTS[c].values()))))
        t=v if t is None else t+v
    return t
Str=sc(LTR)
Ste=sc(LTE)
pickle.dump({'beta':beta,'alpha':alpha,'PTS':PTS,'VB':VB,'FACT':FACT,'OFF':OFF},open('scorecard_model.pkl','wb'))
B=[]
B.append('APPENDIX A - WEIGHT OF EVIDENCE BIN TABLES')
B.append('Bins fitted on TRAIN vintages 2012-2015 only. Applied unchanged to test.')
B.append('WoE = ln(P(non-default in bin)/P(default in bin)). Higher WoE = lower risk.')
B.append('Points use PDO=20, base 600 at 20:1 odds. Higher points = lower risk.')
B.append('')
ord_iv=sorted(VB,key=lambda c:-SPEC[c][3])
for c in ord_iv:
    k,a,m,iv=SPEC[c]
    g=pd.DataFrame({'b':LTR[c].values,'y':tr['default'].values}).groupby('b')['y'].agg(['size','sum'])
    B.append(c+'   IV='+str(round(iv,4))+'   beta='+str(round(float(beta[c]),4)))
    B.append('  '+'bin'.ljust(30)+'n'.rjust(9)+'dflt%'.rjust(9)+'WoE'.rjust(10)+'points'.rjust(9))
    for b in g.index:
        if b not in m: continue
        nn=int(g.loc[b,'size']); bad=int(g.loc[b,'sum'])
        B.append('  '+str(b)[:30].ljust(30)+str(nn).rjust(9)+str(round(100.0*bad/nn,2)).rjust(9)+str(round(float(m[b]),4)).rjust(10)+str(round(PTS[c][b],1)).rjust(9))
    B.append('')
open('APPENDIX_A_WOE_BINS.txt','w').write(chr(10).join(B))
print('APPENDIX A written, features='+str(len(VB)))

C=[]
C.append('APPENDIX B - SCORECARD SUMMARY AND COEFFICIENTS')
C.append('')
C.append('Model: logistic regression on Weight of Evidence transformed characteristics')
C.append('Characteristics: '+str(NC))
C.append('Intercept: '+str(round(alpha,6)))
C.append('Scaling: PDO=20, base score 600 at 20:1 good:bad odds, factor='+str(round(FACT,4))+', offset='+str(round(OFF,4)))
C.append('')
C.append('  '+'characteristic'.ljust(32)+'IV'.rjust(9)+'beta'.rjust(10)+'pts_min'.rjust(9)+'pts_max'.rjust(9)+'range'.rjust(8))
for c in ord_iv:
    lo=min(PTS[c].values()); hi=max(PTS[c].values())
    C.append('  '+c.ljust(32)+str(round(SPEC[c][3],4)).rjust(9)+str(round(float(beta[c]),4)).rjust(10)+str(round(lo,1)).rjust(9)+str(round(hi,1)).rjust(9)+str(round(hi-lo,1)).rjust(8))
C.append('')
C.append('SCORE DISTRIBUTION')
C.append('  train  min='+str(round(float(Str.min()),1))+'  p5='+str(round(float(Str.quantile(.05)),1))+'  median='+str(round(float(Str.median()),1))+'  p95='+str(round(float(Str.quantile(.95)),1))+'  max='+str(round(float(Str.max()),1)))
C.append('  test   min='+str(round(float(Ste.min()),1))+'  p5='+str(round(float(Ste.quantile(.05)),1))+'  median='+str(round(float(Ste.median()),1))+'  p95='+str(round(float(Ste.quantile(.95)),1))+'  max='+str(round(float(Ste.max()),1)))
C.append('')
C.append('DEFAULT RATE BY SCORE BAND (test, out of time)')
bands=[0,540,560,580,600,620,640,9999]
bl=pd.cut(Ste,bands)
gb=pd.DataFrame({'b':bl.astype(str),'y':te['default'].values}).groupby('b')['y'].agg(['size','mean'])
for b in gb.index:
    C.append('  '+str(b).ljust(20)+'n='+str(int(gb.loc[b,'size'])).rjust(7)+'  default='+str(round(100*float(gb.loc[b,'mean']),2)).rjust(6)+'%')
open('APPENDIX_B_SCORECARD.txt','w').write(chr(10).join(C))
print('APPENDIX B written')
D=[]
D.append('APPENDIX C - POPULATION STABILITY INDEX')
D.append('')
D.append('PSI = sum over bands of (actual pct - expected pct) * ln(actual pct / expected pct)')
D.append('Expected = TRAIN score distribution (2012-2015). Bands are train score deciles.')
D.append('Interpretation: below 0.10 stable, 0.10 to 0.25 moderate shift, above 0.25 significant.')
D.append('')
q=np.unique(np.quantile(Str,np.linspace(0,1,11)))
q[0]=-1e18; q[-1]=1e18
e=pd.cut(Str,q).value_counts(normalize=True).sort_index()
def psi_of(s):
    a=pd.cut(s,q).value_counts(normalize=True).sort_index()
    a=a.reindex(e.index).fillna(1e-6)
    return float((((a-e)*np.log(a/e)).sum())),a
tot,a_all=psi_of(Ste)
D.append('  '+'score band'.ljust(28)+'expected%'.rjust(11)+'actual%'.rjust(10)+'contrib'.rjust(10))
for b in e.index:
    ct=float((a_all[b]-e[b])*np.log(a_all[b]/e[b]))
    D.append('  '+str(b)[:28].ljust(28)+str(round(100*float(e[b]),2)).rjust(11)+str(round(100*float(a_all[b]),2)).rjust(10)+str(round(ct,5)).rjust(10))
D.append('')
D.append('  TOTAL PSI train vs test (2016-2017): '+str(round(tot,4)))
D.append('')
D.append('PSI BY VINTAGE (expected = train)')
for v in sorted(d['vintage'].unique()):
    idx=d['vintage']==v
    sub=d[idx]
    L={c:binlab(sub,c) for c in VB}
    sv=sc(L)
    pv,_=psi_of(sv)
    D.append('  '+str(int(v))+'  n='+str(int(idx.sum())).rjust(7)+'  PSI='+str(round(pv,4)))
open('APPENDIX_C_PSI.txt','w').write(chr(10).join(D))
print('APPENDIX C written, PSI='+str(round(tot,4)))

LAB={'term':'Length of loan term requested','acc_open_past_24mths':'Number of accounts opened in the last 24 months','dti':'Ratio of debt obligations to income','num_tl_op_past_12m':'Number of accounts opened in the last 12 months','bc_open_to_buy':'Available credit on revolving bankcard accounts','verification_status':'Extent of income verification','avg_cur_bal':'Average balance carried across accounts','mo_sin_rcnt_tl':'Time since most recent account opened','tot_hi_cred_lim':'Total credit limit extended across accounts','total_bc_limit':'Total bankcard credit limit','loan_amnt':'Amount of credit requested','tot_cur_bal':'Total current balance across accounts','annual_inc':'Income reported on application','mo_sin_rcnt_rev_tl_op':'Time since most recent revolving account opened','revol_util':'Proportion of revolving credit in use','revol_bal':'Balance carried on revolving accounts','inq_last_6mths':'Number of credit inquiries in the last 6 months','bc_util':'Proportion of bankcard credit in use','open_acc':'Number of open accounts','mort_acc':'Number of mortgage accounts','percent_bc_gt_75':'Proportion of bankcards used above 75 percent of limit','total_bal_ex_mort':'Total non-mortgage balance','home_ownership':'Housing status reported on application','purpose':'Stated purpose of the loan','emp_length':'Length of employment reported','total_rev_hi_lim':'Total revolving credit limit','mths_since_recent_bc':'Time since most recent bankcard opened','mths_since_recent_inq':'Time since most recent credit inquiry','pub_rec':'Public records on file','pub_rec_bankruptcies':'Bankruptcy filings on file','tax_liens':'Tax liens on file','application_type':'Individual or joint application','total_il_high_credit_limit':'Total installment credit limit','acc_now_delinq':'Accounts currently delinquent','tot_coll_amt':'Amounts in collection','collections_12_mths_ex_med':'Collections in the last 12 months','chargeoff_within_12_mths':'Charge-offs in the last 12 months','delinq_amnt':'Amount currently delinquent','num_tl_30dpd':'Accounts 30 days past due'}
MX={c:max(PTS[c].values()) for c in VB}
E=[]
E.append('APPENDIX D - ADVERSE ACTION REASON CODES ON REAL RECORDS')
E.append('')
E.append('Regulation B section 1002.9 requires a statement of the specific principal reasons')
E.append('for adverse action. Reasons below are the characteristics on which the applicant')
E.append('lost the most scorecard points relative to the highest scoring bin for that')
E.append('characteristic. Records are drawn from the held-out 2016-2017 test vintages.')
E.append('')
lo=np.argsort(np.asarray(Ste.values))[:5]
for r,i in enumerate(lo):
    ix=Ste.index[i]
    defs=[]
    for c in VB:
        b=LTE[c].loc[ix]
        pv=PTS[c].get(b,float(np.mean(list(PTS[c].values()))))
        defs.append((MX[c]-pv,c,b,pv))
    defs.sort(reverse=True)
    E.append('  APPLICANT '+str(r+1)+'   total score '+str(round(float(Ste.loc[ix]),1))+'   modelled default probability '+str(round(100*float(pte[i]),2))+'%')
    E.append('    realized outcome in data: '+('DEFAULT' if int(te['default'].loc[ix])==1 else 'repaid in full'))
    E.append('    principal reasons:')
    for k in range(4):
        dd,c,b,pv=defs[k]
        E.append('      '+str(k+1)+'. '+LAB.get(c,c)+'  [bin '+str(b)[:24]+', '+str(round(dd,1))+' points below best]')
    E.append('')
E.append('REASON CODE FREQUENCY across the lowest scoring 5000 test records')
lo2=np.argsort(np.asarray(Ste.values))[:5000]
cnt={}
for i in lo2:
    ix=Ste.index[i]
    defs=[]
    for c in VB:
        b=LTE[c].loc[ix]
        pv=PTS[c].get(b,float(np.mean(list(PTS[c].values()))))
        defs.append((MX[c]-pv,c))
    defs.sort(reverse=True)
    for dd,c in defs[:4]:
        cnt[c]=cnt.get(c,0)+1
for c,v in sorted(cnt.items(),key=lambda x:-x[1])[:12]:
    E.append('  '+LAB.get(c,c).ljust(50)+str(v).rjust(6)+'  ('+str(round(100.0*v/5000,1))+'%)')
open('APPENDIX_D_REASON_CODES.txt','w').write(chr(10).join(E))
print('APPENDIX D written')
F=[]
F.append('STATED LIMITATIONS')
F.append('  1 No cash-flow, rental, utility or telecom variables exist in this source. Alternative-data')
F.append('    feature validation is deferred to the retrospective phase.')
F.append('  2 This source contains no race, ethnicity or sex fields, so disparate impact could not be')
F.append('    tested at this phase. Fair lending testing across protected classes is assigned to the')
F.append('    retrospective phase against a partner institution borrower record set.')
F.append('  3 2018 vintages excluded: term length exceeds the observation window, so resolved 2018')
F.append('    loans are biased toward early payoff and early default.')
F.append('  4 2016-2017 test vintages are not fully matured for 60-month terms.')
F.append('  5 Calibration: train era default rate 18.64 percent against test era 23.23 percent.')
F.append('    Rank ordering transfers out of time. The intercept requires recalibration on any')
F.append('    deployment population before use in a live decision.')
F.append('  6 These applicants are not a CDFI applicant population. Coefficients must be refit on a')
F.append('    partner institution portfolio before any live lending decision.')
F.append('  7 This population is credit visible. The reduced feature set removes the bureau score and')
F.append('    deep tradeline history to approximate a thin file applicant. It does not replicate one.')
F.append('')
F.append('REPRODUCTION')
F.append('  extract.py    builds matured.parquet from the public source file')
F.append('  scorecard.py  fits Weight of Evidence bins and both model variants')
F.append('  build4.py     produces Appendices A through D')
F.append('  Source: LendingClub accepted loan file 2007 through 2018Q4, 2,260,701 rows.')
open('LIMITATIONS_AND_REPRO.txt','w').write(chr(10).join(F))
print('LIMITATIONS written')
print()
print('VARIANT B REFIT CHECK  AUC='+str(round(AUC,4))+'  chars='+str(NC))
