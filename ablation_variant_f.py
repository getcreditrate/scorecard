# Copyright 2026 Get Credit Rate Foundation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
#
# Reproduces Variant D and estimates Variant F (debt to income ratio removed). Addendum to the Model Validation Report dated 18 September 2026.
import pandas as pd, numpy as np, json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
d=pd.read_parquet('matured.parquet')
d=d[(d.vintage>=2012)&(d.vintage<=2017)].copy()
tr=d[d.vintage<=2015].copy(); te=d[d.vintage>=2016].copy()
MINB=1000
def qb(s,nb):
    v=pd.to_numeric(s,errors='coerce').astype(float); q=np.unique(np.nanquantile(v,np.linspace(0,1,nb+1)))
    if len(q)<3: return None
    q=q.astype(float); q[0]=-1e18; q[-1]=1e18; return q
def woe(bs,y,minb):
    g=pd.DataFrame({'b':bs,'y':np.asarray(y)}).groupby('b')['y'].agg(['size','sum']); g=g[g['size']>=minb]
    if len(g)<2: return None,0.0
    B=g['sum']+0.5; G=g['size']-g['sum']+0.5; pb=B/B.sum(); pg=G/G.sum(); w=np.log(pg/pb)
    return w.to_dict(),float(((pg-pb)*w).sum())
D8=['annual_inc','dti','emp_length','home_ownership','loan_amnt','purpose','term','verification_status']
SPEC={}
for c in D8:
    if pd.api.types.is_numeric_dtype(d[c]):
        q=qb(tr[c],8); bs=pd.cut(pd.to_numeric(tr[c],errors='coerce').astype(float),q).astype(str); m,iv=woe(bs,tr['default'],MINB); SPEC[c]=('n',q,m,iv)
    else:
        s=tr[c].fillna('MISS').astype(str); vc=s.value_counts(); keep=set(vc[vc>=MINB].index); bs=s.where(s.isin(keep),'OTHER'); m,iv=woe(bs,tr['default'],MINB); SPEC[c]=('c',keep,m,iv)
def tx(df,cols):
    X=pd.DataFrame(index=df.index)
    for c in cols:
        k,a,m,iv=SPEC[c]
        if k=='n': bs=pd.cut(pd.to_numeric(df[c],errors='coerce').astype(float),a).astype(str)
        else: s=df[c].fillna('MISS').astype(str); bs=s.where(s.isin(a),'OTHER')
        X[c]=bs.map(m).astype(float).fillna(0.0)
    return X
sg=te['sub_grade'].astype(str); om={v:i for i,v in enumerate(sorted(sg.unique()))}; ab=roc_auc_score(te['default'],sg.map(om).astype(float)); bg=2*ab-1
out={'train_n':int(len(tr)),'test_n':int(len(te)),'loans_total':int(len(d)),'defaults':int(d['default'].sum()),'bench_auc':round(ab,4),'bench_gini':round(bg,4)}
for tag,cols in [('D',D8),('F',[c for c in D8 if c!='dti'])]:
    m=LogisticRegression(max_iter=1000,C=1.0).fit(tx(tr,cols),tr['default']); p=m.predict_proba(tx(te,cols))[:,1]
    a=roc_auc_score(te['default'],p); g=2*a-1
    out[tag]={'k':len(cols),'auc':round(a,4),'gini':round(g,4),'pct_bench_gini':round(100*g/bg,1)}
out['dti_IV']=round(SPEC['dti'][3],4)
print(json.dumps(out,indent=1))
json.dump(out,open('variantF_results.json','w'),indent=1)

