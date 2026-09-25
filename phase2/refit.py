# Copyright 2026 Get Credit Rate Foundation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
#
# Phase 2 retrospective refit. Refits the application-level scorecard (Variant D feature set)
# on a partner institution's resolved loan book under the Retrospective Validation Protocol:
# out-of-time split by origination vintage, WoE bins fitted on the estimation split only,
# discrimination (AUC, Gini, KS) and calibration by decile on the holdout, PSI between splits.
# Usage: python refit.py <parquet or csv path> [--train-through YEAR]
# Tested end to end on the public LendingClub population (matured.parquet) on 2026-09-20.
import sys, json, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

FEATURES = ['annual_inc','dti','emp_length','home_ownership','loan_amnt','purpose','term','verification_status']
MINB = 1000          # minimum observations per WoE bin (protocol 3.3)
NBINS = 8            # quantile bins for numeric characteristics (protocol 3.3)
MIN_HOLDOUT = 1000   # protocol 2.4: no conclusion below this many resolved holdout loans

def load(path):
    d = pd.read_parquet(path) if path.endswith('.parquet') else pd.read_csv(path, low_memory=False)
    need = set(FEATURES + ['default','vintage'])
    missing = need - set(d.columns)
    if missing: raise SystemExit(f'partner file lacks required fields: {sorted(missing)} (see DATA_SPECIFICATION.md)')
    return d

def qb(s, nb):
    v = pd.to_numeric(s, errors='coerce').astype(float); q = np.unique(np.nanquantile(v, np.linspace(0, 1, nb + 1)))
    if len(q) < 3: return None
    q = q.astype(float); q[0] = -1e18; q[-1] = 1e18; return q

def woe(bs, y, minb):
    g = pd.DataFrame({'b': bs, 'y': np.asarray(y)}).groupby('b')['y'].agg(['size', 'sum']); g = g[g['size'] >= minb]
    if len(g) < 2: return None, 0.0
    B = g['sum'] + 0.5; G = g['size'] - g['sum'] + 0.5; pb = B / B.sum(); pg = G / G.sum(); w = np.log(pg / pb)
    return w.to_dict(), float(((pg - pb) * w).sum())

def fit_spec(tr, cols):
    spec = {}
    for c in cols:
        if pd.api.types.is_numeric_dtype(tr[c]):
            q = qb(tr[c], NBINS)
            if q is None: continue
            bs = pd.cut(pd.to_numeric(tr[c], errors='coerce').astype(float), q).astype(str)
            m, iv = woe(bs, tr['default'], MINB)
            if m: spec[c] = ('n', q, m, iv)
        else:
            s = tr[c].fillna('MISS').astype(str); vc = s.value_counts(); keep = set(vc[vc >= MINB].index)
            bs = s.where(s.isin(keep), 'OTHER'); m, iv = woe(bs, tr['default'], MINB)
            if m: spec[c] = ('c', keep, m, iv)
    return spec

def transform(df, spec):
    X = pd.DataFrame(index=df.index)
    for c, (k, a, m, iv) in spec.items():
        if k == 'n': bs = pd.cut(pd.to_numeric(df[c], errors='coerce').astype(float), a).astype(str)
        else: s = df[c].fillna('MISS').astype(str); bs = s.where(s.isin(a), 'OTHER')
        X[c] = bs.map(m).astype(float).fillna(0.0)
    return X

def ks_stat(y, p):
    o = np.argsort(-np.asarray(p)); yy = np.asarray(y)[o]
    cb = np.cumsum(yy) / max(yy.sum(), 1); cg = np.cumsum(1 - yy) / max((1 - yy).sum(), 1)
    return float(np.max(np.abs(cb - cg)))

def psi(expected, actual, bins=10):
    cuts = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1))); cuts[0] = -np.inf; cuts[-1] = np.inf
    e = np.histogram(expected, cuts)[0] / len(expected); a = np.histogram(actual, cuts)[0] / len(actual)
    e = np.clip(e, 1e-6, None); a = np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))

def main():
    path = sys.argv[1]; train_through = None
    if '--train-through' in sys.argv: train_through = int(sys.argv[sys.argv.index('--train-through') + 1])
    d = load(path)
    vint = sorted(d['vintage'].dropna().unique())
    if train_through is None: train_through = vint[int(len(vint) * 0.6) - 1]   # protocol 2.3: earliest ~60% of vintages estimate, remainder hold out
    tr = d[d['vintage'] <= train_through]; te = d[d['vintage'] > train_through]
    if len(te) < MIN_HOLDOUT: raise SystemExit(f'holdout has {len(te)} loans, below protocol minimum {MIN_HOLDOUT}; no conclusion drawn')
    spec = fit_spec(tr, FEATURES)
    m = LogisticRegression(max_iter=1000, C=1.0).fit(transform(tr, spec), tr['default'])
    p_tr = m.predict_proba(transform(tr, spec))[:, 1]; p_te = m.predict_proba(transform(te, spec))[:, 1]
    auc = roc_auc_score(te['default'], p_te); gini = 2 * auc - 1; ks = ks_stat(te['default'], p_te)
    # calibration by decile (protocol 3.5)
    dec = pd.qcut(p_te, 10, labels=False, duplicates='drop')
    cal = pd.DataFrame({'pred': p_te, 'obs': te['default'].values, 'dec': dec}).groupby('dec').agg(n=('obs', 'size'), predicted=('pred', 'mean'), observed=('obs', 'mean')).reset_index()
    out = {
        'run': 'Phase 2 retrospective refit under Retrospective Validation Protocol v1.0',
        'input': path, 'loans_total': int(len(d)), 'estimation_n': int(len(tr)), 'holdout_n': int(len(te)),
        'estimation_vintages': [int(v) for v in vint if v <= train_through], 'holdout_vintages': [int(v) for v in vint if v > train_through],
        'estimation_default_rate': round(float(tr['default'].mean()), 4), 'holdout_default_rate': round(float(te['default'].mean()), 4),
        'characteristics_used': list(spec.keys()),
        'holdout_auc': round(auc, 4), 'holdout_gini': round(gini, 4), 'holdout_ks': round(ks, 4),
        'ks_x100': round(ks * 100, 1), 'protocol_ks_floor_x100': 20, 'passes_ks_floor': bool(ks * 100 >= 20),
        'psi_estimation_vs_holdout_scores': round(psi(p_tr, p_te), 4),
        'calibration_by_decile': cal.round(4).to_dict(orient='records'),
        'intercept': round(float(m.intercept_[0]), 4), 'coefficients': {c: round(float(b), 4) for c, b in zip(spec.keys(), m.coef_[0])},
        'information_value': {c: round(spec[c][3], 4) for c in spec},
    }
    json.dump(out, open('refit_results.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ('calibration_by_decile', 'coefficients', 'information_value')}, indent=1))

if __name__ == '__main__': main()