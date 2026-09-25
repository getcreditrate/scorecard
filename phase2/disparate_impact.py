# Copyright 2026 Get Credit Rate Foundation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
#
# Phase 2 disparate impact test under the Retrospective Validation Protocol, section 4.
# Inputs: a scored file with columns score (higher = lower risk or probability of default; set --higher-is-riskier),
# default (0/1), and one or more protected-class columns (race, ethnicity, sex). For each protected class:
#   (a) adverse impact ratio at a stated approval cutoff (four-fifths rule screen, protocol 4.2);
#   (b) standardized mean difference in score between each group and the reference group (protocol 4.3);
#   (c) within-group calibration: observed default rate among approved applicants by group (protocol 4.4).
# Groups with fewer than MIN_GROUP outcomes are reported but no conclusion is drawn (protocol 2.4).
# Usage: python disparate_impact.py <scored csv/parquet> --classes race,ethnicity,sex --cutoff 0.20 [--higher-is-riskier]
# Tested end to end on 2026-09-20 with a clearly labelled SYNTHETIC protected-class column appended to the
# public LendingClub population, solely to prove the pipeline executes. The public data contain no protected-class fields.
import sys, json, numpy as np, pandas as pd

MIN_GROUP = 100   # protocol 2.4

def main():
    path = sys.argv[1]
    classes = sys.argv[sys.argv.index('--classes') + 1].split(',')
    cutoff = float(sys.argv[sys.argv.index('--cutoff') + 1]) if '--cutoff' in sys.argv else 0.20
    higher_riskier = '--higher-is-riskier' in sys.argv
    d = pd.read_parquet(path) if path.endswith('.parquet') else pd.read_csv(path, low_memory=False)
    s = d['score'].astype(float)
    approved = (s <= cutoff) if higher_riskier else (s >= cutoff)
    report = {'run': 'Phase 2 disparate impact test under Retrospective Validation Protocol v1.0', 'input': path, 'n': int(len(d)),
              'cutoff': cutoff, 'higher_is_riskier': higher_riskier, 'overall_approval_rate': round(float(approved.mean()), 4), 'classes': {}}
    for cls in classes:
        if cls not in d.columns: report['classes'][cls] = 'column absent'; continue
        g = d[cls].fillna('UNKNOWN').astype(str)
        rates = approved.groupby(g).mean(); counts = g.value_counts()
        ref = rates[counts >= MIN_GROUP].idxmax() if (counts >= MIN_GROUP).any() else rates.idxmax()
        rows = {}
        for grp in counts.index:
            n = int(counts[grp]); ar = float(rates[grp]); air = ar / float(rates[ref]) if rates[ref] > 0 else float('nan')
            smd = float((s[g == grp].mean() - s[g == ref].mean()) / s.std()) if s.std() > 0 else 0.0
            appr_def = d.loc[approved & (g == grp), 'default']
            rows[grp] = {'n': n, 'approval_rate': round(ar, 4), 'adverse_impact_ratio_vs_reference': round(air, 4),
                         'four_fifths_screen': ('reference' if grp == ref else ('PASS' if air >= 0.8 else 'FLAG')) if n >= MIN_GROUP else 'INSUFFICIENT N',
                         'standardized_mean_score_difference': round(smd, 4),
                         'observed_default_rate_among_approved': round(float(appr_def.mean()), 4) if len(appr_def) else None,
                         'conclusion_permitted': bool(n >= MIN_GROUP)}
        report['classes'][cls] = {'reference_group': str(ref), 'groups': rows}
    json.dump(report, open('disparate_impact_results.json', 'w'), indent=1)
    print(json.dumps(report, indent=1)[:3000])

if __name__ == '__main__': main()