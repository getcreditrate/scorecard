# Phase 2 Code Readiness Record

Both Phase 2 scripts were written to the Retrospective Validation Protocol v1.0 and executed end to end on the public validation population on 20 September 2026, so that they run on partner records without modification.

## refit.py on the public LendingClub resolved population
Estimation vintages 2007 to 2015; holdout 2016 to 2018.

| Measure | Result | Protocol |
|---|---|---|
| Loans in file | 1345350 | 2.1 |
| Estimation / holdout | 826606 / 518744 | 2.3 |
| Estimation / holdout default rate | 18.43% / 22.42% | 3.4 |
| Holdout AUC | 0.6526 | 3.5 |
| Holdout Gini | 0.3052 | 3.5 |
| Holdout KS (x100) | 21.9 | 3.6 floor 20: passes |
| PSI, estimation vs holdout scores | 0.0089 | 3.6: below 0.10 |

Consistent with Variant D of the Model Validation Report (AUC 0.6535 on the 2012 to 2017 window). Full output: `refit_results_public.json`.

## disparate_impact.py
The public population has no protected class field. To prove execution, the script was run on the scored holdout with a randomly assigned column explicitly named `SYNTHETIC_group`. Adverse impact ratios of 1.00 across three synthetic groups were reported, as expected for random assignment, and the minimum sample rule was applied. **No inference about any real population is drawn or implied.** Output: `disparate_impact_results_synthetic.json`.

## Reproduce
```
python phase2/refit.py matured.parquet --train-through 2015
python phase2/disparate_impact.py <scored file> --classes race,ethnicity,sex --cutoff 0.20 --higher-is-riskier
```