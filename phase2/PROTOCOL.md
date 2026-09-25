# Retrospective Validation Protocol, version 1.0

Frozen 20 September 2026, before receipt of any partner institution record. Issued by Get Credit Rate Foundation under its Model Risk Management Policy. Any deviation during Phase 2B is recorded in the validation report with the reason.

## 1. Purpose
Fix in advance what the retrospective validation of the scorecard on a partner CDFI's closed loan book will measure, how, and against which thresholds, so the validation cannot be shaped by the data it tests.

## 2. Population and split
- 2.1 Closed originations with realized outcome (fully repaid or charged off). Open loans excluded. Most recent vintages shorter than the longest contractual term excluded (early payoff / early default bias).
- 2.2 Default = charge off or any status the partner treats as a credit loss under its own policy; partner definition applied unchanged.
- 2.3 Out-of-time split by origination vintage: earliest vintages comprising about 60 percent of resolved loans estimate; all later vintages hold out. Bins and coefficients fitted on the estimation set only.
- 2.4 Minimum sample: no discrimination conclusion below 1,000 resolved holdout loans or 100 realized defaults; no subgroup conclusion below 100 loans. Below these floors the result is recorded as insufficient.

## 3. Refit and recalibration
- 3.1 Feature set: the Variant D application-level characteristics (reported income, debt to income ratio, employment length, housing status, requested amount, stated purpose, requested term, income verification) or the subset the partner records; omissions stated. Term reported both retained and removed (Variants D and E).
- 3.2 Alternative data features, where matched: model estimated with and without; incremental lift = change in holdout AUC and Gini; below 0.005 AUC recorded as no material lift.
- 3.3 Eight quantile bins on the estimation set, minimum 1,000 observations per bin (fewer bins where unsupported); categorical levels below 1,000 pooled; WoE with 0.5 smoothing; logistic regression on WoE; any coefficient contradicting its WoE direction dropped and recorded.
- 3.4 Intercept recalibrated to partner default rate; reported separately from coefficient refit.
- 3.5 Holdout measures: AUC, Gini (2 x AUC - 1), KS; calibration by score decile; PSI between estimation and holdout score distributions.
- 3.6 Thresholds (Technical Architecture s.5.7): not production ready below KS 20 (x100); 30 to 40 strong for a first generation alternative data scorecard. PSI below 0.10 no shift; 0.10 to 0.25 investigate; above 0.25 model review. Holdout AUC below the partner's own incumbent grade, where scoreable, is a finding.

## 4. Fair lending testing
- 4.1 Protected classes: race, ethnicity, sex as recorded by the partner for regulatory reporting, or proxied by a partner-approved method. Held separately; joined only for this test; never a model input.
- 4.2 Adverse impact ratio at the partner's approval cutoff, each group against the reference group (largest group meeting minimum sample). Below 0.80 is a flag requiring 4.5.
- 4.3 Standardized mean score difference by group.
- 4.4 Within-group calibration: observed default rate among approved, by group.
- 4.5 Less discriminatory alternative search where 4.2 flags: alternative characteristic set or binning removing the flag within 0.01 AUC; search and result recorded whether or not found.

## 5. Deliverable
A Phase 2 Validation Report in the format of the Model Validation Report: population and exclusions; split; feature set and omissions; bins and coefficients; discrimination and calibration against 3.6; alternative data lift; fair lending results; every deviation with reason; stated limitations. Presented to the second line review function and to the independent validator.

## Implementation
`refit.py` implements sections 2 and 3. `disparate_impact.py` implements section 4. Both were executed end to end on the public validation population on 20 September 2026 (see `CODE_READINESS.md`).