# Get Credit Rate Foundation: Explainable Credit Scorecard Framework

**An open standard for lending to people the credit bureaus cannot see.**

Open-source Weight of Evidence (WoE) logistic regression scorecard for lenders that must explain every decision: Community Development Financial Institutions (CDFIs) and other regulated lenders serving applicants with thin or no credit bureau file.

Released by [Get Credit Rate Foundation](https://getcreditrate.com), a Delaware nonprofit corporation organized to qualify under Section 501(c)(3), under the Apache License 2.0. Author: Prince Akonor Asare, Director and Model Risk Owner.

| | |
|---|---|
| **Status** | Phase 1 (method) and Phase 2 (onboarding standard) complete. Phase 3 (per-institution calibration and champion-challenger pilot) scheduled to begin October 2026 with a Treasury-certified CDFI, subject to that institution's internal review and an executed data sharing agreement. |
| **Validation** | 1,249,246 resolved consumer loans, 254,055 realized defaults, out-of-time test on 462,426 loans |
| **Headline** | 81.9% of benchmark ranking power with every credit report field removed (8 application-level characteristics); 76.7% with debt-to-income also removed |
| **License** | Apache 2.0. Free to use, modify and redistribute. No fee, no permission, no vendor. |
| **Governance** | Model Risk Management Policy keyed to SR 11-7 / OCC 2011-12 / FDIC FIL-22-2017. No live use before local refit, fair-lending test and independent validation. |
| **History** | See [CHANGELOG.md](CHANGELOG.md) for the development record from 2024 |

---

## Why this exists

About 28 million American adults have no credit file and another 21 million have files too thin to score. CDFIs exist to lend to them, and the Federal Reserve's CDFI Survey records that technology and staffing are the constraints the sector reports most. Commercial scoring is closed, priced for large lenders, and cannot be inspected by the examiner of the institution that uses it.

This framework is the opposite on every point: open, free, and built so that every score decomposes into points an examiner can read and every decline produces the specific adverse action reasons the Equal Credit Opportunity Act requires. It is shared across institutions as a standard; the coefficients are always estimated on each institution's own borrowers, as Regulation B requires of any scoring system obtained from another person.

## What is here

### Phase 1: the method (files at repository root)

| Path | Contents |
|---|---|
| `extract_matured.py` | Builds the matured (resolved) loan set from the public LendingClub accepted-loan release (2007 to 2018 Q4, 2,260,701 rows) |
| `fit_scorecard.py` | Fits WoE bins on training vintages (2012 to 2015) and estimates the full-file and thin-file model variants; out-of-time test on 2016 to 2017 vintages |
| `build_appendices.py` | Builds the points-based scorecard (PDO 20, base 600 at 20:1 odds) and produces the bin tables, scorecard summary, population stability and adverse-action reason code appendices |
| `ablation.py` | Feature ablation across Variants B, C, D and E against the platform's own risk grade |
| `ablation_variant_f.py` | Addendum of 18 September 2026: reproduces Variant D and estimates Variant F with the debt to income ratio also removed |
| `variantF_results.json` | Machine-readable results of the addendum run |
| `Model_Validation_Report_with_Addendum_2026-09-18.pdf` | Model Validation Report (Version 1.0, October 2025) with Addendum of 18 September 2026 |
| `Model_Card_and_Explainability_Architecture.pdf` | Model Card and Explainability Architecture (Version 1.0, October 2025) |

### Phase 2: the onboarding standard (`phase2/`)

Issued 20 September 2026 and frozen before any institution's record was received, so that no calibration can be shaped by the data it tests. Published here 25 September 2026.

| Path | Contents |
|---|---|
| `phase2/PROTOCOL.md` | Retrospective Validation Protocol v1.0: population, out-of-time split, refit and recalibration method, discrimination and calibration thresholds, fair-lending tests with pass and fail rules, minimum sample floors |
| `phase2/DATA_SPECIFICATION.md` | Every field a partner institution supplies, and the de-identification standard |
| `phase2/VALIDATOR_SCOPE.md` | Scope of work for the independent external validator |
| `phase2/refit.py` | Implements Protocol sections 2 and 3 on an institution's closed loan book |
| `phase2/disparate_impact.py` | Implements Protocol section 4: adverse impact ratio (four-fifths screen), standardized mean difference, within-group calibration |
| `phase2/refit_results_public.json` | End-to-end run on the public population: holdout KS 21.9 against the floor of 20; PSI 0.009 |
| `phase2/disparate_impact_results_synthetic.json` | Execution proof on a synthetic group column; no inference about any real population |
| `phase2/CODE_READINESS.md` | Code Readiness Record |

The Data Sharing Agreement template and the Specialty Consumer Reporting Agency Memorandum are held for counsel review and are available from the Foundation on request.

## Headline results (out-of-time, 462,426 test loans, 2016 to 2017 vintages)

| Variant | Feature set | k | AUC | Gini | Share of benchmark Gini |
|---|---|---|---|---|---|
| Benchmark | Platform internal risk grade | | 0.6874 | 0.3749 | 100.0% |
| A | Full file, includes score and deep history | 51 | 0.6957 | 0.3914 | 104.4% |
| B | Score and deep history removed | 31 | 0.6883 | 0.3766 | 100.5% |
| C | B with requested term also removed | 30 | 0.6756 | 0.3512 | 93.7% |
| D | All credit report fields removed (8 application-level characteristics) | 8 | 0.6535 | 0.3070 | 81.9% |
| E | D with requested term also removed | 7 | 0.6377 | 0.2754 | 73.5% |
| F | D with debt to income ratio also removed | 7 | 0.6437 | 0.2874 | 76.7% |

Population: 1,249,246 resolved consumer loans in 2012 to 2017 origination vintages, 254,055 realized defaults. Estimation set 786,820 loans; test set 462,426 loans. Full method, calibration analysis and stated limitations are in `Model_Validation_Report_with_Addendum_2026-09-18.pdf`.

## Why WoE logistic regression

Every score decomposes into points per characteristic traceable to a published bin, so each declined application produces specific adverse action reasons drawn from the applicant's own data, as the Equal Credit Opportunity Act and Regulation B require. The audit record retained per decision holds the inputs, the bins, the points awarded, the model version and the timestamp. A gradient boosted model would score marginally better on this population and could not do any of that. See `Model_Card_and_Explainability_Architecture.pdf`.

## How an institution adopts it

1. **Calibrate.** Supply resolved closed-loan records under `phase2/DATA_SPECIFICATION.md`. `phase2/refit.py` refits the coefficients and recalibrates the intercept on your own borrowers under the frozen protocol. Regulation B, 12 CFR 1002.2(p)(2), requires this of any scoring system obtained from another person; the framework is built so that you meet it on your own records.
2. **Test for fair lending.** `phase2/disparate_impact.py` runs the four-fifths screen, standardized mean difference and within-group calibration across race, ethnicity and sex, with the minimum group floors in the protocol.
3. **Validate independently.** An external validator with no role in developing the model reviews the refit, the tests and the documentation under `phase2/VALIDATOR_SCOPE.md`.
4. **Run in parallel.** Score live applications alongside your existing process, controlling no decision, until your own risk function confirms performance at least equal to the incumbent for the applicants newly reached.
5. **Go live.** Only after every open finding is resolved or accepted in writing by your institution.

The Foundation provides implementation support, fair-lending validation and training. Nothing about the code requires it.

## Limitations, stated plainly

1. Every borrower in the validation data held a credit file. The results show that the architecture ranks repayment risk without bureau inputs on a credit-visible population; they do not measure performance on borrowers who have no file. That measurement happens at each institution in the calibration stage of Phase 3, under the Phase 2 protocol, on the institution's own records.
2. No cash flow, rent, utility or telecom variable exists in the public source. Validation of those features is part of Phase 3 calibration, where matched data is available under a permissible purpose determination.
3. No race, ethnicity or sex field exists in the public source, so disparate impact has not been tested here. Fair lending testing runs at each institution under `phase2/PROTOCOL.md` section 4.
4. Requested loan term carries the highest information value of any characteristic and is a product selection, not a borrower attribute. Where term is set by policy rather than chosen by the applicant, Variant E is the figure to rely on.
5. Calibration drifted between training and test eras (18.64% against 23.23% default rate). Rank ordering transfers out of time; the intercept must be recalibrated on any deploying population.
6. Under the Foundation's Model Risk Management Policy, no model here may be used in a live credit decision until coefficients are refit on the deploying institution's borrowers, disparate impact is tested on that portfolio, and independent validation is complete.

## Roadmap

| Phase | What | Status |
|---|---|---|
| 1 | The method: model, validation, explainability architecture, compliance register, governance, open release | Complete, September 2026 |
| 2 | The onboarding standard: pre-registered protocol, data specification, validator scope, implementing code | Complete, 20 September 2026 |
| 3 | Per-institution calibration and champion-challenger pilot at three to five CDFIs | Scheduled to begin October 2026 |
| 4 | National scaling through CDFI intermediaries; playbooks and compliance modules published under Apache 2.0 | Scheduled on completion of Phase 3 |

## Reproducing the Phase 1 results

```
pip install -r requirements.txt
# place the public LendingClub accepted-loan release at data/accepted_2007_to_2018Q4.csv (data/ is excluded from the repository)
python extract_matured.py        # writes matured.parquet
python fit_scorecard.py          # writes woe_spec.pkl and the validation report text
python build_appendices.py       # writes scorecard_model.pkl and appendices
python ablation.py               # Variants B to E
python ablation_variant_f.py     # Variant D reproduction and Variant F
python phase2/refit.py matured.parquet --train-through 2015   # Phase 2 pipeline check on the public population
```

The validation data is not redistributed in this repository. It is a public release available from its original source and from Kaggle.

## Governance

Model development, documentation, validation and monitoring follow the Foundation's Model Risk Management Policy, keyed to SR 11-7, OCC Bulletin 2011-12 and FDIC FIL-22-2017, which five federal agencies have confirmed applies to models leveraging alternative data. The board's second line review is constituted by the two directors who are not the model developer; independent external validation is commissioned before any live use. The adopted policy and board consents carry director signatures and are available from the Foundation on request.

## Citing this work

See [CITATION.cff](CITATION.cff). Questions, corrections and institutions interested in Phase 3: open an issue or write to the Foundation through [getcreditrate.com](https://getcreditrate.com).

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
