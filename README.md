# Get Credit Rate Foundation: Explainable Credit Scorecard Framework

Open-source Weight of Evidence (WoE) logistic regression scorecard for lenders that must explain every decision: Community Development Financial Institutions (CDFIs) and other regulated lenders serving applicants with thin or no credit bureau file.

Author: Prince Akonor Asare, Director, Get Credit Rate Foundation.

Released by [Get Credit Rate Foundation](https://getcreditrate.com), a Delaware nonprofit corporation organized to qualify under Section 501(c)(3), under the Apache License 2.0.

## What is here

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

Every score decomposes into points per characteristic traceable to a published bin, so each declined application produces specific adverse action reasons drawn from the applicant's own data, as the Equal Credit Opportunity Act and Regulation B require. The audit record retained per decision holds the inputs, the bins, the points awarded, the model version and the timestamp. See `Model_Card_and_Explainability_Architecture.pdf`.

## Limitations, stated plainly

1. Every borrower in the validation data held a credit file. The results show that the architecture ranks repayment risk without bureau inputs on a credit-visible population; they do not measure performance on borrowers who have no file. That measurement is Phase 2 of the Foundation's implementation plan, on a partner institution's loan records.
2. No cash flow, rent, utility or telecom variable exists in the public source. Validation of those features is deferred to Phase 2.
3. No race, ethnicity or sex field exists in the public source, so disparate impact has not been tested here. Fair lending testing is assigned to Phase 2.
4. Calibration drifted between training and test eras (18.64% against 23.23% default rate). Rank ordering transfers out of time; the intercept must be recalibrated on any deploying population.
5. Under the Foundation's Model Risk Management Policy, no model here may be used in a live credit decision until coefficients are refit on the deploying institution's borrowers, disparate impact is tested on that portfolio, and independent validation is complete.

## Reproducing the results

```
pip install -r requirements.txt
# place the public LendingClub accepted-loan release at data/accepted_2007_to_2018Q4.csv (data/ is excluded from the repository)
python extract_matured.py        # writes matured.parquet
python fit_scorecard.py          # writes woe_spec.pkl and the validation report text
python build_appendices.py       # writes scorecard_model.pkl and appendices
python ablation.py               # Variants B to E
python ablation_variant_f.py     # Variant D reproduction and Variant F
```

The validation data is not redistributed in this repository. It is a public release available from its original source and from Kaggle.

## Governance

Model development, documentation, validation and monitoring follow the Foundation's Model Risk Management Policy, keyed to SR 11-7, OCC Bulletin 2011-12 and FDIC FIL-22-2017, which five federal agencies have confirmed applies to models leveraging alternative data. The adopted policy is available from the Foundation on request.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.

