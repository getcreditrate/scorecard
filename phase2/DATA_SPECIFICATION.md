# Partner Data Specification, version 1.0

Fields a partner institution supplies for Phase 2B retrospective validation. Field names may differ; the mapping is recorded. No names, addresses, dates of birth, Social Security numbers or account numbers are transferred. The partner retains the identifier key.

## A. Required loan-level fields (modelling file)
| Field | Description | Format |
|---|---|---|
| loan_id | Hashed identifier; no reversible key retained by the Foundation | string |
| vintage | Origination year | integer |
| loan_amnt | Amount originated | numeric |
| term | Contractual term, months | integer |
| purpose | Stated purpose category | string |
| annual_inc | Applicant income as stated on application | numeric |
| dti | Debt to income ratio as computed at underwriting; computation basis stated | numeric |
| emp_length | Employment length as stated | string or numeric |
| home_ownership | Housing status as stated | string |
| verification_status | Whether and how income was verified | string |
| default | 1 if charged off or treated as credit loss; 0 if fully repaid | 0/1 |
| resolution_date | Date of payoff or charge off | date |
| incumbent_grade | Partner's own risk grade at origination, if any | string |

## B. Optional alternative data fields (where matched under a permissible purpose determination)
utility_payment_history, rental_payment_history, cashflow_features (average monthly inflow, inflow volatility, existing debt service, minimum balance).

## C. Protected class fields (separate file, joined only for fair lending testing, never a model input)
loan_id, race, ethnicity, sex, as recorded by the partner for regulatory reporting.

## D. Format and minimums
CSV or Parquet. Minimum 1,000 resolved loans and 100 realized defaults in the intended holdout vintages. Encrypted transfer.