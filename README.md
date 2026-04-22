# Expedia Churn Case Study - Source Code

This repository contains the source code and notebook for a booking-level churn case study for Hotels.com.

The objective was to:

- identify the most important drivers of churn behavior
- build an explainable statistical model to identify which customers are more vs. less likely to churn
- summarize the results into business recommendations for leadership

## Repository Contents

- `Hotels_Churn_Case_Study.ipynb`  
  Main notebook containing the analysis workflow

- `scripts/build_churn_case_study.py`  
  End-to-end Python script for cleaning, modeling, evaluation, and output generation

- `examples/minimal_churn_workflow.py`  
  A shorter version of the modeling logic for quick review

- `requirements.txt`  
  Python packages used in the project

- `assets/`  
  A few representative charts used in the case study

## Modeling Approach

- Unit of analysis: booking-level rows
- Target: `churn_flag`
- Final model: Logistic Regression
- Challenger model: shallow Decision Tree
- Validation approach: time-based train/test split using `bk_date`

## Why Logistic Regression Was Selected

Both Logistic Regression and the Decision Tree performed similarly on holdout data, but Logistic Regression was selected as the final model because it was easier to interpret and explain to stakeholders while still delivering strong predictive performance.

Key holdout metrics for Logistic Regression:

- ROC-AUC: `0.876`
- PR-AUC: `0.847`
- Accuracy: `79.6%`

## Main Churn Drivers

The strongest model drivers were:

1. `loyalty_tier`
2. `customer_type`
3. `marketing_channel`
4. `bkg_confirmation_pages_count`
5. `bounce_visits_count`

## Data Handling Notes

- Identifier columns such as `email_address` and `booking_id` were excluded from modeling
- `cancel_date` was excluded from the final model to avoid target leakage because it is only known after the cancellation event
- Missing values were handled in a simple, explainable way:
  - `cancel_flag` -> filled with `0`
  - `marketing_channel` -> filled with `"Missing"`
  - `total_visit_minutes` -> median imputation

## Reproduce Locally

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/build_churn_case_study.py
```

For a lighter review of the model logic:

```bash
.venv/bin/python examples/minimal_churn_workflow.py /path/to/case_study_data.csv
```

## Note On Data

The raw case-study CSV is not included in this repository. This repository is intended to share the modeling logic, notebook, and representative outputs without redistributing the source dataset.
