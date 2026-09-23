# Olist Trust Gap Model

Predicting a customer's review score (1 to 5 stars) before the review is written, from how early or late the order arrived, how reliable the seller has been, and basic order details.

Course project for DSAI 4103 Advanced Business Analytics. Author: Maryam Mahaboob.

## Overview

Olist, a Brazilian e-commerce marketplace, promises a delivery date at checkout. When that promise is broken, customers leave bad reviews. This project predicts the review score once an order is delivered, so likely unhappy customers could be contacted early.

- **Data:** [Olist Brazilian E-Commerce dataset on Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), 9 tables. Raw data is not included; see [data/README.md](data/README.md).
- **Rows:** 96,478 delivered orders; 95,808 remain after joining reviews and cleaning.
- **Split:** time-based 80/20 on purchase date: 76,646 training orders and 19,162 test orders.
- **Features:** 25, including the trust gap (actual minus promised delivery date, in days), four seller reliability features, state and category target encodings, and order details.
- **Model:** FLAML AutoML on a 15,000-order training sample selected XGBoost. The final model is an XGBoost classifier with balanced class weights.

## Key results

Numbers are from the notebook's saved outputs on the 19,162 test orders.

| Metric | Model | Baseline: always predict 5 stars |
|---|---|---|
| Macro F1 | **0.2581** | 0.1569 |
| Accuracy | 0.4236 | 0.6455 |
| Weighted F1 | 0.4580 | |
| ROC-AUC (macro, one-vs-rest) | 0.5893 | |

Accuracy is below the always-5-stars baseline because the model is weighted to find unhappy customers (macro F1), not to maximise accuracy. Per-class F1 ranges from 0.0606 (2 stars) to 0.5931 (5 stars).

![Delay damage curve](figures/plot3_delay_damage_curve.png)

- **Most orders arrive early.** 92.0% arrived before the promised date and 6.7% arrived late.
- **Lateness drives scores down.** Average score: early 4.29, 1 to 3 days late 3.29, 4 to 7 days late 2.11, 8 to 14 days late 1.67.
- **Top SHAP features** (mean absolute SHAP value, 2,000 test orders): seller_avg_score (0.1555), trust_gap_days (0.1089), item_count (0.0691), actual_delivery_days (0.0611), seller_order_volume (0.0548).
- **Fairness.** TO, RO and SE score more than 0.05 below the overall macro F1 (0.1235, 0.1790, 0.1907), as do 8 of 41 product categories. A reweighting experiment on weak states and categories lowered macro F1 to 0.2571 and widened the state F1 range (0.1657 to 0.2310), so the original model was kept.

**Caveats.** On training rows, the seller, state and category averages include that row's own review score, which likely inflates seller_avg_score's importance. XGBoost early stopping used the test set, so test scores are slightly optimistic.

**Note on the model files.** The files in `model_package/` come from a later re-run of the same notebook. Their test macro F1 is 0.2601, compared with 0.2581 in the notebook's saved outputs.

## Repo structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── olist_trust_gap.ipynb            cleaning, EDA, model, SHAP, fairness
├── model_package/
│   ├── xgb_final_model.json             trained XGBoost model
│   ├── model_metadata.json              feature list, metrics, limitations
│   ├── encoding_maps.json               state and category encodings
│   └── score.py                         scores a single order
├── figures/                             EDA plots (plot1 to plot10)
├── dashboard/
│   ├── OlistTrustGap_Dashboard.pbix     Power BI file
│   └── OlistTrustGap_Dashboard.pdf      PDF export of the dashboard
├── presentation/
│   └── OlistTrustGap_Slides.pdf         slides
└── data/
    └── README.md                        where to get the raw data
```

Dashboard: [PDF](dashboard/OlistTrustGap_Dashboard.pdf) · [Power BI file](dashboard/OlistTrustGap_Dashboard.pbix). Slides: [presentation/OlistTrustGap_Slides.pdf](presentation/OlistTrustGap_Slides.pdf).

## How to run

**Notebook (Google Colab):**

1. Download the 9 CSV files listed in [data/README.md](data/README.md).
2. Put them in `MyDrive/olist_project/` on Google Drive.
3. Open `notebooks/olist_trust_gap.ipynb` in Colab, run the first cell (`!pip install flaml --quiet`), mount Drive when asked, then run all cells. FLAML AutoML takes about 3 minutes.

**Score an order with the saved model:**

```
pip install -r requirements.txt
python model_package/score.py
```

`score.py` runs two example orders. To score your own, call `predict_score()` with a dict of order features, including the four seller features.
