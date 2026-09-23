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

## What we're trying to find

- Does a late delivery (actual vs promised date) drive low review scores?
- Can we predict an unhappy customer's review before it's written, so they can be contacted early?
- Which factors matter most: delivery timing or seller reliability?

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
