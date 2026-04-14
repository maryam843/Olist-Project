# Olist Trust Gap Model

### Delivery Promise Integrity — Predicting Customer Review Scores Before They Are Written

**DSAI 4103 Business Analytics | Maryam Mahaboob | 60301005**

Olist (Brazilian e-commerce) promises a delivery date at checkout.
This project predicts the **customer review score (1–5 stars)** before it is written, using delivery timing, seller reliability, and order features.

## Dataset
**Olist Brazilian E-Commerce** — https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
9 tables | 96,478 delivered orders | Oct 2016 – Aug 2018
Download all CSV files from Kaggle and place them in a folder on your Google Drive.

## How to Run

### Upload the dataset to Google Drive

Create a folder in your Drive:

Place all 9 CSV files inside it:
- olist_orders_dataset.csv
- olist_order_reviews_dataset.csv
- olist_order_items_dataset.csv
- olist_order_payments_dataset.csv
- olist_customers_dataset.csv
- olist_products_dataset.csv
- olist_sellers_dataset.csv
- olist_geolocation_dataset.csv
- product_category_name_translation.csv

### Open the notebook in Google Colab
Upload `SourceCode.ipynb` to Colab or open directly from Drive.

### Run the first cell to install dependencies
```python
!pip install flaml --quiet
```

### Mount Google Drive when prompted
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Run all cells in order

After running, the following files are saved to MyDrive/olist_project/model_package/:
  xgb_final_model.json: XGBoost model weights 
  model_metadata.json: Metrics, feature list, per-class F1 
  encoding_maps.json: State and category target encodings 
  score.py: Scoring script — input order features, output predicted star rating 

## Dashboard
Built in Power BI using 6 exported CSV files.
Live dashboard: https://app.powerbi.com/groups/me/reports/a6c52a94-efe4-4964-85d7-a339085926e7/b778fa8adfa82d3a35b1?ctid=b30f4b44-46c6-4070-9997-f87b38d4771c&experience=power-bi


## Requirements
All handled automatically in Colab. No local installation needed.
Core libraries used:
- pandas, numpy
- xgboost
- flaml
- shap
- scikit-learn
- matplotlib, seaborn
