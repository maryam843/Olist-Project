#!/usr/bin/env python3
"""
score.py — Olist Trust Gap Model Scoring Script
Project: Delivery Promise Integrity — The Trust Gap Model
Course:  DSAI 4103 Business Analytics

Takes order features as input.
Returns predicted review score (1-5) and class probabilities.

Usage:
    python score.py
    OR: from score import predict_score
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import json

MODEL_PATH    = "model_package/xgb_final_model.json"
METADATA_PATH = "model_package/model_metadata.json"
ENCODING_PATH = "model_package/encoding_maps.json"


def load_artifacts():
    """Load model, metadata, and encoding maps."""
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    with open(METADATA_PATH) as f:
        meta = json.load(f)
    with open(ENCODING_PATH) as f:
        enc = json.load(f)
    return model, meta, enc


def predict_score(order_dict):
    """
    Predict review score for a single order.

    Parameters
    ----------
    order_dict : dict
        Must contain these keys:
        trust_gap_days, actual_delivery_days, promised_delivery_days,
        shipping_lag_days, approval_lag_days, total_price, total_freight,
        freight_ratio, item_count, payment_installments, payment_value,
        product_weight_g, product_length_cm, product_height_cm,
        product_width_cm, same_state, customer_state,
        product_category_name_english, payment_type,
        seller_avg_score, seller_late_rate, seller_avg_gap,
        seller_order_volume

    Returns
    -------
    dict with predicted_score (int 1-5) and probabilities per class
    """
    model, meta, enc = load_artifacts()

    df = pd.DataFrame([order_dict])

    # Apply target encodings using training-set maps
    global_avg = enc["global_avg_score"]
    df["state_risk_score"]    = df["customer_state"].map(
        enc["state_risk_score"]).fillna(global_avg)
    df["category_risk_score"] = df["product_category_name_english"].map(
        enc["category_risk_score"]).fillna(global_avg)

    # Payment type dummies
    df["pay_credit_card"] = (df["payment_type"] == "credit_card").astype(float)
    df["pay_debit_card"]  = (df["payment_type"] == "debit_card").astype(float)
    df["pay_voucher"]     = (df["payment_type"] == "voucher").astype(float)

    X      = df[meta["feature_cols"]].astype(float).values
    pred   = int(model.predict(X)[0]) + 1
    proba  = model.predict_proba(X)[0]

    return {
        "predicted_score": pred,
        "confidence"     : round(float(proba[pred - 1]), 4),
        "probabilities"  : {
            "1_star": round(float(proba[0]), 4),
            "2_star": round(float(proba[1]), 4),
            "3_star": round(float(proba[2]), 4),
            "4_star": round(float(proba[3]), 4),
            "5_star": round(float(proba[4]), 4),
        },
        "model_version": meta["model_name"]
    }


if __name__ == "__main__":
    # ── Example 1: Late order from bad seller
    late_bad_seller = {
        "trust_gap_days"                : 8,
        "actual_delivery_days"          : 25,
        "promised_delivery_days"        : 17,
        "shipping_lag_days"             : 5,
        "approval_lag_days"             : 0,
        "total_price"                   : 180.0,
        "total_freight"                 : 28.0,
        "freight_ratio"                 : 0.156,
        "item_count"                    : 1,
        "payment_installments"          : 3,
        "payment_value"                 : 208.0,
        "product_weight_g"              : 1200,
        "product_length_cm"             : 35,
        "product_height_cm"             : 20,
        "product_width_cm"              : 25,
        "same_state"                    : 0,
        "customer_state"                : "MA",
        "product_category_name_english" : "furniture_living_room",
        "payment_type"                  : "credit_card",
        "seller_avg_score"              : 3.1,
        "seller_late_rate"              : 0.25,
        "seller_avg_gap"                : 5.0,
        "seller_order_volume"           : 45
    }

    # ── Example 2: Early order from reliable seller
    early_good_seller = {
        "trust_gap_days"                : -10,
        "actual_delivery_days"          : 7,
        "promised_delivery_days"        : 17,
        "shipping_lag_days"             : 1,
        "approval_lag_days"             : 0,
        "total_price"                   : 95.0,
        "total_freight"                 : 14.0,
        "freight_ratio"                 : 0.147,
        "item_count"                    : 1,
        "payment_installments"          : 1,
        "payment_value"                 : 109.0,
        "product_weight_g"              : 400,
        "product_length_cm"             : 20,
        "product_height_cm"             : 10,
        "product_width_cm"              : 15,
        "same_state"                    : 1,
        "customer_state"                : "SP",
        "product_category_name_english" : "health_beauty",
        "payment_type"                  : "credit_card",
        "seller_avg_score"              : 4.7,
        "seller_late_rate"              : 0.02,
        "seller_avg_gap"                : -12.0,
        "seller_order_volume"           : 380
    }

    print("OLIST TRUST GAP MODEL — SCORING EXAMPLES")
    print("=" * 50)

    for name, order in [("Late order, bad seller", late_bad_seller),
                         ("Early order, good seller", early_good_seller)]:
        result = predict_score(order)
        print(f"\nExample: {name}")
        print(f"  Predicted score: {result['predicted_score']}★")
        print(f"  Confidence:      {result['confidence']*100:.1f}%")
        print(f"  Probabilities:   {result['probabilities']}")
