# Dynamic Pricing in E-Commerce — Review 2

**Dataset:** Brazilian Olist E-Commerce (2016–2018)  
**Algorithms:** XGBoost | LightGBM  
**Objective:** Predict product prices using two advanced gradient boosting models trained on the Olist e-commerce dataset.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Workflow](#workflow)
4. [Feature Engineering](#feature-engineering)
5. [Algorithms](#algorithms)
   - [XGBoost](#algorithm-1--xgboost-extreme-gradient-boosting)
   - [LightGBM](#algorithm-2--lightgbm-light-gradient-boosting-machine)
6. [Results](#results)
7. [Requirements](#requirements)
8. [How to Run](#how-to-run)

---

## Project Overview

This project implements a dynamic pricing pipeline on the Brazilian Olist e-commerce dataset. Review 2 focuses on two state-of-the-art gradient boosting regressors — **XGBoost** and **LightGBM** — and performs a head-to-head comparison using identical features and train/test splits to ensure a fair evaluation.

---

## Dataset

Eight Olist CSV files are merged into a single master dataframe:

| File | Description |
|---|---|
| `olist_orders_dataset.csv` | Order metadata and timestamps |
| `olist_order_items_dataset.csv` | Per-item price and freight values |
| `olist_order_payments_dataset.csv` | Payment type, value, installments |
| `olist_order_reviews_dataset.csv` | Customer review scores |
| `olist_products_dataset.csv` | Product dimensions and category |
| `olist_customers_dataset.csv` | Customer location data |
| `olist_sellers_dataset.csv` | Seller location data |
| `product_category_name_translation.csv` | Portuguese → English category names |

**Filters applied:**
- Only **delivered** orders are retained.
- Prices restricted to the range **R$ 0 – R$ 5,000**.

---

## Workflow

| Step | Description |
|---|---|
| 1 | Import Libraries |
| 2 | Load & Merge Datasets |
| 3 | Feature Engineering & Train-Test Split |
| 4 | Algorithm 1 — XGBoost |
| 5 | Algorithm 2 — LightGBM |
| 6 | Final Comparison — XGBoost vs LightGBM |

---

## Feature Engineering

Seven features are used across both models to ensure a fair apples-to-apples comparison:

| Feature | Type | Description |
|---|---|---|
| `freight_value` | Continuous | Shipping cost in R$ |
| `review_score` | Continuous (1–5) | Mean customer rating |
| `payment_installments` | Integer | Max number of payment installments |
| `days_to_deliver` | Integer | Actual delivery time in days |
| `purchase_month` | Integer (1–12) | Month of purchase |
| `is_weekend` | Binary (0/1) | 1 if ordered on Saturday or Sunday |
| `category_encoded` | Integer (0–71) | Label-encoded product category |

Additional engineered columns (stored in master dataframe):
- `purchase_dow` — day of week
- `purchase_year` — year of purchase
- `delivery_delay` — actual vs estimated delivery gap (days)
- `freight_ratio` — freight value as a proportion of price

**Train/Test Split:** 80% train / 20% test, `random_state=42`

---

## Algorithms

### Algorithm 1 — XGBoost (Extreme Gradient Boosting)

XGBoost builds trees **sequentially** — each new tree corrects the residual errors of all previous trees. It minimises a regularised objective:

$$\mathcal{L} = \sum_{i=1}^{n} l(y_i, \hat{y}_i) + \sum_{k=1}^{K} \Omega(f_k), \quad \Omega(f) = \gamma T + \frac{1}{2}\lambda \|\mathbf{w}\|^2$$

where $l$ is the loss, $T$ is the number of leaves, $\mathbf{w}$ are leaf weights, and $\Omega$ is the regularisation term preventing overfitting.

**Hyperparameters:**

| Parameter | Value | Effect |
|---|---|---|
| `n_estimators` | 300 | 300 sequential trees |
| `max_depth` | 6 | Controls tree complexity |
| `learning_rate` | 0.05 | Small steps → better generalisation |
| `subsample` | 0.8 | 80% of rows per tree — prevents overfitting |
| `colsample_bytree` | 0.8 | 80% of features per tree |
| `reg_alpha` | 0.1 | L1 regularisation |
| `reg_lambda` | 1.0 | L2 regularisation |

---

### Algorithm 2 — LightGBM (Light Gradient Boosting Machine)

LightGBM is a gradient boosting framework by Microsoft that grows trees **leaf-wise** (best-first) instead of level-wise (depth-first), producing deeper, more accurate trees with faster training.

$$F^*(x) = \sum_{t=1}^{T} f_t(x), \quad f_t = \text{CART tree fitted to negative gradient}$$

LightGBM uses **Gradient-based One-Side Sampling (GOSS)**: keep all large-gradient samples + a random subset of small-gradient samples, reducing data size without sacrificing accuracy.

**Key Innovations over XGBoost:**

| Feature | XGBoost | LightGBM |
|---|---|---|
| Tree growth | Level-wise | **Leaf-wise** (lower loss) |
| Histogram binning | Yes (slower) | **Yes (faster, GOSS)** |
| Categorical support | Manual encoding | **Native** |
| Training speed | Fast | **Faster** |
| Memory usage | Moderate | **Lower** |

**Hyperparameters:**

| Parameter | Value | Role |
|---|---|---|
| `n_estimators` | 300 | Number of boosting rounds |
| `max_depth` | 6 | Maximum tree depth |
| `learning_rate` | 0.05 | Shrinkage rate per tree |
| `subsample` | 0.8 | Row sampling fraction |
| `colsample_bytree` | 0.8 | Feature sampling fraction |
| `reg_alpha` | 0.1 | L1 regularisation |
| `reg_lambda` | 1.0 | L2 regularisation |

---

## Results

| Model | RMSE | MAE | R² |
|---|---|---|---|
| **XGBoost** | **142.93** | **63.20** | **0.391** |
| LightGBM | 143.67 | 63.93 | 0.385 |

**Winner: XGBoost**

$$\text{RMSE Improvement (\%)} = \frac{\text{RMSE}_{\text{LightGBM}} - \text{RMSE}_{\text{XGBoost}}}{\text{RMSE}_{\text{LightGBM}}} \times 100 \approx 0.52\%$$

$$R^2 \text{ Gain} = R^2_{\text{XGBoost}} - R^2_{\text{LightGBM}} = +0.006$$

Both models perform comparably. XGBoost edges out LightGBM marginally on all three metrics for this dataset and feature set. Visualisations generated include:
- Actual vs Predicted scatter plots
- Feature importance bar charts
- Residual distribution histograms
- Side-by-side metric comparison bar charts

---

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
lightgbm
```

Install via:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm
```

> **GPU support** is detected automatically at runtime via `nvidia-smi`. If a CUDA-capable GPU is found, XGBoost uses `device="cuda"` and LightGBM uses `device="gpu"`.

---

## How to Run

1. Clone / download this repository.
2. Place all 8 Olist CSV files in the `project/` directory.
3. Open `review2_dynamic_pricing.ipynb` in Jupyter or VS Code.
4. Run all cells top-to-bottom (Kernel → Restart & Run All).

The notebook will automatically merge datasets, engineer features, train both models, and display all evaluation plots and metrics.
