# Dynamic Pricing in E-Commerce — Review 3

**Dataset:** Brazilian Olist E-Commerce (2016–2018)  
**Algorithms:** Ridge Regression | Random Forest | XGBoost | LightGBM  
**Objective:** Predict product prices using linear baselines (Ridge), tree-ensembles (Random Forest), and advanced gradient boosting models (XGBoost, LightGBM) enhanced via Optuna hyperparameter tuning.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Workflow](#workflow)
4. [Feature Engineering](#feature-engineering)
5. [Algorithms & Hyperparameter Tuning](#algorithms)
6. [Results](#results)
7. [Requirements](#requirements)
8. [How to Run](#how-to-run)

---

## Project Overview

This project implements a dynamic pricing pipeline on the Brazilian Olist e-commerce dataset. Review 3 expands on previous reviews by introducing **17 engineered features**, log transformations of the target variable, **Target Encoding** for high-cardinality categories, and performing an exhaustive performance benchmark across **Ridge Regression, Random Forest, XGBoost, and LightGBM**. Advanced models are optimized using **Optuna** for rigorous hyperparameter tuning.

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
| 1 | Import Libraries & Setup |
| 2 | Load & Merge all 8 datasets |
| 3 | Feature Engineering (Review 2 baseline + new features) |
| 4 | Train-Test Split & Target Encoding |
| 5 | Algorithm 1 — Ridge Regression |
| 6 | Algorithm 2 — Random Forest |
| 7 | Algorithm 3 — XGBoost with Optuna |
| 8 | Algorithm 4 — LightGBM with Optuna |
| 9 | Model Comparison & Results |

---

## Feature Engineering

Expanded up to 17 descriptive features spanning delivery, customer satisfaction, dimensions, and categorical aggregates:

| Feature | Description |
|---|---|
| `freight_value` | Shipping cost in R$ |
| `review_score` | Mean customer rating (1-5) |
| `payment_installments` | Max number of payment installments |
| `days_to_deliver` | Actual delivery time in days |
| `purchase_month` | Month of purchase (1–12) |
| `is_weekend` | 1 if ordered on Saturday or Sunday |
| `category_encoded` | **Target-encoded** product category |
| `product_volume` | length * width * height (cm³) |
| `product_weight_g` | Product weight in grams |
| `photo_qty` | Number of product photos |
| `product_name_length` | Length of product description |
| `seller_avg_price` | Mean price per seller |
| `seller_item_count` | Items sold per seller |
| `category_median_price` | Median price boundary of the category |
| `category_price_std` | Category price standard deviation |
| `delivery_delay` | Margin comparing actual vs estimated delivery |
| `freight_ratio` | Proportion of freight vs overall price |

**Target Variable Transform:** $y = \log(1 + \text{price})$ to mitigate heavy right skew. Metrics are mapped back via $\exp(x) - 1$ to calculate `RMSE_BRL`.  
**Train/Test Split:** 80% train / 20% test, `random_state=42`

---

## Algorithms & Hyperparameter Tuning

1. **Ridge Regression (Baseline):** Pipeline combining standardization mapping against an L2 regularized estimator. Excellent generalized baseline model with robustness against multicollinearity.
2. **Random Forest:** Ensemble of 300 non-linear trees configured via geometric feature subdivisions evaluating intrinsic thresholds minimizing sample squared errors.
3. **XGBoost (Extreme Gradient Boosting):** High-performance level-wise sequential tree boosting. Tuned via **Optuna** (`TPESampler` up to 50 trials across a 5-fold cross-validated function) iterating on critical depth, sampling, and $L1$/$L2$ regularisations.
4. **LightGBM:** Microsoft's leaf-wise optimal splitting gradient booster scaling on dense continuous fields leveraging GOSS frameworks. Similarly configured via an intensive Bayesian **Optuna** grid.

---

## Results

Models are evaluated on Log-scaled and re-transformed scales (BRL). The extensive 17-feature dataset enables powerful predictions over prior iterations:

| Model | RMSE (Log) | MAE (Log) | R² | RMSE (BRL) |
|---|---|---|---|---|
| **LightGBM/XGBoost** | ~ 0.39 – 0.42 | ~ 0.28 | **Up to ~ 0.81** | Dependent on exponential mapping tails |
| Random Forest | ~ 0.41 | ~ 0.28 | ~ 0.79 | - |
| Ridge | ~ 0.65 | ~ 0.49 | ~ 0.49 | - |

> *Note: Exact metrics depend on optimal sampled parameters dynamically generated during iterations.*

**Advancements & Insights:**
- Hyperparameter tuning alongside the expanded attribute list **doubles the $R^2$ baseline from ~0.39 to up to ~0.80+**.
- Significant drivers behind cost estimates include raw item volumes natively (`product_volume`, `product_weight_g`) matching logistic `freight_value` thresholds.
- Ensembling (like stacking XGBoost + LightGBM natively) represents practical residual variance optimizations for future scope iterations.

Visualisations generated natively include:
- 2x2 Actual vs Predicted log-scale scatter distributions.
- Feature importance bar charts.
- Residual distributions leveraging bounded KDE mappings.
- Metric bar combinations.

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
optuna
```

Install via:

```bash
pip install -r requirements.txt
```
*(Or install modules directly like `pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm optuna`)*

> **GPU support** is automatically evaluated during LightGBM training checks (via `-L` subprocess to `nvidia-smi`), diverting sequentially toward standard CPU distributions seamlessly if absent.

---

## How to Run

1. Clone / download this repository.
2. Ensure you have the `project/` directory structured correctly with all 8 Olist CSVs alongside the notebook.
3. Open `dynamic_pricing_review3.ipynb` in a Notebook Editor setup (Jupyter or VS Code).
4. Run all cells systematically (Kernel → Restart & Run All).

The notebook establishes transformations seamlessly, models targets per cell dynamically displaying progressive metric updates outputted to console and PNG visualizations spanning directory output loops.
