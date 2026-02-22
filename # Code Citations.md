# Code Citations

## License: MIT
https://github.com/mlouarra/Chicago_Crime/blob/48e85467cf8d124e7e1aa84cd7b83d00f52fa51f/docs/Cahier_des_charges_mlops.tex

```
Here is the complete content for your report sections. Copy these directly into your document.

---

## 3. PROPOSED METHODOLOGY

### 3.1 Overview of the Proposed System

The proposed system is a **Machine Learning-based Dynamic Pricing Framework (MLPF)** designed to predict optimal product prices in e-commerce environments using historical transaction data. Unlike static rule-based pricing systems, the proposed framework leverages supervised learning algorithms to model the complex, non-linear relationships between product characteristics, customer behaviour, logistics features, and observed market prices. The system ingests raw transactional data from multiple sources, processes and engineers meaningful features, and produces price predictions that can guide real-time pricing decisions for online sellers.

The framework is built on the Brazilian Olist E-Commerce dataset comprising over 110,000 delivered orders spanning September 2016 to August 2018. The data pipeline integrates eight relational CSV files into a unified master dataframe, applies feature engineering to derive pricing-relevant signals, and trains two baseline supervised learning models — Linear Regression and Decision Tree Regressor — to benchmark price prediction accuracy. The models are evaluated on a held-out test set and compared using standard regression metrics, establishing a performance baseline that motivates the adoption of advanced ensemble models in subsequent project phases.

---

### 3.2 System Architecture

The system architecture, illustrated in **Figure 1**, follows an end-to-end supervised machine learning pipeline organised into five sequential processing modules. Raw data from the Olist platform enters the system at the Data Ingestion layer, passes through preprocessing and feature engineering stages, and finally reaches the model training and evaluation layer where predictions and performance metrics are generated.

```
┌─────────────────────────────────────────────────────────────────────┐
│                  PROPOSED SYSTEM ARCHITECTURE                       │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │  MODULE 1    │    │  MODULE 2    │    │      MODULE 3         │  │
│  │  Data        │───▶│  Data        │───▶│   Feature             │  │
│  │  Ingestion   │    │  Preprocessing│   │   Engineering         │  │
│  │  (8 CSVs)    │    │  & Merging   │    │   & Selection         │  │
│  └──────────────┘    └──────────────┘    └───────────┬───────────┘  │
│                                                       │             │
│  ┌──────────────┐    ┌──────────────┐                │             │
│  │  MODULE 5    │    │  MODULE 4    │◀───────────────┘             │
│  │  Model       │◀───│  Model       │                              │
│  │  Evaluation  │    │  Training    │                              │
│  │  & Comparison│    │  LR + DT     │                              │
│  └──────────────┘    └──────────────┘                              │
└─────────────────────────────────────────────────────────────────────┘
```
**Figure 1:** System Architecture of the Proposed Machine Learning-based Dynamic Pricing Framework (MLPF).

---

### 3.3 Module 1 — Data Ingestion and Integration

**Module Title:** Multi-Source Data Ingestion and Relational Integration

**Description:**
The first module is responsible for loading, parsing, and relationally joining eight separate CSV files sourced from the Olist Brazilian E-Commerce platform. Each file encapsulates a distinct domain of the e-commerce transaction lifecycle: order metadata, order items, payment records, customer reviews, product attributes, customer profiles, seller profiles, and category translations. The eight datasets are merged through a sequence of left joins on common key fields (`order_id`, `product_id`, `customer_id`, `seller_id`), producing a single denormalised master dataframe that serves as the input to all subsequent modules.

Timestamp fields are parsed into datetime objects to enable temporal feature derivation. Payment records for the same order are aggregated using summation for the total payment value and the mode for payment type. Review records are aggregated per order using the mean review score. Product categories are translated from Portuguese to English using the translation lookup table. After integration, a quality filter is applied to retain only rows corresponding to **delivered orders** with prices in the range (0, 5000] R$.

Let $D = \{d_1, d_2, \ldots, d_8\}$ denote the set of eight source datasets. The resulting master dataframe $M$ is defined as:

$$M = d_1 \bowtie_{order\_id} d_2 \bowtie_{order\_id} d_3' \bowtie_{order\_id} d_4' \bowtie_{product\_id} d_5' \bowtie_{customer\_id} d_6 \bowtie_{seller\_id} d_7 \tag{1}$$

where $d_3'$, $d_4'$, $d_5'$ denote the aggregated payment, review, and translated product tables respectively, and $\bowtie$ denotes a left equi-join operation.

The payment aggregation for order $o$ is:

$$P_{total}(o) = \sum_{i=1}^{n_o} p_i \tag{2}$$

$$P_{installments}(o) = \max_{i=1}^{n_o}(k_i) \tag{3}$$

where $p_i$ is the payment value and $k_i$ is the number of installments for the $i$-th payment record of order $o$, and $n_o$ is the total payment records for order $o$.

The review score aggregation is:

$$R(o) = \frac{1}{m_o} \sum_{j=1}^{m_o} r_j \tag{4}$$

where $r_j$ is the $j$-th review score for order $o$ and $m_o$ is the number of reviews.

The quality filter applied to the master dataframe is:

$$M_{filtered} = \{x \in M \mid status(x) = \text{"delivered"} \wedge 0 < price(x) \leq 5000\} \tag{5}$$

---

### 3.4 Module 2 — Data Preprocessing

**Module Title:** Temporal Parsing, Missing Value Handling, and Data Normalisation

**Description:**
Module 2 processes the raw master dataframe to prepare it for feature engineering. Three key preprocessing operations are performed: timestamp parsing, missing value imputation, and outlier-aware filtering.

Timestamp columns (`order_purchase_timestamp`, `order_delivered_customer_date`, `order_estimated_delivery_date`) are parsed from string representation into Python `datetime` objects using the `errors="coerce"` strategy, which converts unparseable values to `NaT` (Not a Time) rather than raising exceptions. This preserves data integrity while allowing downstream temporal arithmetic.

Missing values in the `delivery_delay` column (arising from undelivered orders already filtered out) are imputed with zero. Rows with missing values in the selected feature set are dropped during the modelling phase. The final filtered dataset retains **109,357 rows** out of 110,194 after removing records with NaN in any model feature.

The imputation strategy for delivery delay is:

$$\delta_{delay}(o) = \begin{cases} t_{delivered}(o) - t_{estimated}(o) & \text{if } t_{delivered}(o) \text{ is known} \\ 0 & \text{otherwise} \end{cases} \tag{6}$$

where $t_{delivered}(o)$ and $t_{estimated}(o)$ are the actual and estimated delivery timestamps for order $o$.

The freight ratio, a normalised measure of shipping cost relative to product price, is computed as:

$$f_{ratio}(o) = \frac{freight\_value(o)}{price(o) + \epsilon} \tag{7}$$

where $\epsilon = 10^{-3}$ is a small constant added to prevent division by zero.

---

### 3.5 Module 3 — Feature Engineering and Selection

**Module Title:** Temporal, Logistic, and Categorical Feature Derivation

**Description:**
Module 3 derives a rich set of explanatory features from the raw merged columns. Features are grouped into three categories: temporal features extracted from timestamps, logistic features derived from delivery information, and encoded categorical features.

**Temporal features** capture the time-based demand patterns known to influence pricing. The purchase month captures seasonality, day-of-week captures weekly demand cycles, and a binary weekend flag distinguishes weekday from weekend purchasing behaviour.

**Logistic features** capture the cost and efficiency of order fulfilment. The actual days to deliver is a proxy for distance and product type. The delivery delay captures fulfilment reliability.

**Categorical encoding** uses Label Encoding to convert the 72-class `category_en` feature into an integer representation. While ordinal encoding introduces implicit ordering, it allows tree-based models (which partition on thresholds) to use this feature effectively.

The temporal features are defined as:

$$month(o) = \text{month}(t_{purchase}(o)) \in \{1, 2, \ldots, 12\} \tag{8}$$

$$dow(o) = \text{dayofweek}(t_{purchase}(o)) \in \{0, 1, \ldots, 6\} \tag{9}$$

$$is\_weekend(o) = \mathbb{1}[dow(o) \in \{5, 6\}] \tag{10}$$

The logistic features are:

$$days\_to\_deliver(o) = \lfloor t_{delivered}(o) - t_{purchase}(o) \rfloor_{days} \tag{11}$$

The label encoding function for category $c_i$ is:

$$\text{encode}(c_i) = \text{rank}(c_i) \in \{0, 1, \ldots, C-1\} \tag{12}$$

where $C = 72$ is the total number of unique product categories and $\text{rank}(c_i)$ is the alphabetical rank of category $c_i$.

The final feature vector for each order $o$ is:

$$\mathbf{x}(o) = [f_{freight}, r_{score}, k_{install}, d_{deliver}, m_{purchase}, w_{end}, c_{encoded}]^T \in \mathbb{R}^7 \tag{13}$$

---

### 3.6 Module 4 — Model Training

**Module Title:** Supervised Regression Model Training — Linear Regression and Decision Tree

**Description:**
Module 4 trains two supervised regression models on the processed feature matrix to predict the target variable `price`. The dataset is partitioned into training (80%) and testing (20%) subsets using a fixed random seed for reproducibility. A helper `evaluate()` function computes three metrics after each model's predictions are obtained.

The train-test split is formalised as:

$$|\mathcal{D}_{train}| = \lfloor 0.8 \cdot N \rfloor, \quad |\mathcal{D}_{test}| = N - |\mathcal{D}_{train}| \tag{14}$$

where $N = 109{,}357$ is the total number of modelling samples.

#### Algorithm 1 — Linear Regression

**Algorithm 1: Ordinary Least Squares Linear Regression**
```
INPUT : Training set {(xᵢ, yᵢ)}ᵢ₌₁ᴺ, feature matrix X ∈ ℝᴺˣ⁷, target y ∈ ℝᴺ
OUTPUT: Coefficient vector β ∈ ℝ⁸ (7 features + intercept)

1. Augment X with a bias column: X̃ ← [1 | X]  (shape N × 8)
2. Compute the OLS closed-form solution:
       β̂ = (X̃ᵀ X̃)⁻¹ X̃ᵀ y
3. For each test sample xᵢ ∈ D_test:
       ŷᵢ ← β̂₀ + Σⱼ β̂ⱼ · xᵢⱼ
4. Compute evaluation metrics RMSE, MAE, R²
5. RETURN β̂, {ŷᵢ}
```

**Explanation:** Linear Regression models the target price as a linear combination of the 7 input features. The optimal weights $\hat{\beta}$ are obtained analytically by minimising the Residual Sum of Squares (RSS):

$$\hat{\beta} = \arg\min_{\beta} \sum_{i=1}^{N} (y_i - \mathbf{x}_i^T \beta)^2 = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y} \tag{15}$$

The predicted price for a new sample $\mathbf{x}^*$ is:

$$\hat{y}^* = \beta_0 + \sum_{j=1}^{7} \beta_j x_j^* \tag{16}$$

---

#### Algorithm 2 — Decision Tree Regressor

**Algorithm 2: Decision Tree Regression with CART (max_depth = 5)**
```
INPUT : Training set {(xᵢ, yᵢ)}, max_depth = 5, min_samples_split = 2
OUTPUT: Trained tree T with leaf predictions

FUNCTION BuildTree(S, depth):
1.  IF depth = max_depth OR |S| < min_samples_split:
        RETURN LeafNode( value = mean({yᵢ : i ∈ S}) )
2.  best_feature ← NULL;  best_threshold ← NULL;  best_MSE ← ∞
3.  FOR each feature j ∈ {1, …, 7}:
        FOR each threshold t in sorted unique values of xⱼ:
            S_L ← {i ∈ S : xᵢⱼ ≤ t}
            S_R ← {i ∈ S : xᵢⱼ > t}
            mse ← (|S_L| · MSE(S_L) + |S_R| · MSE(S_R)) / |S|
            IF mse < best_MSE:
                best_MSE ← mse;  best_feature ← j;  best_threshold ← t
4.  Create InternalNode(feature = best_feature, threshold = best_threshold)
5.  node.left  ← BuildTree(S_L, depth + 1)
6.  node.right ← BuildTree(S_R, depth + 1)
7.  RETURN node

MAIN:
T ← BuildTree(D_train, depth = 0)
FOR each xᵢ ∈ D_test:
    ŷᵢ ← Traverse(T, xᵢ)   // follow splits until leaf, return leaf mean
RETURN T, {ŷᵢ}
```

**Explanation:** The Decision Tree uses the CART (Classification and Regression Trees) algorithm. At each internal node, it selects the feature $j$ and threshold $t$ that minimises the weighted Mean Squared Error across the resulting child partitions:

$$\text{MSE}_{split} = \frac{|S_L|}{|S|} \cdot \text{MSE}(S_L) + \frac{|S_R|}{|S|} \cdot \text{MSE}(S_R) \tag{17}$$

where the MSE of a partition $S_k$ is:

$$\text{MSE}(S_k) = \frac{1}{|S_k|} \sum_{i \in S_k} (y_i - \bar{y}_{S_k})^2, \quad \bar{y}_{S_k} = \frac{1}{|S_k|} \sum_{i \in S_k} y_i \tag{18}$$

The prediction for a test sample is the mean price of all training samples that fall in the same leaf node as that sample. The feature importance $I_j$ for feature $j$ across all nodes $t$ in the tree is:

$$I_j = \sum_{t : \text{split on } j} \frac{|S_t|}{N} \left[ \text{MSE}(S_t) - \frac{|S_{t,L}|}{|S_t|} \text{MSE}(S_{t,L}) - \frac{|S_{t,R}|}{|S_t|} \text{MSE}(S_{t,R}) \right] \tag{19}$$

---

### 3.7 Module 5 — Model Evaluation and Comparison

**Module Title:** Performance Evaluation Using Regression Metrics

**Description:**
Module 5 evaluates and compares both trained models on the held-out test set $\mathcal{D}_{test}$ using three standard regression metrics. All metrics are computed on the same 21,872 test samples to ensure a fair comparison. The three metrics — RMSE, MAE, and R² — capture different aspects of model quality: RMSE penalises large errors more heavily, MAE gives the average absolute deviation in interpretable price units (R$), and R² measures what fraction of price variance the model explains.

The Root Mean Squared Error (RMSE) is:

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} \tag{20}$$

The Mean Absolute Error (MAE) is:

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| \tag{21}$$

The coefficient of determination $R^2$ is:

$$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2} \tag{22}$$

where
```


## License: unknown
https://github.com/oracle/tribuo-site/blob/5fa2491d60dc4181efca3439883e19aa1134aec2/learn/4.0/tutorials/regression-tribuo-v4.html

```
Here is the complete content for your report sections. Copy these directly into your document.

---

## 3. PROPOSED METHODOLOGY

### 3.1 Overview of the Proposed System

The proposed system is a **Machine Learning-based Dynamic Pricing Framework (MLPF)** designed to predict optimal product prices in e-commerce environments using historical transaction data. Unlike static rule-based pricing systems, the proposed framework leverages supervised learning algorithms to model the complex, non-linear relationships between product characteristics, customer behaviour, logistics features, and observed market prices. The system ingests raw transactional data from multiple sources, processes and engineers meaningful features, and produces price predictions that can guide real-time pricing decisions for online sellers.

The framework is built on the Brazilian Olist E-Commerce dataset comprising over 110,000 delivered orders spanning September 2016 to August 2018. The data pipeline integrates eight relational CSV files into a unified master dataframe, applies feature engineering to derive pricing-relevant signals, and trains two baseline supervised learning models — Linear Regression and Decision Tree Regressor — to benchmark price prediction accuracy. The models are evaluated on a held-out test set and compared using standard regression metrics, establishing a performance baseline that motivates the adoption of advanced ensemble models in subsequent project phases.

---

### 3.2 System Architecture

The system architecture, illustrated in **Figure 1**, follows an end-to-end supervised machine learning pipeline organised into five sequential processing modules. Raw data from the Olist platform enters the system at the Data Ingestion layer, passes through preprocessing and feature engineering stages, and finally reaches the model training and evaluation layer where predictions and performance metrics are generated.

```
┌─────────────────────────────────────────────────────────────────────┐
│                  PROPOSED SYSTEM ARCHITECTURE                       │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │  MODULE 1    │    │  MODULE 2    │    │      MODULE 3         │  │
│  │  Data        │───▶│  Data        │───▶│   Feature             │  │
│  │  Ingestion   │    │  Preprocessing│   │   Engineering         │  │
│  │  (8 CSVs)    │    │  & Merging   │    │   & Selection         │  │
│  └──────────────┘    └──────────────┘    └───────────┬───────────┘  │
│                                                       │             │
│  ┌──────────────┐    ┌──────────────┐                │             │
│  │  MODULE 5    │    │  MODULE 4    │◀───────────────┘             │
│  │  Model       │◀───│  Model       │                              │
│  │  Evaluation  │    │  Training    │                              │
│  │  & Comparison│    │  LR + DT     │                              │
│  └──────────────┘    └──────────────┘                              │
└─────────────────────────────────────────────────────────────────────┘
```
**Figure 1:** System Architecture of the Proposed Machine Learning-based Dynamic Pricing Framework (MLPF).

---

### 3.3 Module 1 — Data Ingestion and Integration

**Module Title:** Multi-Source Data Ingestion and Relational Integration

**Description:**
The first module is responsible for loading, parsing, and relationally joining eight separate CSV files sourced from the Olist Brazilian E-Commerce platform. Each file encapsulates a distinct domain of the e-commerce transaction lifecycle: order metadata, order items, payment records, customer reviews, product attributes, customer profiles, seller profiles, and category translations. The eight datasets are merged through a sequence of left joins on common key fields (`order_id`, `product_id`, `customer_id`, `seller_id`), producing a single denormalised master dataframe that serves as the input to all subsequent modules.

Timestamp fields are parsed into datetime objects to enable temporal feature derivation. Payment records for the same order are aggregated using summation for the total payment value and the mode for payment type. Review records are aggregated per order using the mean review score. Product categories are translated from Portuguese to English using the translation lookup table. After integration, a quality filter is applied to retain only rows corresponding to **delivered orders** with prices in the range (0, 5000] R$.

Let $D = \{d_1, d_2, \ldots, d_8\}$ denote the set of eight source datasets. The resulting master dataframe $M$ is defined as:

$$M = d_1 \bowtie_{order\_id} d_2 \bowtie_{order\_id} d_3' \bowtie_{order\_id} d_4' \bowtie_{product\_id} d_5' \bowtie_{customer\_id} d_6 \bowtie_{seller\_id} d_7 \tag{1}$$

where $d_3'$, $d_4'$, $d_5'$ denote the aggregated payment, review, and translated product tables respectively, and $\bowtie$ denotes a left equi-join operation.

The payment aggregation for order $o$ is:

$$P_{total}(o) = \sum_{i=1}^{n_o} p_i \tag{2}$$

$$P_{installments}(o) = \max_{i=1}^{n_o}(k_i) \tag{3}$$

where $p_i$ is the payment value and $k_i$ is the number of installments for the $i$-th payment record of order $o$, and $n_o$ is the total payment records for order $o$.

The review score aggregation is:

$$R(o) = \frac{1}{m_o} \sum_{j=1}^{m_o} r_j \tag{4}$$

where $r_j$ is the $j$-th review score for order $o$ and $m_o$ is the number of reviews.

The quality filter applied to the master dataframe is:

$$M_{filtered} = \{x \in M \mid status(x) = \text{"delivered"} \wedge 0 < price(x) \leq 5000\} \tag{5}$$

---

### 3.4 Module 2 — Data Preprocessing

**Module Title:** Temporal Parsing, Missing Value Handling, and Data Normalisation

**Description:**
Module 2 processes the raw master dataframe to prepare it for feature engineering. Three key preprocessing operations are performed: timestamp parsing, missing value imputation, and outlier-aware filtering.

Timestamp columns (`order_purchase_timestamp`, `order_delivered_customer_date`, `order_estimated_delivery_date`) are parsed from string representation into Python `datetime` objects using the `errors="coerce"` strategy, which converts unparseable values to `NaT` (Not a Time) rather than raising exceptions. This preserves data integrity while allowing downstream temporal arithmetic.

Missing values in the `delivery_delay` column (arising from undelivered orders already filtered out) are imputed with zero. Rows with missing values in the selected feature set are dropped during the modelling phase. The final filtered dataset retains **109,357 rows** out of 110,194 after removing records with NaN in any model feature.

The imputation strategy for delivery delay is:

$$\delta_{delay}(o) = \begin{cases} t_{delivered}(o) - t_{estimated}(o) & \text{if } t_{delivered}(o) \text{ is known} \\ 0 & \text{otherwise} \end{cases} \tag{6}$$

where $t_{delivered}(o)$ and $t_{estimated}(o)$ are the actual and estimated delivery timestamps for order $o$.

The freight ratio, a normalised measure of shipping cost relative to product price, is computed as:

$$f_{ratio}(o) = \frac{freight\_value(o)}{price(o) + \epsilon} \tag{7}$$

where $\epsilon = 10^{-3}$ is a small constant added to prevent division by zero.

---

### 3.5 Module 3 — Feature Engineering and Selection

**Module Title:** Temporal, Logistic, and Categorical Feature Derivation

**Description:**
Module 3 derives a rich set of explanatory features from the raw merged columns. Features are grouped into three categories: temporal features extracted from timestamps, logistic features derived from delivery information, and encoded categorical features.

**Temporal features** capture the time-based demand patterns known to influence pricing. The purchase month captures seasonality, day-of-week captures weekly demand cycles, and a binary weekend flag distinguishes weekday from weekend purchasing behaviour.

**Logistic features** capture the cost and efficiency of order fulfilment. The actual days to deliver is a proxy for distance and product type. The delivery delay captures fulfilment reliability.

**Categorical encoding** uses Label Encoding to convert the 72-class `category_en` feature into an integer representation. While ordinal encoding introduces implicit ordering, it allows tree-based models (which partition on thresholds) to use this feature effectively.

The temporal features are defined as:

$$month(o) = \text{month}(t_{purchase}(o)) \in \{1, 2, \ldots, 12\} \tag{8}$$

$$dow(o) = \text{dayofweek}(t_{purchase}(o)) \in \{0, 1, \ldots, 6\} \tag{9}$$

$$is\_weekend(o) = \mathbb{1}[dow(o) \in \{5, 6\}] \tag{10}$$

The logistic features are:

$$days\_to\_deliver(o) = \lfloor t_{delivered}(o) - t_{purchase}(o) \rfloor_{days} \tag{11}$$

The label encoding function for category $c_i$ is:

$$\text{encode}(c_i) = \text{rank}(c_i) \in \{0, 1, \ldots, C-1\} \tag{12}$$

where $C = 72$ is the total number of unique product categories and $\text{rank}(c_i)$ is the alphabetical rank of category $c_i$.

The final feature vector for each order $o$ is:

$$\mathbf{x}(o) = [f_{freight}, r_{score}, k_{install}, d_{deliver}, m_{purchase}, w_{end}, c_{encoded}]^T \in \mathbb{R}^7 \tag{13}$$

---

### 3.6 Module 4 — Model Training

**Module Title:** Supervised Regression Model Training — Linear Regression and Decision Tree

**Description:**
Module 4 trains two supervised regression models on the processed feature matrix to predict the target variable `price`. The dataset is partitioned into training (80%) and testing (20%) subsets using a fixed random seed for reproducibility. A helper `evaluate()` function computes three metrics after each model's predictions are obtained.

The train-test split is formalised as:

$$|\mathcal{D}_{train}| = \lfloor 0.8 \cdot N \rfloor, \quad |\mathcal{D}_{test}| = N - |\mathcal{D}_{train}| \tag{14}$$

where $N = 109{,}357$ is the total number of modelling samples.

#### Algorithm 1 — Linear Regression

**Algorithm 1: Ordinary Least Squares Linear Regression**
```
INPUT : Training set {(xᵢ, yᵢ)}ᵢ₌₁ᴺ, feature matrix X ∈ ℝᴺˣ⁷, target y ∈ ℝᴺ
OUTPUT: Coefficient vector β ∈ ℝ⁸ (7 features + intercept)

1. Augment X with a bias column: X̃ ← [1 | X]  (shape N × 8)
2. Compute the OLS closed-form solution:
       β̂ = (X̃ᵀ X̃)⁻¹ X̃ᵀ y
3. For each test sample xᵢ ∈ D_test:
       ŷᵢ ← β̂₀ + Σⱼ β̂ⱼ · xᵢⱼ
4. Compute evaluation metrics RMSE, MAE, R²
5. RETURN β̂, {ŷᵢ}
```

**Explanation:** Linear Regression models the target price as a linear combination of the 7 input features. The optimal weights $\hat{\beta}$ are obtained analytically by minimising the Residual Sum of Squares (RSS):

$$\hat{\beta} = \arg\min_{\beta} \sum_{i=1}^{N} (y_i - \mathbf{x}_i^T \beta)^2 = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y} \tag{15}$$

The predicted price for a new sample $\mathbf{x}^*$ is:

$$\hat{y}^* = \beta_0 + \sum_{j=1}^{7} \beta_j x_j^* \tag{16}$$

---

#### Algorithm 2 — Decision Tree Regressor

**Algorithm 2: Decision Tree Regression with CART (max_depth = 5)**
```
INPUT : Training set {(xᵢ, yᵢ)}, max_depth = 5, min_samples_split = 2
OUTPUT: Trained tree T with leaf predictions

FUNCTION BuildTree(S, depth):
1.  IF depth = max_depth OR |S| < min_samples_split:
        RETURN LeafNode( value = mean({yᵢ : i ∈ S}) )
2.  best_feature ← NULL;  best_threshold ← NULL;  best_MSE ← ∞
3.  FOR each feature j ∈ {1, …, 7}:
        FOR each threshold t in sorted unique values of xⱼ:
            S_L ← {i ∈ S : xᵢⱼ ≤ t}
            S_R ← {i ∈ S : xᵢⱼ > t}
            mse ← (|S_L| · MSE(S_L) + |S_R| · MSE(S_R)) / |S|
            IF mse < best_MSE:
                best_MSE ← mse;  best_feature ← j;  best_threshold ← t
4.  Create InternalNode(feature = best_feature, threshold = best_threshold)
5.  node.left  ← BuildTree(S_L, depth + 1)
6.  node.right ← BuildTree(S_R, depth + 1)
7.  RETURN node

MAIN:
T ← BuildTree(D_train, depth = 0)
FOR each xᵢ ∈ D_test:
    ŷᵢ ← Traverse(T, xᵢ)   // follow splits until leaf, return leaf mean
RETURN T, {ŷᵢ}
```

**Explanation:** The Decision Tree uses the CART (Classification and Regression Trees) algorithm. At each internal node, it selects the feature $j$ and threshold $t$ that minimises the weighted Mean Squared Error across the resulting child partitions:

$$\text{MSE}_{split} = \frac{|S_L|}{|S|} \cdot \text{MSE}(S_L) + \frac{|S_R|}{|S|} \cdot \text{MSE}(S_R) \tag{17}$$

where the MSE of a partition $S_k$ is:

$$\text{MSE}(S_k) = \frac{1}{|S_k|} \sum_{i \in S_k} (y_i - \bar{y}_{S_k})^2, \quad \bar{y}_{S_k} = \frac{1}{|S_k|} \sum_{i \in S_k} y_i \tag{18}$$

The prediction for a test sample is the mean price of all training samples that fall in the same leaf node as that sample. The feature importance $I_j$ for feature $j$ across all nodes $t$ in the tree is:

$$I_j = \sum_{t : \text{split on } j} \frac{|S_t|}{N} \left[ \text{MSE}(S_t) - \frac{|S_{t,L}|}{|S_t|} \text{MSE}(S_{t,L}) - \frac{|S_{t,R}|}{|S_t|} \text{MSE}(S_{t,R}) \right] \tag{19}$$

---

### 3.7 Module 5 — Model Evaluation and Comparison

**Module Title:** Performance Evaluation Using Regression Metrics

**Description:**
Module 5 evaluates and compares both trained models on the held-out test set $\mathcal{D}_{test}$ using three standard regression metrics. All metrics are computed on the same 21,872 test samples to ensure a fair comparison. The three metrics — RMSE, MAE, and R² — capture different aspects of model quality: RMSE penalises large errors more heavily, MAE gives the average absolute deviation in interpretable price units (R$), and R² measures what fraction of price variance the model explains.

The Root Mean Squared Error (RMSE) is:

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} \tag{20}$$

The Mean Absolute Error (MAE) is:

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| \tag{21}$$

The coefficient of determination $R^2$ is:

$$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2} \tag{22}$$

where
```


## License: MIT
https://github.com/mlouarra/Chicago_Crime/blob/48e85467cf8d124e7e1aa84cd7b83d00f52fa51f/docs/Cahier_des_charges_mlops.tex

```
Here is the complete content for your report sections. Copy these directly into your document.

---

## 3. PROPOSED METHODOLOGY

### 3.1 Overview of the Proposed System

The proposed system is a **Machine Learning-based Dynamic Pricing Framework (MLPF)** designed to predict optimal product prices in e-commerce environments using historical transaction data. Unlike static rule-based pricing systems, the proposed framework leverages supervised learning algorithms to model the complex, non-linear relationships between product characteristics, customer behaviour, logistics features, and observed market prices. The system ingests raw transactional data from multiple sources, processes and engineers meaningful features, and produces price predictions that can guide real-time pricing decisions for online sellers.

The framework is built on the Brazilian Olist E-Commerce dataset comprising over 110,000 delivered orders spanning September 2016 to August 2018. The data pipeline integrates eight relational CSV files into a unified master dataframe, applies feature engineering to derive pricing-relevant signals, and trains two baseline supervised learning models — Linear Regression and Decision Tree Regressor — to benchmark price prediction accuracy. The models are evaluated on a held-out test set and compared using standard regression metrics, establishing a performance baseline that motivates the adoption of advanced ensemble models in subsequent project phases.

---

### 3.2 System Architecture

The system architecture, illustrated in **Figure 1**, follows an end-to-end supervised machine learning pipeline organised into five sequential processing modules. Raw data from the Olist platform enters the system at the Data Ingestion layer, passes through preprocessing and feature engineering stages, and finally reaches the model training and evaluation layer where predictions and performance metrics are generated.

```
┌─────────────────────────────────────────────────────────────────────┐
│                  PROPOSED SYSTEM ARCHITECTURE                       │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │  MODULE 1    │    │  MODULE 2    │    │      MODULE 3         │  │
│  │  Data        │───▶│  Data        │───▶│   Feature             │  │
│  │  Ingestion   │    │  Preprocessing│   │   Engineering         │  │
│  │  (8 CSVs)    │    │  & Merging   │    │   & Selection         │  │
│  └──────────────┘    └──────────────┘    └───────────┬───────────┘  │
│                                                       │             │
│  ┌──────────────┐    ┌──────────────┐                │             │
│  │  MODULE 5    │    │  MODULE 4    │◀───────────────┘             │
│  │  Model       │◀───│  Model       │                              │
│  │  Evaluation  │    │  Training    │                              │
│  │  & Comparison│    │  LR + DT     │                              │
│  └──────────────┘    └──────────────┘                              │
└─────────────────────────────────────────────────────────────────────┘
```
**Figure 1:** System Architecture of the Proposed Machine Learning-based Dynamic Pricing Framework (MLPF).

---

### 3.3 Module 1 — Data Ingestion and Integration

**Module Title:** Multi-Source Data Ingestion and Relational Integration

**Description:**
The first module is responsible for loading, parsing, and relationally joining eight separate CSV files sourced from the Olist Brazilian E-Commerce platform. Each file encapsulates a distinct domain of the e-commerce transaction lifecycle: order metadata, order items, payment records, customer reviews, product attributes, customer profiles, seller profiles, and category translations. The eight datasets are merged through a sequence of left joins on common key fields (`order_id`, `product_id`, `customer_id`, `seller_id`), producing a single denormalised master dataframe that serves as the input to all subsequent modules.

Timestamp fields are parsed into datetime objects to enable temporal feature derivation. Payment records for the same order are aggregated using summation for the total payment value and the mode for payment type. Review records are aggregated per order using the mean review score. Product categories are translated from Portuguese to English using the translation lookup table. After integration, a quality filter is applied to retain only rows corresponding to **delivered orders** with prices in the range (0, 5000] R$.

Let $D = \{d_1, d_2, \ldots, d_8\}$ denote the set of eight source datasets. The resulting master dataframe $M$ is defined as:

$$M = d_1 \bowtie_{order\_id} d_2 \bowtie_{order\_id} d_3' \bowtie_{order\_id} d_4' \bowtie_{product\_id} d_5' \bowtie_{customer\_id} d_6 \bowtie_{seller\_id} d_7 \tag{1}$$

where $d_3'$, $d_4'$, $d_5'$ denote the aggregated payment, review, and translated product tables respectively, and $\bowtie$ denotes a left equi-join operation.

The payment aggregation for order $o$ is:

$$P_{total}(o) = \sum_{i=1}^{n_o} p_i \tag{2}$$

$$P_{installments}(o) = \max_{i=1}^{n_o}(k_i) \tag{3}$$

where $p_i$ is the payment value and $k_i$ is the number of installments for the $i$-th payment record of order $o$, and $n_o$ is the total payment records for order $o$.

The review score aggregation is:

$$R(o) = \frac{1}{m_o} \sum_{j=1}^{m_o} r_j \tag{4}$$

where $r_j$ is the $j$-th review score for order $o$ and $m_o$ is the number of reviews.

The quality filter applied to the master dataframe is:

$$M_{filtered} = \{x \in M \mid status(x) = \text{"delivered"} \wedge 0 < price(x) \leq 5000\} \tag{5}$$

---

### 3.4 Module 2 — Data Preprocessing

**Module Title:** Temporal Parsing, Missing Value Handling, and Data Normalisation

**Description:**
Module 2 processes the raw master dataframe to prepare it for feature engineering. Three key preprocessing operations are performed: timestamp parsing, missing value imputation, and outlier-aware filtering.

Timestamp columns (`order_purchase_timestamp`, `order_delivered_customer_date`, `order_estimated_delivery_date`) are parsed from string representation into Python `datetime` objects using the `errors="coerce"` strategy, which converts unparseable values to `NaT` (Not a Time) rather than raising exceptions. This preserves data integrity while allowing downstream temporal arithmetic.

Missing values in the `delivery_delay` column (arising from undelivered orders already filtered out) are imputed with zero. Rows with missing values in the selected feature set are dropped during the modelling phase. The final filtered dataset retains **109,357 rows** out of 110,194 after removing records with NaN in any model feature.

The imputation strategy for delivery delay is:

$$\delta_{delay}(o) = \begin{cases} t_{delivered}(o) - t_{estimated}(o) & \text{if } t_{delivered}(o) \text{ is known} \\ 0 & \text{otherwise} \end{cases} \tag{6}$$

where $t_{delivered}(o)$ and $t_{estimated}(o)$ are the actual and estimated delivery timestamps for order $o$.

The freight ratio, a normalised measure of shipping cost relative to product price, is computed as:

$$f_{ratio}(o) = \frac{freight\_value(o)}{price(o) + \epsilon} \tag{7}$$

where $\epsilon = 10^{-3}$ is a small constant added to prevent division by zero.

---

### 3.5 Module 3 — Feature Engineering and Selection

**Module Title:** Temporal, Logistic, and Categorical Feature Derivation

**Description:**
Module 3 derives a rich set of explanatory features from the raw merged columns. Features are grouped into three categories: temporal features extracted from timestamps, logistic features derived from delivery information, and encoded categorical features.

**Temporal features** capture the time-based demand patterns known to influence pricing. The purchase month captures seasonality, day-of-week captures weekly demand cycles, and a binary weekend flag distinguishes weekday from weekend purchasing behaviour.

**Logistic features** capture the cost and efficiency of order fulfilment. The actual days to deliver is a proxy for distance and product type. The delivery delay captures fulfilment reliability.

**Categorical encoding** uses Label Encoding to convert the 72-class `category_en` feature into an integer representation. While ordinal encoding introduces implicit ordering, it allows tree-based models (which partition on thresholds) to use this feature effectively.

The temporal features are defined as:

$$month(o) = \text{month}(t_{purchase}(o)) \in \{1, 2, \ldots, 12\} \tag{8}$$

$$dow(o) = \text{dayofweek}(t_{purchase}(o)) \in \{0, 1, \ldots, 6\} \tag{9}$$

$$is\_weekend(o) = \mathbb{1}[dow(o) \in \{5, 6\}] \tag{10}$$

The logistic features are:

$$days\_to\_deliver(o) = \lfloor t_{delivered}(o) - t_{purchase}(o) \rfloor_{days} \tag{11}$$

The label encoding function for category $c_i$ is:

$$\text{encode}(c_i) = \text{rank}(c_i) \in \{0, 1, \ldots, C-1\} \tag{12}$$

where $C = 72$ is the total number of unique product categories and $\text{rank}(c_i)$ is the alphabetical rank of category $c_i$.

The final feature vector for each order $o$ is:

$$\mathbf{x}(o) = [f_{freight}, r_{score}, k_{install}, d_{deliver}, m_{purchase}, w_{end}, c_{encoded}]^T \in \mathbb{R}^7 \tag{13}$$

---

### 3.6 Module 4 — Model Training

**Module Title:** Supervised Regression Model Training — Linear Regression and Decision Tree

**Description:**
Module 4 trains two supervised regression models on the processed feature matrix to predict the target variable `price`. The dataset is partitioned into training (80%) and testing (20%) subsets using a fixed random seed for reproducibility. A helper `evaluate()` function computes three metrics after each model's predictions are obtained.

The train-test split is formalised as:

$$|\mathcal{D}_{train}| = \lfloor 0.8 \cdot N \rfloor, \quad |\mathcal{D}_{test}| = N - |\mathcal{D}_{train}| \tag{14}$$

where $N = 109{,}357$ is the total number of modelling samples.

#### Algorithm 1 — Linear Regression

**Algorithm 1: Ordinary Least Squares Linear Regression**
```
INPUT : Training set {(xᵢ, yᵢ)}ᵢ₌₁ᴺ, feature matrix X ∈ ℝᴺˣ⁷, target y ∈ ℝᴺ
OUTPUT: Coefficient vector β ∈ ℝ⁸ (7 features + intercept)

1. Augment X with a bias column: X̃ ← [1 | X]  (shape N × 8)
2. Compute the OLS closed-form solution:
       β̂ = (X̃ᵀ X̃)⁻¹ X̃ᵀ y
3. For each test sample xᵢ ∈ D_test:
       ŷᵢ ← β̂₀ + Σⱼ β̂ⱼ · xᵢⱼ
4. Compute evaluation metrics RMSE, MAE, R²
5. RETURN β̂, {ŷᵢ}
```

**Explanation:** Linear Regression models the target price as a linear combination of the 7 input features. The optimal weights $\hat{\beta}$ are obtained analytically by minimising the Residual Sum of Squares (RSS):

$$\hat{\beta} = \arg\min_{\beta} \sum_{i=1}^{N} (y_i - \mathbf{x}_i^T \beta)^2 = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y} \tag{15}$$

The predicted price for a new sample $\mathbf{x}^*$ is:

$$\hat{y}^* = \beta_0 + \sum_{j=1}^{7} \beta_j x_j^* \tag{16}$$

---

#### Algorithm 2 — Decision Tree Regressor

**Algorithm 2: Decision Tree Regression with CART (max_depth = 5)**
```
INPUT : Training set {(xᵢ, yᵢ)}, max_depth = 5, min_samples_split = 2
OUTPUT: Trained tree T with leaf predictions

FUNCTION BuildTree(S, depth):
1.  IF depth = max_depth OR |S| < min_samples_split:
        RETURN LeafNode( value = mean({yᵢ : i ∈ S}) )
2.  best_feature ← NULL;  best_threshold ← NULL;  best_MSE ← ∞
3.  FOR each feature j ∈ {1, …, 7}:
        FOR each threshold t in sorted unique values of xⱼ:
            S_L ← {i ∈ S : xᵢⱼ ≤ t}
            S_R ← {i ∈ S : xᵢⱼ > t}
            mse ← (|S_L| · MSE(S_L) + |S_R| · MSE(S_R)) / |S|
            IF mse < best_MSE:
                best_MSE ← mse;  best_feature ← j;  best_threshold ← t
4.  Create InternalNode(feature = best_feature, threshold = best_threshold)
5.  node.left  ← BuildTree(S_L, depth + 1)
6.  node.right ← BuildTree(S_R, depth + 1)
7.  RETURN node

MAIN:
T ← BuildTree(D_train, depth = 0)
FOR each xᵢ ∈ D_test:
    ŷᵢ ← Traverse(T, xᵢ)   // follow splits until leaf, return leaf mean
RETURN T, {ŷᵢ}
```

**Explanation:** The Decision Tree uses the CART (Classification and Regression Trees) algorithm. At each internal node, it selects the feature $j$ and threshold $t$ that minimises the weighted Mean Squared Error across the resulting child partitions:

$$\text{MSE}_{split} = \frac{|S_L|}{|S|} \cdot \text{MSE}(S_L) + \frac{|S_R|}{|S|} \cdot \text{MSE}(S_R) \tag{17}$$

where the MSE of a partition $S_k$ is:

$$\text{MSE}(S_k) = \frac{1}{|S_k|} \sum_{i \in S_k} (y_i - \bar{y}_{S_k})^2, \quad \bar{y}_{S_k} = \frac{1}{|S_k|} \sum_{i \in S_k} y_i \tag{18}$$

The prediction for a test sample is the mean price of all training samples that fall in the same leaf node as that sample. The feature importance $I_j$ for feature $j$ across all nodes $t$ in the tree is:

$$I_j = \sum_{t : \text{split on } j} \frac{|S_t|}{N} \left[ \text{MSE}(S_t) - \frac{|S_{t,L}|}{|S_t|} \text{MSE}(S_{t,L}) - \frac{|S_{t,R}|}{|S_t|} \text{MSE}(S_{t,R}) \right] \tag{19}$$

---

### 3.7 Module 5 — Model Evaluation and Comparison

**Module Title:** Performance Evaluation Using Regression Metrics

**Description:**
Module 5 evaluates and compares both trained models on the held-out test set $\mathcal{D}_{test}$ using three standard regression metrics. All metrics are computed on the same 21,872 test samples to ensure a fair comparison. The three metrics — RMSE, MAE, and R² — capture different aspects of model quality: RMSE penalises large errors more heavily, MAE gives the average absolute deviation in interpretable price units (R$), and R² measures what fraction of price variance the model explains.

The Root Mean Squared Error (RMSE) is:

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} \tag{20}$$

The Mean Absolute Error (MAE) is:

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| \tag{21}$$

The coefficient of determination $R^2$ is:

$$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2} \tag{22}$$

where
```


## License: unknown
https://github.com/oracle/tribuo-site/blob/5fa2491d60dc4181efca3439883e19aa1134aec2/learn/4.0/tutorials/regression-tribuo-v4.html

```
Here is the complete content for your report sections. Copy these directly into your document.

---

## 3. PROPOSED METHODOLOGY

### 3.1 Overview of the Proposed System

The proposed system is a **Machine Learning-based Dynamic Pricing Framework (MLPF)** designed to predict optimal product prices in e-commerce environments using historical transaction data. Unlike static rule-based pricing systems, the proposed framework leverages supervised learning algorithms to model the complex, non-linear relationships between product characteristics, customer behaviour, logistics features, and observed market prices. The system ingests raw transactional data from multiple sources, processes and engineers meaningful features, and produces price predictions that can guide real-time pricing decisions for online sellers.

The framework is built on the Brazilian Olist E-Commerce dataset comprising over 110,000 delivered orders spanning September 2016 to August 2018. The data pipeline integrates eight relational CSV files into a unified master dataframe, applies feature engineering to derive pricing-relevant signals, and trains two baseline supervised learning models — Linear Regression and Decision Tree Regressor — to benchmark price prediction accuracy. The models are evaluated on a held-out test set and compared using standard regression metrics, establishing a performance baseline that motivates the adoption of advanced ensemble models in subsequent project phases.

---

### 3.2 System Architecture

The system architecture, illustrated in **Figure 1**, follows an end-to-end supervised machine learning pipeline organised into five sequential processing modules. Raw data from the Olist platform enters the system at the Data Ingestion layer, passes through preprocessing and feature engineering stages, and finally reaches the model training and evaluation layer where predictions and performance metrics are generated.

```
┌─────────────────────────────────────────────────────────────────────┐
│                  PROPOSED SYSTEM ARCHITECTURE                       │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │  MODULE 1    │    │  MODULE 2    │    │      MODULE 3         │  │
│  │  Data        │───▶│  Data        │───▶│   Feature             │  │
│  │  Ingestion   │    │  Preprocessing│   │   Engineering         │  │
│  │  (8 CSVs)    │    │  & Merging   │    │   & Selection         │  │
│  └──────────────┘    └──────────────┘    └───────────┬───────────┘  │
│                                                       │             │
│  ┌──────────────┐    ┌──────────────┐                │             │
│  │  MODULE 5    │    │  MODULE 4    │◀───────────────┘             │
│  │  Model       │◀───│  Model       │                              │
│  │  Evaluation  │    │  Training    │                              │
│  │  & Comparison│    │  LR + DT     │                              │
│  └──────────────┘    └──────────────┘                              │
└─────────────────────────────────────────────────────────────────────┘
```
**Figure 1:** System Architecture of the Proposed Machine Learning-based Dynamic Pricing Framework (MLPF).

---

### 3.3 Module 1 — Data Ingestion and Integration

**Module Title:** Multi-Source Data Ingestion and Relational Integration

**Description:**
The first module is responsible for loading, parsing, and relationally joining eight separate CSV files sourced from the Olist Brazilian E-Commerce platform. Each file encapsulates a distinct domain of the e-commerce transaction lifecycle: order metadata, order items, payment records, customer reviews, product attributes, customer profiles, seller profiles, and category translations. The eight datasets are merged through a sequence of left joins on common key fields (`order_id`, `product_id`, `customer_id`, `seller_id`), producing a single denormalised master dataframe that serves as the input to all subsequent modules.

Timestamp fields are parsed into datetime objects to enable temporal feature derivation. Payment records for the same order are aggregated using summation for the total payment value and the mode for payment type. Review records are aggregated per order using the mean review score. Product categories are translated from Portuguese to English using the translation lookup table. After integration, a quality filter is applied to retain only rows corresponding to **delivered orders** with prices in the range (0, 5000] R$.

Let $D = \{d_1, d_2, \ldots, d_8\}$ denote the set of eight source datasets. The resulting master dataframe $M$ is defined as:

$$M = d_1 \bowtie_{order\_id} d_2 \bowtie_{order\_id} d_3' \bowtie_{order\_id} d_4' \bowtie_{product\_id} d_5' \bowtie_{customer\_id} d_6 \bowtie_{seller\_id} d_7 \tag{1}$$

where $d_3'$, $d_4'$, $d_5'$ denote the aggregated payment, review, and translated product tables respectively, and $\bowtie$ denotes a left equi-join operation.

The payment aggregation for order $o$ is:

$$P_{total}(o) = \sum_{i=1}^{n_o} p_i \tag{2}$$

$$P_{installments}(o) = \max_{i=1}^{n_o}(k_i) \tag{3}$$

where $p_i$ is the payment value and $k_i$ is the number of installments for the $i$-th payment record of order $o$, and $n_o$ is the total payment records for order $o$.

The review score aggregation is:

$$R(o) = \frac{1}{m_o} \sum_{j=1}^{m_o} r_j \tag{4}$$

where $r_j$ is the $j$-th review score for order $o$ and $m_o$ is the number of reviews.

The quality filter applied to the master dataframe is:

$$M_{filtered} = \{x \in M \mid status(x) = \text{"delivered"} \wedge 0 < price(x) \leq 5000\} \tag{5}$$

---

### 3.4 Module 2 — Data Preprocessing

**Module Title:** Temporal Parsing, Missing Value Handling, and Data Normalisation

**Description:**
Module 2 processes the raw master dataframe to prepare it for feature engineering. Three key preprocessing operations are performed: timestamp parsing, missing value imputation, and outlier-aware filtering.

Timestamp columns (`order_purchase_timestamp`, `order_delivered_customer_date`, `order_estimated_delivery_date`) are parsed from string representation into Python `datetime` objects using the `errors="coerce"` strategy, which converts unparseable values to `NaT` (Not a Time) rather than raising exceptions. This preserves data integrity while allowing downstream temporal arithmetic.

Missing values in the `delivery_delay` column (arising from undelivered orders already filtered out) are imputed with zero. Rows with missing values in the selected feature set are dropped during the modelling phase. The final filtered dataset retains **109,357 rows** out of 110,194 after removing records with NaN in any model feature.

The imputation strategy for delivery delay is:

$$\delta_{delay}(o) = \begin{cases} t_{delivered}(o) - t_{estimated}(o) & \text{if } t_{delivered}(o) \text{ is known} \\ 0 & \text{otherwise} \end{cases} \tag{6}$$

where $t_{delivered}(o)$ and $t_{estimated}(o)$ are the actual and estimated delivery timestamps for order $o$.

The freight ratio, a normalised measure of shipping cost relative to product price, is computed as:

$$f_{ratio}(o) = \frac{freight\_value(o)}{price(o) + \epsilon} \tag{7}$$

where $\epsilon = 10^{-3}$ is a small constant added to prevent division by zero.

---

### 3.5 Module 3 — Feature Engineering and Selection

**Module Title:** Temporal, Logistic, and Categorical Feature Derivation

**Description:**
Module 3 derives a rich set of explanatory features from the raw merged columns. Features are grouped into three categories: temporal features extracted from timestamps, logistic features derived from delivery information, and encoded categorical features.

**Temporal features** capture the time-based demand patterns known to influence pricing. The purchase month captures seasonality, day-of-week captures weekly demand cycles, and a binary weekend flag distinguishes weekday from weekend purchasing behaviour.

**Logistic features** capture the cost and efficiency of order fulfilment. The actual days to deliver is a proxy for distance and product type. The delivery delay captures fulfilment reliability.

**Categorical encoding** uses Label Encoding to convert the 72-class `category_en` feature into an integer representation. While ordinal encoding introduces implicit ordering, it allows tree-based models (which partition on thresholds) to use this feature effectively.

The temporal features are defined as:

$$month(o) = \text{month}(t_{purchase}(o)) \in \{1, 2, \ldots, 12\} \tag{8}$$

$$dow(o) = \text{dayofweek}(t_{purchase}(o)) \in \{0, 1, \ldots, 6\} \tag{9}$$

$$is\_weekend(o) = \mathbb{1}[dow(o) \in \{5, 6\}] \tag{10}$$

The logistic features are:

$$days\_to\_deliver(o) = \lfloor t_{delivered}(o) - t_{purchase}(o) \rfloor_{days} \tag{11}$$

The label encoding function for category $c_i$ is:

$$\text{encode}(c_i) = \text{rank}(c_i) \in \{0, 1, \ldots, C-1\} \tag{12}$$

where $C = 72$ is the total number of unique product categories and $\text{rank}(c_i)$ is the alphabetical rank of category $c_i$.

The final feature vector for each order $o$ is:

$$\mathbf{x}(o) = [f_{freight}, r_{score}, k_{install}, d_{deliver}, m_{purchase}, w_{end}, c_{encoded}]^T \in \mathbb{R}^7 \tag{13}$$

---

### 3.6 Module 4 — Model Training

**Module Title:** Supervised Regression Model Training — Linear Regression and Decision Tree

**Description:**
Module 4 trains two supervised regression models on the processed feature matrix to predict the target variable `price`. The dataset is partitioned into training (80%) and testing (20%) subsets using a fixed random seed for reproducibility. A helper `evaluate()` function computes three metrics after each model's predictions are obtained.

The train-test split is formalised as:

$$|\mathcal{D}_{train}| = \lfloor 0.8 \cdot N \rfloor, \quad |\mathcal{D}_{test}| = N - |\mathcal{D}_{train}| \tag{14}$$

where $N = 109{,}357$ is the total number of modelling samples.

#### Algorithm 1 — Linear Regression

**Algorithm 1: Ordinary Least Squares Linear Regression**
```
INPUT : Training set {(xᵢ, yᵢ)}ᵢ₌₁ᴺ, feature matrix X ∈ ℝᴺˣ⁷, target y ∈ ℝᴺ
OUTPUT: Coefficient vector β ∈ ℝ⁸ (7 features + intercept)

1. Augment X with a bias column: X̃ ← [1 | X]  (shape N × 8)
2. Compute the OLS closed-form solution:
       β̂ = (X̃ᵀ X̃)⁻¹ X̃ᵀ y
3. For each test sample xᵢ ∈ D_test:
       ŷᵢ ← β̂₀ + Σⱼ β̂ⱼ · xᵢⱼ
4. Compute evaluation metrics RMSE, MAE, R²
5. RETURN β̂, {ŷᵢ}
```

**Explanation:** Linear Regression models the target price as a linear combination of the 7 input features. The optimal weights $\hat{\beta}$ are obtained analytically by minimising the Residual Sum of Squares (RSS):

$$\hat{\beta} = \arg\min_{\beta} \sum_{i=1}^{N} (y_i - \mathbf{x}_i^T \beta)^2 = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y} \tag{15}$$

The predicted price for a new sample $\mathbf{x}^*$ is:

$$\hat{y}^* = \beta_0 + \sum_{j=1}^{7} \beta_j x_j^* \tag{16}$$

---

#### Algorithm 2 — Decision Tree Regressor

**Algorithm 2: Decision Tree Regression with CART (max_depth = 5)**
```
INPUT : Training set {(xᵢ, yᵢ)}, max_depth = 5, min_samples_split = 2
OUTPUT: Trained tree T with leaf predictions

FUNCTION BuildTree(S, depth):
1.  IF depth = max_depth OR |S| < min_samples_split:
        RETURN LeafNode( value = mean({yᵢ : i ∈ S}) )
2.  best_feature ← NULL;  best_threshold ← NULL;  best_MSE ← ∞
3.  FOR each feature j ∈ {1, …, 7}:
        FOR each threshold t in sorted unique values of xⱼ:
            S_L ← {i ∈ S : xᵢⱼ ≤ t}
            S_R ← {i ∈ S : xᵢⱼ > t}
            mse ← (|S_L| · MSE(S_L) + |S_R| · MSE(S_R)) / |S|
            IF mse < best_MSE:
                best_MSE ← mse;  best_feature ← j;  best_threshold ← t
4.  Create InternalNode(feature = best_feature, threshold = best_threshold)
5.  node.left  ← BuildTree(S_L, depth + 1)
6.  node.right ← BuildTree(S_R, depth + 1)
7.  RETURN node

MAIN:
T ← BuildTree(D_train, depth = 0)
FOR each xᵢ ∈ D_test:
    ŷᵢ ← Traverse(T, xᵢ)   // follow splits until leaf, return leaf mean
RETURN T, {ŷᵢ}
```

**Explanation:** The Decision Tree uses the CART (Classification and Regression Trees) algorithm. At each internal node, it selects the feature $j$ and threshold $t$ that minimises the weighted Mean Squared Error across the resulting child partitions:

$$\text{MSE}_{split} = \frac{|S_L|}{|S|} \cdot \text{MSE}(S_L) + \frac{|S_R|}{|S|} \cdot \text{MSE}(S_R) \tag{17}$$

where the MSE of a partition $S_k$ is:

$$\text{MSE}(S_k) = \frac{1}{|S_k|} \sum_{i \in S_k} (y_i - \bar{y}_{S_k})^2, \quad \bar{y}_{S_k} = \frac{1}{|S_k|} \sum_{i \in S_k} y_i \tag{18}$$

The prediction for a test sample is the mean price of all training samples that fall in the same leaf node as that sample. The feature importance $I_j$ for feature $j$ across all nodes $t$ in the tree is:

$$I_j = \sum_{t : \text{split on } j} \frac{|S_t|}{N} \left[ \text{MSE}(S_t) - \frac{|S_{t,L}|}{|S_t|} \text{MSE}(S_{t,L}) - \frac{|S_{t,R}|}{|S_t|} \text{MSE}(S_{t,R}) \right] \tag{19}$$

---

### 3.7 Module 5 — Model Evaluation and Comparison

**Module Title:** Performance Evaluation Using Regression Metrics

**Description:**
Module 5 evaluates and compares both trained models on the held-out test set $\mathcal{D}_{test}$ using three standard regression metrics. All metrics are computed on the same 21,872 test samples to ensure a fair comparison. The three metrics — RMSE, MAE, and R² — capture different aspects of model quality: RMSE penalises large errors more heavily, MAE gives the average absolute deviation in interpretable price units (R$), and R² measures what fraction of price variance the model explains.

The Root Mean Squared Error (RMSE) is:

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} \tag{20}$$

The Mean Absolute Error (MAE) is:

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| \tag{21}$$

The coefficient of determination $R^2$ is:

$$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2} \tag{22}$$

where
```


## License: MIT
https://github.com/mlouarra/Chicago_Crime/blob/48e85467cf8d124e7e1aa84cd7b83d00f52fa51f/docs/Cahier_des_charges_mlops.tex

```
Here is the complete content for your report sections. Copy these directly into your document.

---

## 3. PROPOSED METHODOLOGY

### 3.1 Overview of the Proposed System

The proposed system is a **Machine Learning-based Dynamic Pricing Framework (MLPF)** designed to predict optimal product prices in e-commerce environments using historical transaction data. Unlike static rule-based pricing systems, the proposed framework leverages supervised learning algorithms to model the complex, non-linear relationships between product characteristics, customer behaviour, logistics features, and observed market prices. The system ingests raw transactional data from multiple sources, processes and engineers meaningful features, and produces price predictions that can guide real-time pricing decisions for online sellers.

The framework is built on the Brazilian Olist E-Commerce dataset comprising over 110,000 delivered orders spanning September 2016 to August 2018. The data pipeline integrates eight relational CSV files into a unified master dataframe, applies feature engineering to derive pricing-relevant signals, and trains two baseline supervised learning models — Linear Regression and Decision Tree Regressor — to benchmark price prediction accuracy. The models are evaluated on a held-out test set and compared using standard regression metrics, establishing a performance baseline that motivates the adoption of advanced ensemble models in subsequent project phases.

---

### 3.2 System Architecture

The system architecture, illustrated in **Figure 1**, follows an end-to-end supervised machine learning pipeline organised into five sequential processing modules. Raw data from the Olist platform enters the system at the Data Ingestion layer, passes through preprocessing and feature engineering stages, and finally reaches the model training and evaluation layer where predictions and performance metrics are generated.

```
┌─────────────────────────────────────────────────────────────────────┐
│                  PROPOSED SYSTEM ARCHITECTURE                       │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │  MODULE 1    │    │  MODULE 2    │    │      MODULE 3         │  │
│  │  Data        │───▶│  Data        │───▶│   Feature             │  │
│  │  Ingestion   │    │  Preprocessing│   │   Engineering         │  │
│  │  (8 CSVs)    │    │  & Merging   │    │   & Selection         │  │
│  └──────────────┘    └──────────────┘    └───────────┬───────────┘  │
│                                                       │             │
│  ┌──────────────┐    ┌──────────────┐                │             │
│  │  MODULE 5    │    │  MODULE 4    │◀───────────────┘             │
│  │  Model       │◀───│  Model       │                              │
│  │  Evaluation  │    │  Training    │                              │
│  │  & Comparison│    │  LR + DT     │                              │
│  └──────────────┘    └──────────────┘                              │
└─────────────────────────────────────────────────────────────────────┘
```
**Figure 1:** System Architecture of the Proposed Machine Learning-based Dynamic Pricing Framework (MLPF).

---

### 3.3 Module 1 — Data Ingestion and Integration

**Module Title:** Multi-Source Data Ingestion and Relational Integration

**Description:**
The first module is responsible for loading, parsing, and relationally joining eight separate CSV files sourced from the Olist Brazilian E-Commerce platform. Each file encapsulates a distinct domain of the e-commerce transaction lifecycle: order metadata, order items, payment records, customer reviews, product attributes, customer profiles, seller profiles, and category translations. The eight datasets are merged through a sequence of left joins on common key fields (`order_id`, `product_id`, `customer_id`, `seller_id`), producing a single denormalised master dataframe that serves as the input to all subsequent modules.

Timestamp fields are parsed into datetime objects to enable temporal feature derivation. Payment records for the same order are aggregated using summation for the total payment value and the mode for payment type. Review records are aggregated per order using the mean review score. Product categories are translated from Portuguese to English using the translation lookup table. After integration, a quality filter is applied to retain only rows corresponding to **delivered orders** with prices in the range (0, 5000] R$.

Let $D = \{d_1, d_2, \ldots, d_8\}$ denote the set of eight source datasets. The resulting master dataframe $M$ is defined as:

$$M = d_1 \bowtie_{order\_id} d_2 \bowtie_{order\_id} d_3' \bowtie_{order\_id} d_4' \bowtie_{product\_id} d_5' \bowtie_{customer\_id} d_6 \bowtie_{seller\_id} d_7 \tag{1}$$

where $d_3'$, $d_4'$, $d_5'$ denote the aggregated payment, review, and translated product tables respectively, and $\bowtie$ denotes a left equi-join operation.

The payment aggregation for order $o$ is:

$$P_{total}(o) = \sum_{i=1}^{n_o} p_i \tag{2}$$

$$P_{installments}(o) = \max_{i=1}^{n_o}(k_i) \tag{3}$$

where $p_i$ is the payment value and $k_i$ is the number of installments for the $i$-th payment record of order $o$, and $n_o$ is the total payment records for order $o$.

The review score aggregation is:

$$R(o) = \frac{1}{m_o} \sum_{j=1}^{m_o} r_j \tag{4}$$

where $r_j$ is the $j$-th review score for order $o$ and $m_o$ is the number of reviews.

The quality filter applied to the master dataframe is:

$$M_{filtered} = \{x \in M \mid status(x) = \text{"delivered"} \wedge 0 < price(x) \leq 5000\} \tag{5}$$

---

### 3.4 Module 2 — Data Preprocessing

**Module Title:** Temporal Parsing, Missing Value Handling, and Data Normalisation

**Description:**
Module 2 processes the raw master dataframe to prepare it for feature engineering. Three key preprocessing operations are performed: timestamp parsing, missing value imputation, and outlier-aware filtering.

Timestamp columns (`order_purchase_timestamp`, `order_delivered_customer_date`, `order_estimated_delivery_date`) are parsed from string representation into Python `datetime` objects using the `errors="coerce"` strategy, which converts unparseable values to `NaT` (Not a Time) rather than raising exceptions. This preserves data integrity while allowing downstream temporal arithmetic.

Missing values in the `delivery_delay` column (arising from undelivered orders already filtered out) are imputed with zero. Rows with missing values in the selected feature set are dropped during the modelling phase. The final filtered dataset retains **109,357 rows** out of 110,194 after removing records with NaN in any model feature.

The imputation strategy for delivery delay is:

$$\delta_{delay}(o) = \begin{cases} t_{delivered}(o) - t_{estimated}(o) & \text{if } t_{delivered}(o) \text{ is known} \\ 0 & \text{otherwise} \end{cases} \tag{6}$$

where $t_{delivered}(o)$ and $t_{estimated}(o)$ are the actual and estimated delivery timestamps for order $o$.

The freight ratio, a normalised measure of shipping cost relative to product price, is computed as:

$$f_{ratio}(o) = \frac{freight\_value(o)}{price(o) + \epsilon} \tag{7}$$

where $\epsilon = 10^{-3}$ is a small constant added to prevent division by zero.

---

### 3.5 Module 3 — Feature Engineering and Selection

**Module Title:** Temporal, Logistic, and Categorical Feature Derivation

**Description:**
Module 3 derives a rich set of explanatory features from the raw merged columns. Features are grouped into three categories: temporal features extracted from timestamps, logistic features derived from delivery information, and encoded categorical features.

**Temporal features** capture the time-based demand patterns known to influence pricing. The purchase month captures seasonality, day-of-week captures weekly demand cycles, and a binary weekend flag distinguishes weekday from weekend purchasing behaviour.

**Logistic features** capture the cost and efficiency of order fulfilment. The actual days to deliver is a proxy for distance and product type. The delivery delay captures fulfilment reliability.

**Categorical encoding** uses Label Encoding to convert the 72-class `category_en` feature into an integer representation. While ordinal encoding introduces implicit ordering, it allows tree-based models (which partition on thresholds) to use this feature effectively.

The temporal features are defined as:

$$month(o) = \text{month}(t_{purchase}(o)) \in \{1, 2, \ldots, 12\} \tag{8}$$

$$dow(o) = \text{dayofweek}(t_{purchase}(o)) \in \{0, 1, \ldots, 6\} \tag{9}$$

$$is\_weekend(o) = \mathbb{1}[dow(o) \in \{5, 6\}] \tag{10}$$

The logistic features are:

$$days\_to\_deliver(o) = \lfloor t_{delivered}(o) - t_{purchase}(o) \rfloor_{days} \tag{11}$$

The label encoding function for category $c_i$ is:

$$\text{encode}(c_i) = \text{rank}(c_i) \in \{0, 1, \ldots, C-1\} \tag{12}$$

where $C = 72$ is the total number of unique product categories and $\text{rank}(c_i)$ is the alphabetical rank of category $c_i$.

The final feature vector for each order $o$ is:

$$\mathbf{x}(o) = [f_{freight}, r_{score}, k_{install}, d_{deliver}, m_{purchase}, w_{end}, c_{encoded}]^T \in \mathbb{R}^7 \tag{13}$$

---

### 3.6 Module 4 — Model Training

**Module Title:** Supervised Regression Model Training — Linear Regression and Decision Tree

**Description:**
Module 4 trains two supervised regression models on the processed feature matrix to predict the target variable `price`. The dataset is partitioned into training (80%) and testing (20%) subsets using a fixed random seed for reproducibility. A helper `evaluate()` function computes three metrics after each model's predictions are obtained.

The train-test split is formalised as:

$$|\mathcal{D}_{train}| = \lfloor 0.8 \cdot N \rfloor, \quad |\mathcal{D}_{test}| = N - |\mathcal{D}_{train}| \tag{14}$$

where $N = 109{,}357$ is the total number of modelling samples.

#### Algorithm 1 — Linear Regression

**Algorithm 1: Ordinary Least Squares Linear Regression**
```
INPUT : Training set {(xᵢ, yᵢ)}ᵢ₌₁ᴺ, feature matrix X ∈ ℝᴺˣ⁷, target y ∈ ℝᴺ
OUTPUT: Coefficient vector β ∈ ℝ⁸ (7 features + intercept)

1. Augment X with a bias column: X̃ ← [1 | X]  (shape N × 8)
2. Compute the OLS closed-form solution:
       β̂ = (X̃ᵀ X̃)⁻¹ X̃ᵀ y
3. For each test sample xᵢ ∈ D_test:
       ŷᵢ ← β̂₀ + Σⱼ β̂ⱼ · xᵢⱼ
4. Compute evaluation metrics RMSE, MAE, R²
5. RETURN β̂, {ŷᵢ}
```

**Explanation:** Linear Regression models the target price as a linear combination of the 7 input features. The optimal weights $\hat{\beta}$ are obtained analytically by minimising the Residual Sum of Squares (RSS):

$$\hat{\beta} = \arg\min_{\beta} \sum_{i=1}^{N} (y_i - \mathbf{x}_i^T \beta)^2 = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y} \tag{15}$$

The predicted price for a new sample $\mathbf{x}^*$ is:

$$\hat{y}^* = \beta_0 + \sum_{j=1}^{7} \beta_j x_j^* \tag{16}$$

---

#### Algorithm 2 — Decision Tree Regressor

**Algorithm 2: Decision Tree Regression with CART (max_depth = 5)**
```
INPUT : Training set {(xᵢ, yᵢ)}, max_depth = 5, min_samples_split = 2
OUTPUT: Trained tree T with leaf predictions

FUNCTION BuildTree(S, depth):
1.  IF depth = max_depth OR |S| < min_samples_split:
        RETURN LeafNode( value = mean({yᵢ : i ∈ S}) )
2.  best_feature ← NULL;  best_threshold ← NULL;  best_MSE ← ∞
3.  FOR each feature j ∈ {1, …, 7}:
        FOR each threshold t in sorted unique values of xⱼ:
            S_L ← {i ∈ S : xᵢⱼ ≤ t}
            S_R ← {i ∈ S : xᵢⱼ > t}
            mse ← (|S_L| · MSE(S_L) + |S_R| · MSE(S_R)) / |S|
            IF mse < best_MSE:
                best_MSE ← mse;  best_feature ← j;  best_threshold ← t
4.  Create InternalNode(feature = best_feature, threshold = best_threshold)
5.  node.left  ← BuildTree(S_L, depth + 1)
6.  node.right ← BuildTree(S_R, depth + 1)
7.  RETURN node

MAIN:
T ← BuildTree(D_train, depth = 0)
FOR each xᵢ ∈ D_test:
    ŷᵢ ← Traverse(T, xᵢ)   // follow splits until leaf, return leaf mean
RETURN T, {ŷᵢ}
```

**Explanation:** The Decision Tree uses the CART (Classification and Regression Trees) algorithm. At each internal node, it selects the feature $j$ and threshold $t$ that minimises the weighted Mean Squared Error across the resulting child partitions:

$$\text{MSE}_{split} = \frac{|S_L|}{|S|} \cdot \text{MSE}(S_L) + \frac{|S_R|}{|S|} \cdot \text{MSE}(S_R) \tag{17}$$

where the MSE of a partition $S_k$ is:

$$\text{MSE}(S_k) = \frac{1}{|S_k|} \sum_{i \in S_k} (y_i - \bar{y}_{S_k})^2, \quad \bar{y}_{S_k} = \frac{1}{|S_k|} \sum_{i \in S_k} y_i \tag{18}$$

The prediction for a test sample is the mean price of all training samples that fall in the same leaf node as that sample. The feature importance $I_j$ for feature $j$ across all nodes $t$ in the tree is:

$$I_j = \sum_{t : \text{split on } j} \frac{|S_t|}{N} \left[ \text{MSE}(S_t) - \frac{|S_{t,L}|}{|S_t|} \text{MSE}(S_{t,L}) - \frac{|S_{t,R}|}{|S_t|} \text{MSE}(S_{t,R}) \right] \tag{19}$$

---

### 3.7 Module 5 — Model Evaluation and Comparison

**Module Title:** Performance Evaluation Using Regression Metrics

**Description:**
Module 5 evaluates and compares both trained models on the held-out test set $\mathcal{D}_{test}$ using three standard regression metrics. All metrics are computed on the same 21,872 test samples to ensure a fair comparison. The three metrics — RMSE, MAE, and R² — capture different aspects of model quality: RMSE penalises large errors more heavily, MAE gives the average absolute deviation in interpretable price units (R$), and R² measures what fraction of price variance the model explains.

The Root Mean Squared Error (RMSE) is:

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} \tag{20}$$

The Mean Absolute Error (MAE) is:

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| \tag{21}$$

The coefficient of determination $R^2$ is:

$$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2} \tag{22}$$

where
```


## License: unknown
https://github.com/oracle/tribuo-site/blob/5fa2491d60dc4181efca3439883e19aa1134aec2/learn/4.0/tutorials/regression-tribuo-v4.html

```
Here is the complete content for your report sections. Copy these directly into your document.

---

## 3. PROPOSED METHODOLOGY

### 3.1 Overview of the Proposed System

The proposed system is a **Machine Learning-based Dynamic Pricing Framework (MLPF)** designed to predict optimal product prices in e-commerce environments using historical transaction data. Unlike static rule-based pricing systems, the proposed framework leverages supervised learning algorithms to model the complex, non-linear relationships between product characteristics, customer behaviour, logistics features, and observed market prices. The system ingests raw transactional data from multiple sources, processes and engineers meaningful features, and produces price predictions that can guide real-time pricing decisions for online sellers.

The framework is built on the Brazilian Olist E-Commerce dataset comprising over 110,000 delivered orders spanning September 2016 to August 2018. The data pipeline integrates eight relational CSV files into a unified master dataframe, applies feature engineering to derive pricing-relevant signals, and trains two baseline supervised learning models — Linear Regression and Decision Tree Regressor — to benchmark price prediction accuracy. The models are evaluated on a held-out test set and compared using standard regression metrics, establishing a performance baseline that motivates the adoption of advanced ensemble models in subsequent project phases.

---

### 3.2 System Architecture

The system architecture, illustrated in **Figure 1**, follows an end-to-end supervised machine learning pipeline organised into five sequential processing modules. Raw data from the Olist platform enters the system at the Data Ingestion layer, passes through preprocessing and feature engineering stages, and finally reaches the model training and evaluation layer where predictions and performance metrics are generated.

```
┌─────────────────────────────────────────────────────────────────────┐
│                  PROPOSED SYSTEM ARCHITECTURE                       │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │  MODULE 1    │    │  MODULE 2    │    │      MODULE 3         │  │
│  │  Data        │───▶│  Data        │───▶│   Feature             │  │
│  │  Ingestion   │    │  Preprocessing│   │   Engineering         │  │
│  │  (8 CSVs)    │    │  & Merging   │    │   & Selection         │  │
│  └──────────────┘    └──────────────┘    └───────────┬───────────┘  │
│                                                       │             │
│  ┌──────────────┐    ┌──────────────┐                │             │
│  │  MODULE 5    │    │  MODULE 4    │◀───────────────┘             │
│  │  Model       │◀───│  Model       │                              │
│  │  Evaluation  │    │  Training    │                              │
│  │  & Comparison│    │  LR + DT     │                              │
│  └──────────────┘    └──────────────┘                              │
└─────────────────────────────────────────────────────────────────────┘
```
**Figure 1:** System Architecture of the Proposed Machine Learning-based Dynamic Pricing Framework (MLPF).

---

### 3.3 Module 1 — Data Ingestion and Integration

**Module Title:** Multi-Source Data Ingestion and Relational Integration

**Description:**
The first module is responsible for loading, parsing, and relationally joining eight separate CSV files sourced from the Olist Brazilian E-Commerce platform. Each file encapsulates a distinct domain of the e-commerce transaction lifecycle: order metadata, order items, payment records, customer reviews, product attributes, customer profiles, seller profiles, and category translations. The eight datasets are merged through a sequence of left joins on common key fields (`order_id`, `product_id`, `customer_id`, `seller_id`), producing a single denormalised master dataframe that serves as the input to all subsequent modules.

Timestamp fields are parsed into datetime objects to enable temporal feature derivation. Payment records for the same order are aggregated using summation for the total payment value and the mode for payment type. Review records are aggregated per order using the mean review score. Product categories are translated from Portuguese to English using the translation lookup table. After integration, a quality filter is applied to retain only rows corresponding to **delivered orders** with prices in the range (0, 5000] R$.

Let $D = \{d_1, d_2, \ldots, d_8\}$ denote the set of eight source datasets. The resulting master dataframe $M$ is defined as:

$$M = d_1 \bowtie_{order\_id} d_2 \bowtie_{order\_id} d_3' \bowtie_{order\_id} d_4' \bowtie_{product\_id} d_5' \bowtie_{customer\_id} d_6 \bowtie_{seller\_id} d_7 \tag{1}$$

where $d_3'$, $d_4'$, $d_5'$ denote the aggregated payment, review, and translated product tables respectively, and $\bowtie$ denotes a left equi-join operation.

The payment aggregation for order $o$ is:

$$P_{total}(o) = \sum_{i=1}^{n_o} p_i \tag{2}$$

$$P_{installments}(o) = \max_{i=1}^{n_o}(k_i) \tag{3}$$

where $p_i$ is the payment value and $k_i$ is the number of installments for the $i$-th payment record of order $o$, and $n_o$ is the total payment records for order $o$.

The review score aggregation is:

$$R(o) = \frac{1}{m_o} \sum_{j=1}^{m_o} r_j \tag{4}$$

where $r_j$ is the $j$-th review score for order $o$ and $m_o$ is the number of reviews.

The quality filter applied to the master dataframe is:

$$M_{filtered} = \{x \in M \mid status(x) = \text{"delivered"} \wedge 0 < price(x) \leq 5000\} \tag{5}$$

---

### 3.4 Module 2 — Data Preprocessing

**Module Title:** Temporal Parsing, Missing Value Handling, and Data Normalisation

**Description:**
Module 2 processes the raw master dataframe to prepare it for feature engineering. Three key preprocessing operations are performed: timestamp parsing, missing value imputation, and outlier-aware filtering.

Timestamp columns (`order_purchase_timestamp`, `order_delivered_customer_date`, `order_estimated_delivery_date`) are parsed from string representation into Python `datetime` objects using the `errors="coerce"` strategy, which converts unparseable values to `NaT` (Not a Time) rather than raising exceptions. This preserves data integrity while allowing downstream temporal arithmetic.

Missing values in the `delivery_delay` column (arising from undelivered orders already filtered out) are imputed with zero. Rows with missing values in the selected feature set are dropped during the modelling phase. The final filtered dataset retains **109,357 rows** out of 110,194 after removing records with NaN in any model feature.

The imputation strategy for delivery delay is:

$$\delta_{delay}(o) = \begin{cases} t_{delivered}(o) - t_{estimated}(o) & \text{if } t_{delivered}(o) \text{ is known} \\ 0 & \text{otherwise} \end{cases} \tag{6}$$

where $t_{delivered}(o)$ and $t_{estimated}(o)$ are the actual and estimated delivery timestamps for order $o$.

The freight ratio, a normalised measure of shipping cost relative to product price, is computed as:

$$f_{ratio}(o) = \frac{freight\_value(o)}{price(o) + \epsilon} \tag{7}$$

where $\epsilon = 10^{-3}$ is a small constant added to prevent division by zero.

---

### 3.5 Module 3 — Feature Engineering and Selection

**Module Title:** Temporal, Logistic, and Categorical Feature Derivation

**Description:**
Module 3 derives a rich set of explanatory features from the raw merged columns. Features are grouped into three categories: temporal features extracted from timestamps, logistic features derived from delivery information, and encoded categorical features.

**Temporal features** capture the time-based demand patterns known to influence pricing. The purchase month captures seasonality, day-of-week captures weekly demand cycles, and a binary weekend flag distinguishes weekday from weekend purchasing behaviour.

**Logistic features** capture the cost and efficiency of order fulfilment. The actual days to deliver is a proxy for distance and product type. The delivery delay captures fulfilment reliability.

**Categorical encoding** uses Label Encoding to convert the 72-class `category_en` feature into an integer representation. While ordinal encoding introduces implicit ordering, it allows tree-based models (which partition on thresholds) to use this feature effectively.

The temporal features are defined as:

$$month(o) = \text{month}(t_{purchase}(o)) \in \{1, 2, \ldots, 12\} \tag{8}$$

$$dow(o) = \text{dayofweek}(t_{purchase}(o)) \in \{0, 1, \ldots, 6\} \tag{9}$$

$$is\_weekend(o) = \mathbb{1}[dow(o) \in \{5, 6\}] \tag{10}$$

The logistic features are:

$$days\_to\_deliver(o) = \lfloor t_{delivered}(o) - t_{purchase}(o) \rfloor_{days} \tag{11}$$

The label encoding function for category $c_i$ is:

$$\text{encode}(c_i) = \text{rank}(c_i) \in \{0, 1, \ldots, C-1\} \tag{12}$$

where $C = 72$ is the total number of unique product categories and $\text{rank}(c_i)$ is the alphabetical rank of category $c_i$.

The final feature vector for each order $o$ is:

$$\mathbf{x}(o) = [f_{freight}, r_{score}, k_{install}, d_{deliver}, m_{purchase}, w_{end}, c_{encoded}]^T \in \mathbb{R}^7 \tag{13}$$

---

### 3.6 Module 4 — Model Training

**Module Title:** Supervised Regression Model Training — Linear Regression and Decision Tree

**Description:**
Module 4 trains two supervised regression models on the processed feature matrix to predict the target variable `price`. The dataset is partitioned into training (80%) and testing (20%) subsets using a fixed random seed for reproducibility. A helper `evaluate()` function computes three metrics after each model's predictions are obtained.

The train-test split is formalised as:

$$|\mathcal{D}_{train}| = \lfloor 0.8 \cdot N \rfloor, \quad |\mathcal{D}_{test}| = N - |\mathcal{D}_{train}| \tag{14}$$

where $N = 109{,}357$ is the total number of modelling samples.

#### Algorithm 1 — Linear Regression

**Algorithm 1: Ordinary Least Squares Linear Regression**
```
INPUT : Training set {(xᵢ, yᵢ)}ᵢ₌₁ᴺ, feature matrix X ∈ ℝᴺˣ⁷, target y ∈ ℝᴺ
OUTPUT: Coefficient vector β ∈ ℝ⁸ (7 features + intercept)

1. Augment X with a bias column: X̃ ← [1 | X]  (shape N × 8)
2. Compute the OLS closed-form solution:
       β̂ = (X̃ᵀ X̃)⁻¹ X̃ᵀ y
3. For each test sample xᵢ ∈ D_test:
       ŷᵢ ← β̂₀ + Σⱼ β̂ⱼ · xᵢⱼ
4. Compute evaluation metrics RMSE, MAE, R²
5. RETURN β̂, {ŷᵢ}
```

**Explanation:** Linear Regression models the target price as a linear combination of the 7 input features. The optimal weights $\hat{\beta}$ are obtained analytically by minimising the Residual Sum of Squares (RSS):

$$\hat{\beta} = \arg\min_{\beta} \sum_{i=1}^{N} (y_i - \mathbf{x}_i^T \beta)^2 = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y} \tag{15}$$

The predicted price for a new sample $\mathbf{x}^*$ is:

$$\hat{y}^* = \beta_0 + \sum_{j=1}^{7} \beta_j x_j^* \tag{16}$$

---

#### Algorithm 2 — Decision Tree Regressor

**Algorithm 2: Decision Tree Regression with CART (max_depth = 5)**
```
INPUT : Training set {(xᵢ, yᵢ)}, max_depth = 5, min_samples_split = 2
OUTPUT: Trained tree T with leaf predictions

FUNCTION BuildTree(S, depth):
1.  IF depth = max_depth OR |S| < min_samples_split:
        RETURN LeafNode( value = mean({yᵢ : i ∈ S}) )
2.  best_feature ← NULL;  best_threshold ← NULL;  best_MSE ← ∞
3.  FOR each feature j ∈ {1, …, 7}:
        FOR each threshold t in sorted unique values of xⱼ:
            S_L ← {i ∈ S : xᵢⱼ ≤ t}
            S_R ← {i ∈ S : xᵢⱼ > t}
            mse ← (|S_L| · MSE(S_L) + |S_R| · MSE(S_R)) / |S|
            IF mse < best_MSE:
                best_MSE ← mse;  best_feature ← j;  best_threshold ← t
4.  Create InternalNode(feature = best_feature, threshold = best_threshold)
5.  node.left  ← BuildTree(S_L, depth + 1)
6.  node.right ← BuildTree(S_R, depth + 1)
7.  RETURN node

MAIN:
T ← BuildTree(D_train, depth = 0)
FOR each xᵢ ∈ D_test:
    ŷᵢ ← Traverse(T, xᵢ)   // follow splits until leaf, return leaf mean
RETURN T, {ŷᵢ}
```

**Explanation:** The Decision Tree uses the CART (Classification and Regression Trees) algorithm. At each internal node, it selects the feature $j$ and threshold $t$ that minimises the weighted Mean Squared Error across the resulting child partitions:

$$\text{MSE}_{split} = \frac{|S_L|}{|S|} \cdot \text{MSE}(S_L) + \frac{|S_R|}{|S|} \cdot \text{MSE}(S_R) \tag{17}$$

where the MSE of a partition $S_k$ is:

$$\text{MSE}(S_k) = \frac{1}{|S_k|} \sum_{i \in S_k} (y_i - \bar{y}_{S_k})^2, \quad \bar{y}_{S_k} = \frac{1}{|S_k|} \sum_{i \in S_k} y_i \tag{18}$$

The prediction for a test sample is the mean price of all training samples that fall in the same leaf node as that sample. The feature importance $I_j$ for feature $j$ across all nodes $t$ in the tree is:

$$I_j = \sum_{t : \text{split on } j} \frac{|S_t|}{N} \left[ \text{MSE}(S_t) - \frac{|S_{t,L}|}{|S_t|} \text{MSE}(S_{t,L}) - \frac{|S_{t,R}|}{|S_t|} \text{MSE}(S_{t,R}) \right] \tag{19}$$

---

### 3.7 Module 5 — Model Evaluation and Comparison

**Module Title:** Performance Evaluation Using Regression Metrics

**Description:**
Module 5 evaluates and compares both trained models on the held-out test set $\mathcal{D}_{test}$ using three standard regression metrics. All metrics are computed on the same 21,872 test samples to ensure a fair comparison. The three metrics — RMSE, MAE, and R² — capture different aspects of model quality: RMSE penalises large errors more heavily, MAE gives the average absolute deviation in interpretable price units (R$), and R² measures what fraction of price variance the model explains.

The Root Mean Squared Error (RMSE) is:

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} \tag{20}$$

The Mean Absolute Error (MAE) is:

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| \tag{21}$$

The coefficient of determination $R^2$ is:

$$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2} \tag{22}$$

where
```


## License: MIT
https://github.com/mlouarra/Chicago_Crime/blob/48e85467cf8d124e7e1aa84cd7b83d00f52fa51f/docs/Cahier_des_charges_mlops.tex

```
Here is the complete content for your report sections. Copy these directly into your document.

---

## 3. PROPOSED METHODOLOGY

### 3.1 Overview of the Proposed System

The proposed system is a **Machine Learning-based Dynamic Pricing Framework (MLPF)** designed to predict optimal product prices in e-commerce environments using historical transaction data. Unlike static rule-based pricing systems, the proposed framework leverages supervised learning algorithms to model the complex, non-linear relationships between product characteristics, customer behaviour, logistics features, and observed market prices. The system ingests raw transactional data from multiple sources, processes and engineers meaningful features, and produces price predictions that can guide real-time pricing decisions for online sellers.

The framework is built on the Brazilian Olist E-Commerce dataset comprising over 110,000 delivered orders spanning September 2016 to August 2018. The data pipeline integrates eight relational CSV files into a unified master dataframe, applies feature engineering to derive pricing-relevant signals, and trains two baseline supervised learning models — Linear Regression and Decision Tree Regressor — to benchmark price prediction accuracy. The models are evaluated on a held-out test set and compared using standard regression metrics, establishing a performance baseline that motivates the adoption of advanced ensemble models in subsequent project phases.

---

### 3.2 System Architecture

The system architecture, illustrated in **Figure 1**, follows an end-to-end supervised machine learning pipeline organised into five sequential processing modules. Raw data from the Olist platform enters the system at the Data Ingestion layer, passes through preprocessing and feature engineering stages, and finally reaches the model training and evaluation layer where predictions and performance metrics are generated.

```
┌─────────────────────────────────────────────────────────────────────┐
│                  PROPOSED SYSTEM ARCHITECTURE                       │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │  MODULE 1    │    │  MODULE 2    │    │      MODULE 3         │  │
│  │  Data        │───▶│  Data        │───▶│   Feature             │  │
│  │  Ingestion   │    │  Preprocessing│   │   Engineering         │  │
│  │  (8 CSVs)    │    │  & Merging   │    │   & Selection         │  │
│  └──────────────┘    └──────────────┘    └───────────┬───────────┘  │
│                                                       │             │
│  ┌──────────────┐    ┌──────────────┐                │             │
│  │  MODULE 5    │    │  MODULE 4    │◀───────────────┘             │
│  │  Model       │◀───│  Model       │                              │
│  │  Evaluation  │    │  Training    │                              │
│  │  & Comparison│    │  LR + DT     │                              │
│  └──────────────┘    └──────────────┘                              │
└─────────────────────────────────────────────────────────────────────┘
```
**Figure 1:** System Architecture of the Proposed Machine Learning-based Dynamic Pricing Framework (MLPF).

---

### 3.3 Module 1 — Data Ingestion and Integration

**Module Title:** Multi-Source Data Ingestion and Relational Integration

**Description:**
The first module is responsible for loading, parsing, and relationally joining eight separate CSV files sourced from the Olist Brazilian E-Commerce platform. Each file encapsulates a distinct domain of the e-commerce transaction lifecycle: order metadata, order items, payment records, customer reviews, product attributes, customer profiles, seller profiles, and category translations. The eight datasets are merged through a sequence of left joins on common key fields (`order_id`, `product_id`, `customer_id`, `seller_id`), producing a single denormalised master dataframe that serves as the input to all subsequent modules.

Timestamp fields are parsed into datetime objects to enable temporal feature derivation. Payment records for the same order are aggregated using summation for the total payment value and the mode for payment type. Review records are aggregated per order using the mean review score. Product categories are translated from Portuguese to English using the translation lookup table. After integration, a quality filter is applied to retain only rows corresponding to **delivered orders** with prices in the range (0, 5000] R$.

Let $D = \{d_1, d_2, \ldots, d_8\}$ denote the set of eight source datasets. The resulting master dataframe $M$ is defined as:

$$M = d_1 \bowtie_{order\_id} d_2 \bowtie_{order\_id} d_3' \bowtie_{order\_id} d_4' \bowtie_{product\_id} d_5' \bowtie_{customer\_id} d_6 \bowtie_{seller\_id} d_7 \tag{1}$$

where $d_3'$, $d_4'$, $d_5'$ denote the aggregated payment, review, and translated product tables respectively, and $\bowtie$ denotes a left equi-join operation.

The payment aggregation for order $o$ is:

$$P_{total}(o) = \sum_{i=1}^{n_o} p_i \tag{2}$$

$$P_{installments}(o) = \max_{i=1}^{n_o}(k_i) \tag{3}$$

where $p_i$ is the payment value and $k_i$ is the number of installments for the $i$-th payment record of order $o$, and $n_o$ is the total payment records for order $o$.

The review score aggregation is:

$$R(o) = \frac{1}{m_o} \sum_{j=1}^{m_o} r_j \tag{4}$$

where $r_j$ is the $j$-th review score for order $o$ and $m_o$ is the number of reviews.

The quality filter applied to the master dataframe is:

$$M_{filtered} = \{x \in M \mid status(x) = \text{"delivered"} \wedge 0 < price(x) \leq 5000\} \tag{5}$$

---

### 3.4 Module 2 — Data Preprocessing

**Module Title:** Temporal Parsing, Missing Value Handling, and Data Normalisation

**Description:**
Module 2 processes the raw master dataframe to prepare it for feature engineering. Three key preprocessing operations are performed: timestamp parsing, missing value imputation, and outlier-aware filtering.

Timestamp columns (`order_purchase_timestamp`, `order_delivered_customer_date`, `order_estimated_delivery_date`) are parsed from string representation into Python `datetime` objects using the `errors="coerce"` strategy, which converts unparseable values to `NaT` (Not a Time) rather than raising exceptions. This preserves data integrity while allowing downstream temporal arithmetic.

Missing values in the `delivery_delay` column (arising from undelivered orders already filtered out) are imputed with zero. Rows with missing values in the selected feature set are dropped during the modelling phase. The final filtered dataset retains **109,357 rows** out of 110,194 after removing records with NaN in any model feature.

The imputation strategy for delivery delay is:

$$\delta_{delay}(o) = \begin{cases} t_{delivered}(o) - t_{estimated}(o) & \text{if } t_{delivered}(o) \text{ is known} \\ 0 & \text{otherwise} \end{cases} \tag{6}$$

where $t_{delivered}(o)$ and $t_{estimated}(o)$ are the actual and estimated delivery timestamps for order $o$.

The freight ratio, a normalised measure of shipping cost relative to product price, is computed as:

$$f_{ratio}(o) = \frac{freight\_value(o)}{price(o) + \epsilon} \tag{7}$$

where $\epsilon = 10^{-3}$ is a small constant added to prevent division by zero.

---

### 3.5 Module 3 — Feature Engineering and Selection

**Module Title:** Temporal, Logistic, and Categorical Feature Derivation

**Description:**
Module 3 derives a rich set of explanatory features from the raw merged columns. Features are grouped into three categories: temporal features extracted from timestamps, logistic features derived from delivery information, and encoded categorical features.

**Temporal features** capture the time-based demand patterns known to influence pricing. The purchase month captures seasonality, day-of-week captures weekly demand cycles, and a binary weekend flag distinguishes weekday from weekend purchasing behaviour.

**Logistic features** capture the cost and efficiency of order fulfilment. The actual days to deliver is a proxy for distance and product type. The delivery delay captures fulfilment reliability.

**Categorical encoding** uses Label Encoding to convert the 72-class `category_en` feature into an integer representation. While ordinal encoding introduces implicit ordering, it allows tree-based models (which partition on thresholds) to use this feature effectively.

The temporal features are defined as:

$$month(o) = \text{month}(t_{purchase}(o)) \in \{1, 2, \ldots, 12\} \tag{8}$$

$$dow(o) = \text{dayofweek}(t_{purchase}(o)) \in \{0, 1, \ldots, 6\} \tag{9}$$

$$is\_weekend(o) = \mathbb{1}[dow(o) \in \{5, 6\}] \tag{10}$$

The logistic features are:

$$days\_to\_deliver(o) = \lfloor t_{delivered}(o) - t_{purchase}(o) \rfloor_{days} \tag{11}$$

The label encoding function for category $c_i$ is:

$$\text{encode}(c_i) = \text{rank}(c_i) \in \{0, 1, \ldots, C-1\} \tag{12}$$

where $C = 72$ is the total number of unique product categories and $\text{rank}(c_i)$ is the alphabetical rank of category $c_i$.

The final feature vector for each order $o$ is:

$$\mathbf{x}(o) = [f_{freight}, r_{score}, k_{install}, d_{deliver}, m_{purchase}, w_{end}, c_{encoded}]^T \in \mathbb{R}^7 \tag{13}$$

---

### 3.6 Module 4 — Model Training

**Module Title:** Supervised Regression Model Training — Linear Regression and Decision Tree

**Description:**
Module 4 trains two supervised regression models on the processed feature matrix to predict the target variable `price`. The dataset is partitioned into training (80%) and testing (20%) subsets using a fixed random seed for reproducibility. A helper `evaluate()` function computes three metrics after each model's predictions are obtained.

The train-test split is formalised as:

$$|\mathcal{D}_{train}| = \lfloor 0.8 \cdot N \rfloor, \quad |\mathcal{D}_{test}| = N - |\mathcal{D}_{train}| \tag{14}$$

where $N = 109{,}357$ is the total number of modelling samples.

#### Algorithm 1 — Linear Regression

**Algorithm 1: Ordinary Least Squares Linear Regression**
```
INPUT : Training set {(xᵢ, yᵢ)}ᵢ₌₁ᴺ, feature matrix X ∈ ℝᴺˣ⁷, target y ∈ ℝᴺ
OUTPUT: Coefficient vector β ∈ ℝ⁸ (7 features + intercept)

1. Augment X with a bias column: X̃ ← [1 | X]  (shape N × 8)
2. Compute the OLS closed-form solution:
       β̂ = (X̃ᵀ X̃)⁻¹ X̃ᵀ y
3. For each test sample xᵢ ∈ D_test:
       ŷᵢ ← β̂₀ + Σⱼ β̂ⱼ · xᵢⱼ
4. Compute evaluation metrics RMSE, MAE, R²
5. RETURN β̂, {ŷᵢ}
```

**Explanation:** Linear Regression models the target price as a linear combination of the 7 input features. The optimal weights $\hat{\beta}$ are obtained analytically by minimising the Residual Sum of Squares (RSS):

$$\hat{\beta} = \arg\min_{\beta} \sum_{i=1}^{N} (y_i - \mathbf{x}_i^T \beta)^2 = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y} \tag{15}$$

The predicted price for a new sample $\mathbf{x}^*$ is:

$$\hat{y}^* = \beta_0 + \sum_{j=1}^{7} \beta_j x_j^* \tag{16}$$

---

#### Algorithm 2 — Decision Tree Regressor

**Algorithm 2: Decision Tree Regression with CART (max_depth = 5)**
```
INPUT : Training set {(xᵢ, yᵢ)}, max_depth = 5, min_samples_split = 2
OUTPUT: Trained tree T with leaf predictions

FUNCTION BuildTree(S, depth):
1.  IF depth = max_depth OR |S| < min_samples_split:
        RETURN LeafNode( value = mean({yᵢ : i ∈ S}) )
2.  best_feature ← NULL;  best_threshold ← NULL;  best_MSE ← ∞
3.  FOR each feature j ∈ {1, …, 7}:
        FOR each threshold t in sorted unique values of xⱼ:
            S_L ← {i ∈ S : xᵢⱼ ≤ t}
            S_R ← {i ∈ S : xᵢⱼ > t}
            mse ← (|S_L| · MSE(S_L) + |S_R| · MSE(S_R)) / |S|
            IF mse < best_MSE:
                best_MSE ← mse;  best_feature ← j;  best_threshold ← t
4.  Create InternalNode(feature = best_feature, threshold = best_threshold)
5.  node.left  ← BuildTree(S_L, depth + 1)
6.  node.right ← BuildTree(S_R, depth + 1)
7.  RETURN node

MAIN:
T ← BuildTree(D_train, depth = 0)
FOR each xᵢ ∈ D_test:
    ŷᵢ ← Traverse(T, xᵢ)   // follow splits until leaf, return leaf mean
RETURN T, {ŷᵢ}
```

**Explanation:** The Decision Tree uses the CART (Classification and Regression Trees) algorithm. At each internal node, it selects the feature $j$ and threshold $t$ that minimises the weighted Mean Squared Error across the resulting child partitions:

$$\text{MSE}_{split} = \frac{|S_L|}{|S|} \cdot \text{MSE}(S_L) + \frac{|S_R|}{|S|} \cdot \text{MSE}(S_R) \tag{17}$$

where the MSE of a partition $S_k$ is:

$$\text{MSE}(S_k) = \frac{1}{|S_k|} \sum_{i \in S_k} (y_i - \bar{y}_{S_k})^2, \quad \bar{y}_{S_k} = \frac{1}{|S_k|} \sum_{i \in S_k} y_i \tag{18}$$

The prediction for a test sample is the mean price of all training samples that fall in the same leaf node as that sample. The feature importance $I_j$ for feature $j$ across all nodes $t$ in the tree is:

$$I_j = \sum_{t : \text{split on } j} \frac{|S_t|}{N} \left[ \text{MSE}(S_t) - \frac{|S_{t,L}|}{|S_t|} \text{MSE}(S_{t,L}) - \frac{|S_{t,R}|}{|S_t|} \text{MSE}(S_{t,R}) \right] \tag{19}$$

---

### 3.7 Module 5 — Model Evaluation and Comparison

**Module Title:** Performance Evaluation Using Regression Metrics

**Description:**
Module 5 evaluates and compares both trained models on the held-out test set $\mathcal{D}_{test}$ using three standard regression metrics. All metrics are computed on the same 21,872 test samples to ensure a fair comparison. The three metrics — RMSE, MAE, and R² — capture different aspects of model quality: RMSE penalises large errors more heavily, MAE gives the average absolute deviation in interpretable price units (R$), and R² measures what fraction of price variance the model explains.

The Root Mean Squared Error (RMSE) is:

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} \tag{20}$$

The Mean Absolute Error (MAE) is:

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| \tag{21}$$

The coefficient of determination $R^2$ is:

$$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2} \tag{22}$$

where
```


## License: unknown
https://github.com/oracle/tribuo-site/blob/5fa2491d60dc4181efca3439883e19aa1134aec2/learn/4.0/tutorials/regression-tribuo-v4.html

```
Here is the complete content for your report sections. Copy these directly into your document.

---

## 3. PROPOSED METHODOLOGY

### 3.1 Overview of the Proposed System

The proposed system is a **Machine Learning-based Dynamic Pricing Framework (MLPF)** designed to predict optimal product prices in e-commerce environments using historical transaction data. Unlike static rule-based pricing systems, the proposed framework leverages supervised learning algorithms to model the complex, non-linear relationships between product characteristics, customer behaviour, logistics features, and observed market prices. The system ingests raw transactional data from multiple sources, processes and engineers meaningful features, and produces price predictions that can guide real-time pricing decisions for online sellers.

The framework is built on the Brazilian Olist E-Commerce dataset comprising over 110,000 delivered orders spanning September 2016 to August 2018. The data pipeline integrates eight relational CSV files into a unified master dataframe, applies feature engineering to derive pricing-relevant signals, and trains two baseline supervised learning models — Linear Regression and Decision Tree Regressor — to benchmark price prediction accuracy. The models are evaluated on a held-out test set and compared using standard regression metrics, establishing a performance baseline that motivates the adoption of advanced ensemble models in subsequent project phases.

---

### 3.2 System Architecture

The system architecture, illustrated in **Figure 1**, follows an end-to-end supervised machine learning pipeline organised into five sequential processing modules. Raw data from the Olist platform enters the system at the Data Ingestion layer, passes through preprocessing and feature engineering stages, and finally reaches the model training and evaluation layer where predictions and performance metrics are generated.

```
┌─────────────────────────────────────────────────────────────────────┐
│                  PROPOSED SYSTEM ARCHITECTURE                       │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │  MODULE 1    │    │  MODULE 2    │    │      MODULE 3         │  │
│  │  Data        │───▶│  Data        │───▶│   Feature             │  │
│  │  Ingestion   │    │  Preprocessing│   │   Engineering         │  │
│  │  (8 CSVs)    │    │  & Merging   │    │   & Selection         │  │
│  └──────────────┘    └──────────────┘    └───────────┬───────────┘  │
│                                                       │             │
│  ┌──────────────┐    ┌──────────────┐                │             │
│  │  MODULE 5    │    │  MODULE 4    │◀───────────────┘             │
│  │  Model       │◀───│  Model       │                              │
│  │  Evaluation  │    │  Training    │                              │
│  │  & Comparison│    │  LR + DT     │                              │
│  └──────────────┘    └──────────────┘                              │
└─────────────────────────────────────────────────────────────────────┘
```
**Figure 1:** System Architecture of the Proposed Machine Learning-based Dynamic Pricing Framework (MLPF).

---

### 3.3 Module 1 — Data Ingestion and Integration

**Module Title:** Multi-Source Data Ingestion and Relational Integration

**Description:**
The first module is responsible for loading, parsing, and relationally joining eight separate CSV files sourced from the Olist Brazilian E-Commerce platform. Each file encapsulates a distinct domain of the e-commerce transaction lifecycle: order metadata, order items, payment records, customer reviews, product attributes, customer profiles, seller profiles, and category translations. The eight datasets are merged through a sequence of left joins on common key fields (`order_id`, `product_id`, `customer_id`, `seller_id`), producing a single denormalised master dataframe that serves as the input to all subsequent modules.

Timestamp fields are parsed into datetime objects to enable temporal feature derivation. Payment records for the same order are aggregated using summation for the total payment value and the mode for payment type. Review records are aggregated per order using the mean review score. Product categories are translated from Portuguese to English using the translation lookup table. After integration, a quality filter is applied to retain only rows corresponding to **delivered orders** with prices in the range (0, 5000] R$.

Let $D = \{d_1, d_2, \ldots, d_8\}$ denote the set of eight source datasets. The resulting master dataframe $M$ is defined as:

$$M = d_1 \bowtie_{order\_id} d_2 \bowtie_{order\_id} d_3' \bowtie_{order\_id} d_4' \bowtie_{product\_id} d_5' \bowtie_{customer\_id} d_6 \bowtie_{seller\_id} d_7 \tag{1}$$

where $d_3'$, $d_4'$, $d_5'$ denote the aggregated payment, review, and translated product tables respectively, and $\bowtie$ denotes a left equi-join operation.

The payment aggregation for order $o$ is:

$$P_{total}(o) = \sum_{i=1}^{n_o} p_i \tag{2}$$

$$P_{installments}(o) = \max_{i=1}^{n_o}(k_i) \tag{3}$$

where $p_i$ is the payment value and $k_i$ is the number of installments for the $i$-th payment record of order $o$, and $n_o$ is the total payment records for order $o$.

The review score aggregation is:

$$R(o) = \frac{1}{m_o} \sum_{j=1}^{m_o} r_j \tag{4}$$

where $r_j$ is the $j$-th review score for order $o$ and $m_o$ is the number of reviews.

The quality filter applied to the master dataframe is:

$$M_{filtered} = \{x \in M \mid status(x) = \text{"delivered"} \wedge 0 < price(x) \leq 5000\} \tag{5}$$

---

### 3.4 Module 2 — Data Preprocessing

**Module Title:** Temporal Parsing, Missing Value Handling, and Data Normalisation

**Description:**
Module 2 processes the raw master dataframe to prepare it for feature engineering. Three key preprocessing operations are performed: timestamp parsing, missing value imputation, and outlier-aware filtering.

Timestamp columns (`order_purchase_timestamp`, `order_delivered_customer_date`, `order_estimated_delivery_date`) are parsed from string representation into Python `datetime` objects using the `errors="coerce"` strategy, which converts unparseable values to `NaT` (Not a Time) rather than raising exceptions. This preserves data integrity while allowing downstream temporal arithmetic.

Missing values in the `delivery_delay` column (arising from undelivered orders already filtered out) are imputed with zero. Rows with missing values in the selected feature set are dropped during the modelling phase. The final filtered dataset retains **109,357 rows** out of 110,194 after removing records with NaN in any model feature.

The imputation strategy for delivery delay is:

$$\delta_{delay}(o) = \begin{cases} t_{delivered}(o) - t_{estimated}(o) & \text{if } t_{delivered}(o) \text{ is known} \\ 0 & \text{otherwise} \end{cases} \tag{6}$$

where $t_{delivered}(o)$ and $t_{estimated}(o)$ are the actual and estimated delivery timestamps for order $o$.

The freight ratio, a normalised measure of shipping cost relative to product price, is computed as:

$$f_{ratio}(o) = \frac{freight\_value(o)}{price(o) + \epsilon} \tag{7}$$

where $\epsilon = 10^{-3}$ is a small constant added to prevent division by zero.

---

### 3.5 Module 3 — Feature Engineering and Selection

**Module Title:** Temporal, Logistic, and Categorical Feature Derivation

**Description:**
Module 3 derives a rich set of explanatory features from the raw merged columns. Features are grouped into three categories: temporal features extracted from timestamps, logistic features derived from delivery information, and encoded categorical features.

**Temporal features** capture the time-based demand patterns known to influence pricing. The purchase month captures seasonality, day-of-week captures weekly demand cycles, and a binary weekend flag distinguishes weekday from weekend purchasing behaviour.

**Logistic features** capture the cost and efficiency of order fulfilment. The actual days to deliver is a proxy for distance and product type. The delivery delay captures fulfilment reliability.

**Categorical encoding** uses Label Encoding to convert the 72-class `category_en` feature into an integer representation. While ordinal encoding introduces implicit ordering, it allows tree-based models (which partition on thresholds) to use this feature effectively.

The temporal features are defined as:

$$month(o) = \text{month}(t_{purchase}(o)) \in \{1, 2, \ldots, 12\} \tag{8}$$

$$dow(o) = \text{dayofweek}(t_{purchase}(o)) \in \{0, 1, \ldots, 6\} \tag{9}$$

$$is\_weekend(o) = \mathbb{1}[dow(o) \in \{5, 6\}] \tag{10}$$

The logistic features are:

$$days\_to\_deliver(o) = \lfloor t_{delivered}(o) - t_{purchase}(o) \rfloor_{days} \tag{11}$$

The label encoding function for category $c_i$ is:

$$\text{encode}(c_i) = \text{rank}(c_i) \in \{0, 1, \ldots, C-1\} \tag{12}$$

where $C = 72$ is the total number of unique product categories and $\text{rank}(c_i)$ is the alphabetical rank of category $c_i$.

The final feature vector for each order $o$ is:

$$\mathbf{x}(o) = [f_{freight}, r_{score}, k_{install}, d_{deliver}, m_{purchase}, w_{end}, c_{encoded}]^T \in \mathbb{R}^7 \tag{13}$$

---

### 3.6 Module 4 — Model Training

**Module Title:** Supervised Regression Model Training — Linear Regression and Decision Tree

**Description:**
Module 4 trains two supervised regression models on the processed feature matrix to predict the target variable `price`. The dataset is partitioned into training (80%) and testing (20%) subsets using a fixed random seed for reproducibility. A helper `evaluate()` function computes three metrics after each model's predictions are obtained.

The train-test split is formalised as:

$$|\mathcal{D}_{train}| = \lfloor 0.8 \cdot N \rfloor, \quad |\mathcal{D}_{test}| = N - |\mathcal{D}_{train}| \tag{14}$$

where $N = 109{,}357$ is the total number of modelling samples.

#### Algorithm 1 — Linear Regression

**Algorithm 1: Ordinary Least Squares Linear Regression**
```
INPUT : Training set {(xᵢ, yᵢ)}ᵢ₌₁ᴺ, feature matrix X ∈ ℝᴺˣ⁷, target y ∈ ℝᴺ
OUTPUT: Coefficient vector β ∈ ℝ⁸ (7 features + intercept)

1. Augment X with a bias column: X̃ ← [1 | X]  (shape N × 8)
2. Compute the OLS closed-form solution:
       β̂ = (X̃ᵀ X̃)⁻¹ X̃ᵀ y
3. For each test sample xᵢ ∈ D_test:
       ŷᵢ ← β̂₀ + Σⱼ β̂ⱼ · xᵢⱼ
4. Compute evaluation metrics RMSE, MAE, R²
5. RETURN β̂, {ŷᵢ}
```

**Explanation:** Linear Regression models the target price as a linear combination of the 7 input features. The optimal weights $\hat{\beta}$ are obtained analytically by minimising the Residual Sum of Squares (RSS):

$$\hat{\beta} = \arg\min_{\beta} \sum_{i=1}^{N} (y_i - \mathbf{x}_i^T \beta)^2 = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y} \tag{15}$$

The predicted price for a new sample $\mathbf{x}^*$ is:

$$\hat{y}^* = \beta_0 + \sum_{j=1}^{7} \beta_j x_j^* \tag{16}$$

---

#### Algorithm 2 — Decision Tree Regressor

**Algorithm 2: Decision Tree Regression with CART (max_depth = 5)**
```
INPUT : Training set {(xᵢ, yᵢ)}, max_depth = 5, min_samples_split = 2
OUTPUT: Trained tree T with leaf predictions

FUNCTION BuildTree(S, depth):
1.  IF depth = max_depth OR |S| < min_samples_split:
        RETURN LeafNode( value = mean({yᵢ : i ∈ S}) )
2.  best_feature ← NULL;  best_threshold ← NULL;  best_MSE ← ∞
3.  FOR each feature j ∈ {1, …, 7}:
        FOR each threshold t in sorted unique values of xⱼ:
            S_L ← {i ∈ S : xᵢⱼ ≤ t}
            S_R ← {i ∈ S : xᵢⱼ > t}
            mse ← (|S_L| · MSE(S_L) + |S_R| · MSE(S_R)) / |S|
            IF mse < best_MSE:
                best_MSE ← mse;  best_feature ← j;  best_threshold ← t
4.  Create InternalNode(feature = best_feature, threshold = best_threshold)
5.  node.left  ← BuildTree(S_L, depth + 1)
6.  node.right ← BuildTree(S_R, depth + 1)
7.  RETURN node

MAIN:
T ← BuildTree(D_train, depth = 0)
FOR each xᵢ ∈ D_test:
    ŷᵢ ← Traverse(T, xᵢ)   // follow splits until leaf, return leaf mean
RETURN T, {ŷᵢ}
```

**Explanation:** The Decision Tree uses the CART (Classification and Regression Trees) algorithm. At each internal node, it selects the feature $j$ and threshold $t$ that minimises the weighted Mean Squared Error across the resulting child partitions:

$$\text{MSE}_{split} = \frac{|S_L|}{|S|} \cdot \text{MSE}(S_L) + \frac{|S_R|}{|S|} \cdot \text{MSE}(S_R) \tag{17}$$

where the MSE of a partition $S_k$ is:

$$\text{MSE}(S_k) = \frac{1}{|S_k|} \sum_{i \in S_k} (y_i - \bar{y}_{S_k})^2, \quad \bar{y}_{S_k} = \frac{1}{|S_k|} \sum_{i \in S_k} y_i \tag{18}$$

The prediction for a test sample is the mean price of all training samples that fall in the same leaf node as that sample. The feature importance $I_j$ for feature $j$ across all nodes $t$ in the tree is:

$$I_j = \sum_{t : \text{split on } j} \frac{|S_t|}{N} \left[ \text{MSE}(S_t) - \frac{|S_{t,L}|}{|S_t|} \text{MSE}(S_{t,L}) - \frac{|S_{t,R}|}{|S_t|} \text{MSE}(S_{t,R}) \right] \tag{19}$$

---

### 3.7 Module 5 — Model Evaluation and Comparison

**Module Title:** Performance Evaluation Using Regression Metrics

**Description:**
Module 5 evaluates and compares both trained models on the held-out test set $\mathcal{D}_{test}$ using three standard regression metrics. All metrics are computed on the same 21,872 test samples to ensure a fair comparison. The three metrics — RMSE, MAE, and R² — capture different aspects of model quality: RMSE penalises large errors more heavily, MAE gives the average absolute deviation in interpretable price units (R$), and R² measures what fraction of price variance the model explains.

The Root Mean Squared Error (RMSE) is:

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} \tag{20}$$

The Mean Absolute Error (MAE) is:

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| \tag{21}$$

The coefficient of determination $R^2$ is:

$$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2} \tag{22}$$

where
```


## License: MIT
https://github.com/mlouarra/Chicago_Crime/blob/48e85467cf8d124e7e1aa84cd7b83d00f52fa51f/docs/Cahier_des_charges_mlops.tex

```
Here is the complete content for your report sections. Copy these directly into your document.

---

## 3. PROPOSED METHODOLOGY

### 3.1 Overview of the Proposed System

The proposed system is a **Machine Learning-based Dynamic Pricing Framework (MLPF)** designed to predict optimal product prices in e-commerce environments using historical transaction data. Unlike static rule-based pricing systems, the proposed framework leverages supervised learning algorithms to model the complex, non-linear relationships between product characteristics, customer behaviour, logistics features, and observed market prices. The system ingests raw transactional data from multiple sources, processes and engineers meaningful features, and produces price predictions that can guide real-time pricing decisions for online sellers.

The framework is built on the Brazilian Olist E-Commerce dataset comprising over 110,000 delivered orders spanning September 2016 to August 2018. The data pipeline integrates eight relational CSV files into a unified master dataframe, applies feature engineering to derive pricing-relevant signals, and trains two baseline supervised learning models — Linear Regression and Decision Tree Regressor — to benchmark price prediction accuracy. The models are evaluated on a held-out test set and compared using standard regression metrics, establishing a performance baseline that motivates the adoption of advanced ensemble models in subsequent project phases.

---

### 3.2 System Architecture

The system architecture, illustrated in **Figure 1**, follows an end-to-end supervised machine learning pipeline organised into five sequential processing modules. Raw data from the Olist platform enters the system at the Data Ingestion layer, passes through preprocessing and feature engineering stages, and finally reaches the model training and evaluation layer where predictions and performance metrics are generated.

```
┌─────────────────────────────────────────────────────────────────────┐
│                  PROPOSED SYSTEM ARCHITECTURE                       │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │  MODULE 1    │    │  MODULE 2    │    │      MODULE 3         │  │
│  │  Data        │───▶│  Data        │───▶│   Feature             │  │
│  │  Ingestion   │    │  Preprocessing│   │   Engineering         │  │
│  │  (8 CSVs)    │    │  & Merging   │    │   & Selection         │  │
│  └──────────────┘    └──────────────┘    └───────────┬───────────┘  │
│                                                       │             │
│  ┌──────────────┐    ┌──────────────┐                │             │
│  │  MODULE 5    │    │  MODULE 4    │◀───────────────┘             │
│  │  Model       │◀───│  Model       │                              │
│  │  Evaluation  │    │  Training    │                              │
│  │  & Comparison│    │  LR + DT     │                              │
│  └──────────────┘    └──────────────┘                              │
└─────────────────────────────────────────────────────────────────────┘
```
**Figure 1:** System Architecture of the Proposed Machine Learning-based Dynamic Pricing Framework (MLPF).

---

### 3.3 Module 1 — Data Ingestion and Integration

**Module Title:** Multi-Source Data Ingestion and Relational Integration

**Description:**
The first module is responsible for loading, parsing, and relationally joining eight separate CSV files sourced from the Olist Brazilian E-Commerce platform. Each file encapsulates a distinct domain of the e-commerce transaction lifecycle: order metadata, order items, payment records, customer reviews, product attributes, customer profiles, seller profiles, and category translations. The eight datasets are merged through a sequence of left joins on common key fields (`order_id`, `product_id`, `customer_id`, `seller_id`), producing a single denormalised master dataframe that serves as the input to all subsequent modules.

Timestamp fields are parsed into datetime objects to enable temporal feature derivation. Payment records for the same order are aggregated using summation for the total payment value and the mode for payment type. Review records are aggregated per order using the mean review score. Product categories are translated from Portuguese to English using the translation lookup table. After integration, a quality filter is applied to retain only rows corresponding to **delivered orders** with prices in the range (0, 5000] R$.

Let $D = \{d_1, d_2, \ldots, d_8\}$ denote the set of eight source datasets. The resulting master dataframe $M$ is defined as:

$$M = d_1 \bowtie_{order\_id} d_2 \bowtie_{order\_id} d_3' \bowtie_{order\_id} d_4' \bowtie_{product\_id} d_5' \bowtie_{customer\_id} d_6 \bowtie_{seller\_id} d_7 \tag{1}$$

where $d_3'$, $d_4'$, $d_5'$ denote the aggregated payment, review, and translated product tables respectively, and $\bowtie$ denotes a left equi-join operation.

The payment aggregation for order $o$ is:

$$P_{total}(o) = \sum_{i=1}^{n_o} p_i \tag{2}$$

$$P_{installments}(o) = \max_{i=1}^{n_o}(k_i) \tag{3}$$

where $p_i$ is the payment value and $k_i$ is the number of installments for the $i$-th payment record of order $o$, and $n_o$ is the total payment records for order $o$.

The review score aggregation is:

$$R(o) = \frac{1}{m_o} \sum_{j=1}^{m_o} r_j \tag{4}$$

where $r_j$ is the $j$-th review score for order $o$ and $m_o$ is the number of reviews.

The quality filter applied to the master dataframe is:

$$M_{filtered} = \{x \in M \mid status(x) = \text{"delivered"} \wedge 0 < price(x) \leq 5000\} \tag{5}$$

---

### 3.4 Module 2 — Data Preprocessing

**Module Title:** Temporal Parsing, Missing Value Handling, and Data Normalisation

**Description:**
Module 2 processes the raw master dataframe to prepare it for feature engineering. Three key preprocessing operations are performed: timestamp parsing, missing value imputation, and outlier-aware filtering.

Timestamp columns (`order_purchase_timestamp`, `order_delivered_customer_date`, `order_estimated_delivery_date`) are parsed from string representation into Python `datetime` objects using the `errors="coerce"` strategy, which converts unparseable values to `NaT` (Not a Time) rather than raising exceptions. This preserves data integrity while allowing downstream temporal arithmetic.

Missing values in the `delivery_delay` column (arising from undelivered orders already filtered out) are imputed with zero. Rows with missing values in the selected feature set are dropped during the modelling phase. The final filtered dataset retains **109,357 rows** out of 110,194 after removing records with NaN in any model feature.

The imputation strategy for delivery delay is:

$$\delta_{delay}(o) = \begin{cases} t_{delivered}(o) - t_{estimated}(o) & \text{if } t_{delivered}(o) \text{ is known} \\ 0 & \text{otherwise} \end{cases} \tag{6}$$

where $t_{delivered}(o)$ and $t_{estimated}(o)$ are the actual and estimated delivery timestamps for order $o$.

The freight ratio, a normalised measure of shipping cost relative to product price, is computed as:

$$f_{ratio}(o) = \frac{freight\_value(o)}{price(o) + \epsilon} \tag{7}$$

where $\epsilon = 10^{-3}$ is a small constant added to prevent division by zero.

---

### 3.5 Module 3 — Feature Engineering and Selection

**Module Title:** Temporal, Logistic, and Categorical Feature Derivation

**Description:**
Module 3 derives a rich set of explanatory features from the raw merged columns. Features are grouped into three categories: temporal features extracted from timestamps, logistic features derived from delivery information, and encoded categorical features.

**Temporal features** capture the time-based demand patterns known to influence pricing. The purchase month captures seasonality, day-of-week captures weekly demand cycles, and a binary weekend flag distinguishes weekday from weekend purchasing behaviour.

**Logistic features** capture the cost and efficiency of order fulfilment. The actual days to deliver is a proxy for distance and product type. The delivery delay captures fulfilment reliability.

**Categorical encoding** uses Label Encoding to convert the 72-class `category_en` feature into an integer representation. While ordinal encoding introduces implicit ordering, it allows tree-based models (which partition on thresholds) to use this feature effectively.

The temporal features are defined as:

$$month(o) = \text{month}(t_{purchase}(o)) \in \{1, 2, \ldots, 12\} \tag{8}$$

$$dow(o) = \text{dayofweek}(t_{purchase}(o)) \in \{0, 1, \ldots, 6\} \tag{9}$$

$$is\_weekend(o) = \mathbb{1}[dow(o) \in \{5, 6\}] \tag{10}$$

The logistic features are:

$$days\_to\_deliver(o) = \lfloor t_{delivered}(o) - t_{purchase}(o) \rfloor_{days} \tag{11}$$

The label encoding function for category $c_i$ is:

$$\text{encode}(c_i) = \text{rank}(c_i) \in \{0, 1, \ldots, C-1\} \tag{12}$$

where $C = 72$ is the total number of unique product categories and $\text{rank}(c_i)$ is the alphabetical rank of category $c_i$.

The final feature vector for each order $o$ is:

$$\mathbf{x}(o) = [f_{freight}, r_{score}, k_{install}, d_{deliver}, m_{purchase}, w_{end}, c_{encoded}]^T \in \mathbb{R}^7 \tag{13}$$

---

### 3.6 Module 4 — Model Training

**Module Title:** Supervised Regression Model Training — Linear Regression and Decision Tree

**Description:**
Module 4 trains two supervised regression models on the processed feature matrix to predict the target variable `price`. The dataset is partitioned into training (80%) and testing (20%) subsets using a fixed random seed for reproducibility. A helper `evaluate()` function computes three metrics after each model's predictions are obtained.

The train-test split is formalised as:

$$|\mathcal{D}_{train}| = \lfloor 0.8 \cdot N \rfloor, \quad |\mathcal{D}_{test}| = N - |\mathcal{D}_{train}| \tag{14}$$

where $N = 109{,}357$ is the total number of modelling samples.

#### Algorithm 1 — Linear Regression

**Algorithm 1: Ordinary Least Squares Linear Regression**
```
INPUT : Training set {(xᵢ, yᵢ)}ᵢ₌₁ᴺ, feature matrix X ∈ ℝᴺˣ⁷, target y ∈ ℝᴺ
OUTPUT: Coefficient vector β ∈ ℝ⁸ (7 features + intercept)

1. Augment X with a bias column: X̃ ← [1 | X]  (shape N × 8)
2. Compute the OLS closed-form solution:
       β̂ = (X̃ᵀ X̃)⁻¹ X̃ᵀ y
3. For each test sample xᵢ ∈ D_test:
       ŷᵢ ← β̂₀ + Σⱼ β̂ⱼ · xᵢⱼ
4. Compute evaluation metrics RMSE, MAE, R²
5. RETURN β̂, {ŷᵢ}
```

**Explanation:** Linear Regression models the target price as a linear combination of the 7 input features. The optimal weights $\hat{\beta}$ are obtained analytically by minimising the Residual Sum of Squares (RSS):

$$\hat{\beta} = \arg\min_{\beta} \sum_{i=1}^{N} (y_i - \mathbf{x}_i^T \beta)^2 = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y} \tag{15}$$

The predicted price for a new sample $\mathbf{x}^*$ is:

$$\hat{y}^* = \beta_0 + \sum_{j=1}^{7} \beta_j x_j^* \tag{16}$$

---

#### Algorithm 2 — Decision Tree Regressor

**Algorithm 2: Decision Tree Regression with CART (max_depth = 5)**
```
INPUT : Training set {(xᵢ, yᵢ)}, max_depth = 5, min_samples_split = 2
OUTPUT: Trained tree T with leaf predictions

FUNCTION BuildTree(S, depth):
1.  IF depth = max_depth OR |S| < min_samples_split:
        RETURN LeafNode( value = mean({yᵢ : i ∈ S}) )
2.  best_feature ← NULL;  best_threshold ← NULL;  best_MSE ← ∞
3.  FOR each feature j ∈ {1, …, 7}:
        FOR each threshold t in sorted unique values of xⱼ:
            S_L ← {i ∈ S : xᵢⱼ ≤ t}
            S_R ← {i ∈ S : xᵢⱼ > t}
            mse ← (|S_L| · MSE(S_L) + |S_R| · MSE(S_R)) / |S|
            IF mse < best_MSE:
                best_MSE ← mse;  best_feature ← j;  best_threshold ← t
4.  Create InternalNode(feature = best_feature, threshold = best_threshold)
5.  node.left  ← BuildTree(S_L, depth + 1)
6.  node.right ← BuildTree(S_R, depth + 1)
7.  RETURN node

MAIN:
T ← BuildTree(D_train, depth = 0)
FOR each xᵢ ∈ D_test:
    ŷᵢ ← Traverse(T, xᵢ)   // follow splits until leaf, return leaf mean
RETURN T, {ŷᵢ}
```

**Explanation:** The Decision Tree uses the CART (Classification and Regression Trees) algorithm. At each internal node, it selects the feature $j$ and threshold $t$ that minimises the weighted Mean Squared Error across the resulting child partitions:

$$\text{MSE}_{split} = \frac{|S_L|}{|S|} \cdot \text{MSE}(S_L) + \frac{|S_R|}{|S|} \cdot \text{MSE}(S_R) \tag{17}$$

where the MSE of a partition $S_k$ is:

$$\text{MSE}(S_k) = \frac{1}{|S_k|} \sum_{i \in S_k} (y_i - \bar{y}_{S_k})^2, \quad \bar{y}_{S_k} = \frac{1}{|S_k|} \sum_{i \in S_k} y_i \tag{18}$$

The prediction for a test sample is the mean price of all training samples that fall in the same leaf node as that sample. The feature importance $I_j$ for feature $j$ across all nodes $t$ in the tree is:

$$I_j = \sum_{t : \text{split on } j} \frac{|S_t|}{N} \left[ \text{MSE}(S_t) - \frac{|S_{t,L}|}{|S_t|} \text{MSE}(S_{t,L}) - \frac{|S_{t,R}|}{|S_t|} \text{MSE}(S_{t,R}) \right] \tag{19}$$

---

### 3.7 Module 5 — Model Evaluation and Comparison

**Module Title:** Performance Evaluation Using Regression Metrics

**Description:**
Module 5 evaluates and compares both trained models on the held-out test set $\mathcal{D}_{test}$ using three standard regression metrics. All metrics are computed on the same 21,872 test samples to ensure a fair comparison. The three metrics — RMSE, MAE, and R² — capture different aspects of model quality: RMSE penalises large errors more heavily, MAE gives the average absolute deviation in interpretable price units (R$), and R² measures what fraction of price variance the model explains.

The Root Mean Squared Error (RMSE) is:

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} \tag{20}$$

The Mean Absolute Error (MAE) is:

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| \tag{21}$$

The coefficient of determination $R^2$ is:

$$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2} \tag{22}$$

where
```


## License: unknown
https://github.com/oracle/tribuo-site/blob/5fa2491d60dc4181efca3439883e19aa1134aec2/learn/4.0/tutorials/regression-tribuo-v4.html

```
Here is the complete content for your report sections. Copy these directly into your document.

---

## 3. PROPOSED METHODOLOGY

### 3.1 Overview of the Proposed System

The proposed system is a **Machine Learning-based Dynamic Pricing Framework (MLPF)** designed to predict optimal product prices in e-commerce environments using historical transaction data. Unlike static rule-based pricing systems, the proposed framework leverages supervised learning algorithms to model the complex, non-linear relationships between product characteristics, customer behaviour, logistics features, and observed market prices. The system ingests raw transactional data from multiple sources, processes and engineers meaningful features, and produces price predictions that can guide real-time pricing decisions for online sellers.

The framework is built on the Brazilian Olist E-Commerce dataset comprising over 110,000 delivered orders spanning September 2016 to August 2018. The data pipeline integrates eight relational CSV files into a unified master dataframe, applies feature engineering to derive pricing-relevant signals, and trains two baseline supervised learning models — Linear Regression and Decision Tree Regressor — to benchmark price prediction accuracy. The models are evaluated on a held-out test set and compared using standard regression metrics, establishing a performance baseline that motivates the adoption of advanced ensemble models in subsequent project phases.

---

### 3.2 System Architecture

The system architecture, illustrated in **Figure 1**, follows an end-to-end supervised machine learning pipeline organised into five sequential processing modules. Raw data from the Olist platform enters the system at the Data Ingestion layer, passes through preprocessing and feature engineering stages, and finally reaches the model training and evaluation layer where predictions and performance metrics are generated.

```
┌─────────────────────────────────────────────────────────────────────┐
│                  PROPOSED SYSTEM ARCHITECTURE                       │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │  MODULE 1    │    │  MODULE 2    │    │      MODULE 3         │  │
│  │  Data        │───▶│  Data        │───▶│   Feature             │  │
│  │  Ingestion   │    │  Preprocessing│   │   Engineering         │  │
│  │  (8 CSVs)    │    │  & Merging   │    │   & Selection         │  │
│  └──────────────┘    └──────────────┘    └───────────┬───────────┘  │
│                                                       │             │
│  ┌──────────────┐    ┌──────────────┐                │             │
│  │  MODULE 5    │    │  MODULE 4    │◀───────────────┘             │
│  │  Model       │◀───│  Model       │                              │
│  │  Evaluation  │    │  Training    │                              │
│  │  & Comparison│    │  LR + DT     │                              │
│  └──────────────┘    └──────────────┘                              │
└─────────────────────────────────────────────────────────────────────┘
```
**Figure 1:** System Architecture of the Proposed Machine Learning-based Dynamic Pricing Framework (MLPF).

---

### 3.3 Module 1 — Data Ingestion and Integration

**Module Title:** Multi-Source Data Ingestion and Relational Integration

**Description:**
The first module is responsible for loading, parsing, and relationally joining eight separate CSV files sourced from the Olist Brazilian E-Commerce platform. Each file encapsulates a distinct domain of the e-commerce transaction lifecycle: order metadata, order items, payment records, customer reviews, product attributes, customer profiles, seller profiles, and category translations. The eight datasets are merged through a sequence of left joins on common key fields (`order_id`, `product_id`, `customer_id`, `seller_id`), producing a single denormalised master dataframe that serves as the input to all subsequent modules.

Timestamp fields are parsed into datetime objects to enable temporal feature derivation. Payment records for the same order are aggregated using summation for the total payment value and the mode for payment type. Review records are aggregated per order using the mean review score. Product categories are translated from Portuguese to English using the translation lookup table. After integration, a quality filter is applied to retain only rows corresponding to **delivered orders** with prices in the range (0, 5000] R$.

Let $D = \{d_1, d_2, \ldots, d_8\}$ denote the set of eight source datasets. The resulting master dataframe $M$ is defined as:

$$M = d_1 \bowtie_{order\_id} d_2 \bowtie_{order\_id} d_3' \bowtie_{order\_id} d_4' \bowtie_{product\_id} d_5' \bowtie_{customer\_id} d_6 \bowtie_{seller\_id} d_7 \tag{1}$$

where $d_3'$, $d_4'$, $d_5'$ denote the aggregated payment, review, and translated product tables respectively, and $\bowtie$ denotes a left equi-join operation.

The payment aggregation for order $o$ is:

$$P_{total}(o) = \sum_{i=1}^{n_o} p_i \tag{2}$$

$$P_{installments}(o) = \max_{i=1}^{n_o}(k_i) \tag{3}$$

where $p_i$ is the payment value and $k_i$ is the number of installments for the $i$-th payment record of order $o$, and $n_o$ is the total payment records for order $o$.

The review score aggregation is:

$$R(o) = \frac{1}{m_o} \sum_{j=1}^{m_o} r_j \tag{4}$$

where $r_j$ is the $j$-th review score for order $o$ and $m_o$ is the number of reviews.

The quality filter applied to the master dataframe is:

$$M_{filtered} = \{x \in M \mid status(x) = \text{"delivered"} \wedge 0 < price(x) \leq 5000\} \tag{5}$$

---

### 3.4 Module 2 — Data Preprocessing

**Module Title:** Temporal Parsing, Missing Value Handling, and Data Normalisation

**Description:**
Module 2 processes the raw master dataframe to prepare it for feature engineering. Three key preprocessing operations are performed: timestamp parsing, missing value imputation, and outlier-aware filtering.

Timestamp columns (`order_purchase_timestamp`, `order_delivered_customer_date`, `order_estimated_delivery_date`) are parsed from string representation into Python `datetime` objects using the `errors="coerce"` strategy, which converts unparseable values to `NaT` (Not a Time) rather than raising exceptions. This preserves data integrity while allowing downstream temporal arithmetic.

Missing values in the `delivery_delay` column (arising from undelivered orders already filtered out) are imputed with zero. Rows with missing values in the selected feature set are dropped during the modelling phase. The final filtered dataset retains **109,357 rows** out of 110,194 after removing records with NaN in any model feature.

The imputation strategy for delivery delay is:

$$\delta_{delay}(o) = \begin{cases} t_{delivered}(o) - t_{estimated}(o) & \text{if } t_{delivered}(o) \text{ is known} \\ 0 & \text{otherwise} \end{cases} \tag{6}$$

where $t_{delivered}(o)$ and $t_{estimated}(o)$ are the actual and estimated delivery timestamps for order $o$.

The freight ratio, a normalised measure of shipping cost relative to product price, is computed as:

$$f_{ratio}(o) = \frac{freight\_value(o)}{price(o) + \epsilon} \tag{7}$$

where $\epsilon = 10^{-3}$ is a small constant added to prevent division by zero.

---

### 3.5 Module 3 — Feature Engineering and Selection

**Module Title:** Temporal, Logistic, and Categorical Feature Derivation

**Description:**
Module 3 derives a rich set of explanatory features from the raw merged columns. Features are grouped into three categories: temporal features extracted from timestamps, logistic features derived from delivery information, and encoded categorical features.

**Temporal features** capture the time-based demand patterns known to influence pricing. The purchase month captures seasonality, day-of-week captures weekly demand cycles, and a binary weekend flag distinguishes weekday from weekend purchasing behaviour.

**Logistic features** capture the cost and efficiency of order fulfilment. The actual days to deliver is a proxy for distance and product type. The delivery delay captures fulfilment reliability.

**Categorical encoding** uses Label Encoding to convert the 72-class `category_en` feature into an integer representation. While ordinal encoding introduces implicit ordering, it allows tree-based models (which partition on thresholds) to use this feature effectively.

The temporal features are defined as:

$$month(o) = \text{month}(t_{purchase}(o)) \in \{1, 2, \ldots, 12\} \tag{8}$$

$$dow(o) = \text{dayofweek}(t_{purchase}(o)) \in \{0, 1, \ldots, 6\} \tag{9}$$

$$is\_weekend(o) = \mathbb{1}[dow(o) \in \{5, 6\}] \tag{10}$$

The logistic features are:

$$days\_to\_deliver(o) = \lfloor t_{delivered}(o) - t_{purchase}(o) \rfloor_{days} \tag{11}$$

The label encoding function for category $c_i$ is:

$$\text{encode}(c_i) = \text{rank}(c_i) \in \{0, 1, \ldots, C-1\} \tag{12}$$

where $C = 72$ is the total number of unique product categories and $\text{rank}(c_i)$ is the alphabetical rank of category $c_i$.

The final feature vector for each order $o$ is:

$$\mathbf{x}(o) = [f_{freight}, r_{score}, k_{install}, d_{deliver}, m_{purchase}, w_{end}, c_{encoded}]^T \in \mathbb{R}^7 \tag{13}$$

---

### 3.6 Module 4 — Model Training

**Module Title:** Supervised Regression Model Training — Linear Regression and Decision Tree

**Description:**
Module 4 trains two supervised regression models on the processed feature matrix to predict the target variable `price`. The dataset is partitioned into training (80%) and testing (20%) subsets using a fixed random seed for reproducibility. A helper `evaluate()` function computes three metrics after each model's predictions are obtained.

The train-test split is formalised as:

$$|\mathcal{D}_{train}| = \lfloor 0.8 \cdot N \rfloor, \quad |\mathcal{D}_{test}| = N - |\mathcal{D}_{train}| \tag{14}$$

where $N = 109{,}357$ is the total number of modelling samples.

#### Algorithm 1 — Linear Regression

**Algorithm 1: Ordinary Least Squares Linear Regression**
```
INPUT : Training set {(xᵢ, yᵢ)}ᵢ₌₁ᴺ, feature matrix X ∈ ℝᴺˣ⁷, target y ∈ ℝᴺ
OUTPUT: Coefficient vector β ∈ ℝ⁸ (7 features + intercept)

1. Augment X with a bias column: X̃ ← [1 | X]  (shape N × 8)
2. Compute the OLS closed-form solution:
       β̂ = (X̃ᵀ X̃)⁻¹ X̃ᵀ y
3. For each test sample xᵢ ∈ D_test:
       ŷᵢ ← β̂₀ + Σⱼ β̂ⱼ · xᵢⱼ
4. Compute evaluation metrics RMSE, MAE, R²
5. RETURN β̂, {ŷᵢ}
```

**Explanation:** Linear Regression models the target price as a linear combination of the 7 input features. The optimal weights $\hat{\beta}$ are obtained analytically by minimising the Residual Sum of Squares (RSS):

$$\hat{\beta} = \arg\min_{\beta} \sum_{i=1}^{N} (y_i - \mathbf{x}_i^T \beta)^2 = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y} \tag{15}$$

The predicted price for a new sample $\mathbf{x}^*$ is:

$$\hat{y}^* = \beta_0 + \sum_{j=1}^{7} \beta_j x_j^* \tag{16}$$

---

#### Algorithm 2 — Decision Tree Regressor

**Algorithm 2: Decision Tree Regression with CART (max_depth = 5)**
```
INPUT : Training set {(xᵢ, yᵢ)}, max_depth = 5, min_samples_split = 2
OUTPUT: Trained tree T with leaf predictions

FUNCTION BuildTree(S, depth):
1.  IF depth = max_depth OR |S| < min_samples_split:
        RETURN LeafNode( value = mean({yᵢ : i ∈ S}) )
2.  best_feature ← NULL;  best_threshold ← NULL;  best_MSE ← ∞
3.  FOR each feature j ∈ {1, …, 7}:
        FOR each threshold t in sorted unique values of xⱼ:
            S_L ← {i ∈ S : xᵢⱼ ≤ t}
            S_R ← {i ∈ S : xᵢⱼ > t}
            mse ← (|S_L| · MSE(S_L) + |S_R| · MSE(S_R)) / |S|
            IF mse < best_MSE:
                best_MSE ← mse;  best_feature ← j;  best_threshold ← t
4.  Create InternalNode(feature = best_feature, threshold = best_threshold)
5.  node.left  ← BuildTree(S_L, depth + 1)
6.  node.right ← BuildTree(S_R, depth + 1)
7.  RETURN node

MAIN:
T ← BuildTree(D_train, depth = 0)
FOR each xᵢ ∈ D_test:
    ŷᵢ ← Traverse(T, xᵢ)   // follow splits until leaf, return leaf mean
RETURN T, {ŷᵢ}
```

**Explanation:** The Decision Tree uses the CART (Classification and Regression Trees) algorithm. At each internal node, it selects the feature $j$ and threshold $t$ that minimises the weighted Mean Squared Error across the resulting child partitions:

$$\text{MSE}_{split} = \frac{|S_L|}{|S|} \cdot \text{MSE}(S_L) + \frac{|S_R|}{|S|} \cdot \text{MSE}(S_R) \tag{17}$$

where the MSE of a partition $S_k$ is:

$$\text{MSE}(S_k) = \frac{1}{|S_k|} \sum_{i \in S_k} (y_i - \bar{y}_{S_k})^2, \quad \bar{y}_{S_k} = \frac{1}{|S_k|} \sum_{i \in S_k} y_i \tag{18}$$

The prediction for a test sample is the mean price of all training samples that fall in the same leaf node as that sample. The feature importance $I_j$ for feature $j$ across all nodes $t$ in the tree is:

$$I_j = \sum_{t : \text{split on } j} \frac{|S_t|}{N} \left[ \text{MSE}(S_t) - \frac{|S_{t,L}|}{|S_t|} \text{MSE}(S_{t,L}) - \frac{|S_{t,R}|}{|S_t|} \text{MSE}(S_{t,R}) \right] \tag{19}$$

---

### 3.7 Module 5 — Model Evaluation and Comparison

**Module Title:** Performance Evaluation Using Regression Metrics

**Description:**
Module 5 evaluates and compares both trained models on the held-out test set $\mathcal{D}_{test}$ using three standard regression metrics. All metrics are computed on the same 21,872 test samples to ensure a fair comparison. The three metrics — RMSE, MAE, and R² — capture different aspects of model quality: RMSE penalises large errors more heavily, MAE gives the average absolute deviation in interpretable price units (R$), and R² measures what fraction of price variance the model explains.

The Root Mean Squared Error (RMSE) is:

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} \tag{20}$$

The Mean Absolute Error (MAE) is:

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| \tag{21}$$

The coefficient of determination $R^2$ is:

$$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2} \tag{22}$$

where
```


## License: MIT
https://github.com/mlouarra/Chicago_Crime/blob/48e85467cf8d124e7e1aa84cd7b83d00f52fa51f/docs/Cahier_des_charges_mlops.tex

```
Here is the complete content for your report sections. Copy these directly into your document.

---

## 3. PROPOSED METHODOLOGY

### 3.1 Overview of the Proposed System

The proposed system is a **Machine Learning-based Dynamic Pricing Framework (MLPF)** designed to predict optimal product prices in e-commerce environments using historical transaction data. Unlike static rule-based pricing systems, the proposed framework leverages supervised learning algorithms to model the complex, non-linear relationships between product characteristics, customer behaviour, logistics features, and observed market prices. The system ingests raw transactional data from multiple sources, processes and engineers meaningful features, and produces price predictions that can guide real-time pricing decisions for online sellers.

The framework is built on the Brazilian Olist E-Commerce dataset comprising over 110,000 delivered orders spanning September 2016 to August 2018. The data pipeline integrates eight relational CSV files into a unified master dataframe, applies feature engineering to derive pricing-relevant signals, and trains two baseline supervised learning models — Linear Regression and Decision Tree Regressor — to benchmark price prediction accuracy. The models are evaluated on a held-out test set and compared using standard regression metrics, establishing a performance baseline that motivates the adoption of advanced ensemble models in subsequent project phases.

---

### 3.2 System Architecture

The system architecture, illustrated in **Figure 1**, follows an end-to-end supervised machine learning pipeline organised into five sequential processing modules. Raw data from the Olist platform enters the system at the Data Ingestion layer, passes through preprocessing and feature engineering stages, and finally reaches the model training and evaluation layer where predictions and performance metrics are generated.

```
┌─────────────────────────────────────────────────────────────────────┐
│                  PROPOSED SYSTEM ARCHITECTURE                       │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │  MODULE 1    │    │  MODULE 2    │    │      MODULE 3         │  │
│  │  Data        │───▶│  Data        │───▶│   Feature             │  │
│  │  Ingestion   │    │  Preprocessing│   │   Engineering         │  │
│  │  (8 CSVs)    │    │  & Merging   │    │   & Selection         │  │
│  └──────────────┘    └──────────────┘    └───────────┬───────────┘  │
│                                                       │             │
│  ┌──────────────┐    ┌──────────────┐                │             │
│  │  MODULE 5    │    │  MODULE 4    │◀───────────────┘             │
│  │  Model       │◀───│  Model       │                              │
│  │  Evaluation  │    │  Training    │                              │
│  │  & Comparison│    │  LR + DT     │                              │
│  └──────────────┘    └──────────────┘                              │
└─────────────────────────────────────────────────────────────────────┘
```
**Figure 1:** System Architecture of the Proposed Machine Learning-based Dynamic Pricing Framework (MLPF).

---

### 3.3 Module 1 — Data Ingestion and Integration

**Module Title:** Multi-Source Data Ingestion and Relational Integration

**Description:**
The first module is responsible for loading, parsing, and relationally joining eight separate CSV files sourced from the Olist Brazilian E-Commerce platform. Each file encapsulates a distinct domain of the e-commerce transaction lifecycle: order metadata, order items, payment records, customer reviews, product attributes, customer profiles, seller profiles, and category translations. The eight datasets are merged through a sequence of left joins on common key fields (`order_id`, `product_id`, `customer_id`, `seller_id`), producing a single denormalised master dataframe that serves as the input to all subsequent modules.

Timestamp fields are parsed into datetime objects to enable temporal feature derivation. Payment records for the same order are aggregated using summation for the total payment value and the mode for payment type. Review records are aggregated per order using the mean review score. Product categories are translated from Portuguese to English using the translation lookup table. After integration, a quality filter is applied to retain only rows corresponding to **delivered orders** with prices in the range (0, 5000] R$.

Let $D = \{d_1, d_2, \ldots, d_8\}$ denote the set of eight source datasets. The resulting master dataframe $M$ is defined as:

$$M = d_1 \bowtie_{order\_id} d_2 \bowtie_{order\_id} d_3' \bowtie_{order\_id} d_4' \bowtie_{product\_id} d_5' \bowtie_{customer\_id} d_6 \bowtie_{seller\_id} d_7 \tag{1}$$

where $d_3'$, $d_4'$, $d_5'$ denote the aggregated payment, review, and translated product tables respectively, and $\bowtie$ denotes a left equi-join operation.

The payment aggregation for order $o$ is:

$$P_{total}(o) = \sum_{i=1}^{n_o} p_i \tag{2}$$

$$P_{installments}(o) = \max_{i=1}^{n_o}(k_i) \tag{3}$$

where $p_i$ is the payment value and $k_i$ is the number of installments for the $i$-th payment record of order $o$, and $n_o$ is the total payment records for order $o$.

The review score aggregation is:

$$R(o) = \frac{1}{m_o} \sum_{j=1}^{m_o} r_j \tag{4}$$

where $r_j$ is the $j$-th review score for order $o$ and $m_o$ is the number of reviews.

The quality filter applied to the master dataframe is:

$$M_{filtered} = \{x \in M \mid status(x) = \text{"delivered"} \wedge 0 < price(x) \leq 5000\} \tag{5}$$

---

### 3.4 Module 2 — Data Preprocessing

**Module Title:** Temporal Parsing, Missing Value Handling, and Data Normalisation

**Description:**
Module 2 processes the raw master dataframe to prepare it for feature engineering. Three key preprocessing operations are performed: timestamp parsing, missing value imputation, and outlier-aware filtering.

Timestamp columns (`order_purchase_timestamp`, `order_delivered_customer_date`, `order_estimated_delivery_date`) are parsed from string representation into Python `datetime` objects using the `errors="coerce"` strategy, which converts unparseable values to `NaT` (Not a Time) rather than raising exceptions. This preserves data integrity while allowing downstream temporal arithmetic.

Missing values in the `delivery_delay` column (arising from undelivered orders already filtered out) are imputed with zero. Rows with missing values in the selected feature set are dropped during the modelling phase. The final filtered dataset retains **109,357 rows** out of 110,194 after removing records with NaN in any model feature.

The imputation strategy for delivery delay is:

$$\delta_{delay}(o) = \begin{cases} t_{delivered}(o) - t_{estimated}(o) & \text{if } t_{delivered}(o) \text{ is known} \\ 0 & \text{otherwise} \end{cases} \tag{6}$$

where $t_{delivered}(o)$ and $t_{estimated}(o)$ are the actual and estimated delivery timestamps for order $o$.

The freight ratio, a normalised measure of shipping cost relative to product price, is computed as:

$$f_{ratio}(o) = \frac{freight\_value(o)}{price(o) + \epsilon} \tag{7}$$

where $\epsilon = 10^{-3}$ is a small constant added to prevent division by zero.

---

### 3.5 Module 3 — Feature Engineering and Selection

**Module Title:** Temporal, Logistic, and Categorical Feature Derivation

**Description:**
Module 3 derives a rich set of explanatory features from the raw merged columns. Features are grouped into three categories: temporal features extracted from timestamps, logistic features derived from delivery information, and encoded categorical features.

**Temporal features** capture the time-based demand patterns known to influence pricing. The purchase month captures seasonality, day-of-week captures weekly demand cycles, and a binary weekend flag distinguishes weekday from weekend purchasing behaviour.

**Logistic features** capture the cost and efficiency of order fulfilment. The actual days to deliver is a proxy for distance and product type. The delivery delay captures fulfilment reliability.

**Categorical encoding** uses Label Encoding to convert the 72-class `category_en` feature into an integer representation. While ordinal encoding introduces implicit ordering, it allows tree-based models (which partition on thresholds) to use this feature effectively.

The temporal features are defined as:

$$month(o) = \text{month}(t_{purchase}(o)) \in \{1, 2, \ldots, 12\} \tag{8}$$

$$dow(o) = \text{dayofweek}(t_{purchase}(o)) \in \{0, 1, \ldots, 6\} \tag{9}$$

$$is\_weekend(o) = \mathbb{1}[dow(o) \in \{5, 6\}] \tag{10}$$

The logistic features are:

$$days\_to\_deliver(o) = \lfloor t_{delivered}(o) - t_{purchase}(o) \rfloor_{days} \tag{11}$$

The label encoding function for category $c_i$ is:

$$\text{encode}(c_i) = \text{rank}(c_i) \in \{0, 1, \ldots, C-1\} \tag{12}$$

where $C = 72$ is the total number of unique product categories and $\text{rank}(c_i)$ is the alphabetical rank of category $c_i$.

The final feature vector for each order $o$ is:

$$\mathbf{x}(o) = [f_{freight}, r_{score}, k_{install}, d_{deliver}, m_{purchase}, w_{end}, c_{encoded}]^T \in \mathbb{R}^7 \tag{13}$$

---

### 3.6 Module 4 — Model Training

**Module Title:** Supervised Regression Model Training — Linear Regression and Decision Tree

**Description:**
Module 4 trains two supervised regression models on the processed feature matrix to predict the target variable `price`. The dataset is partitioned into training (80%) and testing (20%) subsets using a fixed random seed for reproducibility. A helper `evaluate()` function computes three metrics after each model's predictions are obtained.

The train-test split is formalised as:

$$|\mathcal{D}_{train}| = \lfloor 0.8 \cdot N \rfloor, \quad |\mathcal{D}_{test}| = N - |\mathcal{D}_{train}| \tag{14}$$

where $N = 109{,}357$ is the total number of modelling samples.

#### Algorithm 1 — Linear Regression

**Algorithm 1: Ordinary Least Squares Linear Regression**
```
INPUT : Training set {(xᵢ, yᵢ)}ᵢ₌₁ᴺ, feature matrix X ∈ ℝᴺˣ⁷, target y ∈ ℝᴺ
OUTPUT: Coefficient vector β ∈ ℝ⁸ (7 features + intercept)

1. Augment X with a bias column: X̃ ← [1 | X]  (shape N × 8)
2. Compute the OLS closed-form solution:
       β̂ = (X̃ᵀ X̃)⁻¹ X̃ᵀ y
3. For each test sample xᵢ ∈ D_test:
       ŷᵢ ← β̂₀ + Σⱼ β̂ⱼ · xᵢⱼ
4. Compute evaluation metrics RMSE, MAE, R²
5. RETURN β̂, {ŷᵢ}
```

**Explanation:** Linear Regression models the target price as a linear combination of the 7 input features. The optimal weights $\hat{\beta}$ are obtained analytically by minimising the Residual Sum of Squares (RSS):

$$\hat{\beta} = \arg\min_{\beta} \sum_{i=1}^{N} (y_i - \mathbf{x}_i^T \beta)^2 = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y} \tag{15}$$

The predicted price for a new sample $\mathbf{x}^*$ is:

$$\hat{y}^* = \beta_0 + \sum_{j=1}^{7} \beta_j x_j^* \tag{16}$$

---

#### Algorithm 2 — Decision Tree Regressor

**Algorithm 2: Decision Tree Regression with CART (max_depth = 5)**
```
INPUT : Training set {(xᵢ, yᵢ)}, max_depth = 5, min_samples_split = 2
OUTPUT: Trained tree T with leaf predictions

FUNCTION BuildTree(S, depth):
1.  IF depth = max_depth OR |S| < min_samples_split:
        RETURN LeafNode( value = mean({yᵢ : i ∈ S}) )
2.  best_feature ← NULL;  best_threshold ← NULL;  best_MSE ← ∞
3.  FOR each feature j ∈ {1, …, 7}:
        FOR each threshold t in sorted unique values of xⱼ:
            S_L ← {i ∈ S : xᵢⱼ ≤ t}
            S_R ← {i ∈ S : xᵢⱼ > t}
            mse ← (|S_L| · MSE(S_L) + |S_R| · MSE(S_R)) / |S|
            IF mse < best_MSE:
                best_MSE ← mse;  best_feature ← j;  best_threshold ← t
4.  Create InternalNode(feature = best_feature, threshold = best_threshold)
5.  node.left  ← BuildTree(S_L, depth + 1)
6.  node.right ← BuildTree(S_R, depth + 1)
7.  RETURN node

MAIN:
T ← BuildTree(D_train, depth = 0)
FOR each xᵢ ∈ D_test:
    ŷᵢ ← Traverse(T, xᵢ)   // follow splits until leaf, return leaf mean
RETURN T, {ŷᵢ}
```

**Explanation:** The Decision Tree uses the CART (Classification and Regression Trees) algorithm. At each internal node, it selects the feature $j$ and threshold $t$ that minimises the weighted Mean Squared Error across the resulting child partitions:

$$\text{MSE}_{split} = \frac{|S_L|}{|S|} \cdot \text{MSE}(S_L) + \frac{|S_R|}{|S|} \cdot \text{MSE}(S_R) \tag{17}$$

where the MSE of a partition $S_k$ is:

$$\text{MSE}(S_k) = \frac{1}{|S_k|} \sum_{i \in S_k} (y_i - \bar{y}_{S_k})^2, \quad \bar{y}_{S_k} = \frac{1}{|S_k|} \sum_{i \in S_k} y_i \tag{18}$$

The prediction for a test sample is the mean price of all training samples that fall in the same leaf node as that sample. The feature importance $I_j$ for feature $j$ across all nodes $t$ in the tree is:

$$I_j = \sum_{t : \text{split on } j} \frac{|S_t|}{N} \left[ \text{MSE}(S_t) - \frac{|S_{t,L}|}{|S_t|} \text{MSE}(S_{t,L}) - \frac{|S_{t,R}|}{|S_t|} \text{MSE}(S_{t,R}) \right] \tag{19}$$

---

### 3.7 Module 5 — Model Evaluation and Comparison

**Module Title:** Performance Evaluation Using Regression Metrics

**Description:**
Module 5 evaluates and compares both trained models on the held-out test set $\mathcal{D}_{test}$ using three standard regression metrics. All metrics are computed on the same 21,872 test samples to ensure a fair comparison. The three metrics — RMSE, MAE, and R² — capture different aspects of model quality: RMSE penalises large errors more heavily, MAE gives the average absolute deviation in interpretable price units (R$), and R² measures what fraction of price variance the model explains.

The Root Mean Squared Error (RMSE) is:

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} \tag{20}$$

The Mean Absolute Error (MAE) is:

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| \tag{21}$$

The coefficient of determination $R^2$ is:

$$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2} \tag{22}$$

where
```


## License: unknown
https://github.com/oracle/tribuo-site/blob/5fa2491d60dc4181efca3439883e19aa1134aec2/learn/4.0/tutorials/regression-tribuo-v4.html

```
Here is the complete content for your report sections. Copy these directly into your document.

---

## 3. PROPOSED METHODOLOGY

### 3.1 Overview of the Proposed System

The proposed system is a **Machine Learning-based Dynamic Pricing Framework (MLPF)** designed to predict optimal product prices in e-commerce environments using historical transaction data. Unlike static rule-based pricing systems, the proposed framework leverages supervised learning algorithms to model the complex, non-linear relationships between product characteristics, customer behaviour, logistics features, and observed market prices. The system ingests raw transactional data from multiple sources, processes and engineers meaningful features, and produces price predictions that can guide real-time pricing decisions for online sellers.

The framework is built on the Brazilian Olist E-Commerce dataset comprising over 110,000 delivered orders spanning September 2016 to August 2018. The data pipeline integrates eight relational CSV files into a unified master dataframe, applies feature engineering to derive pricing-relevant signals, and trains two baseline supervised learning models — Linear Regression and Decision Tree Regressor — to benchmark price prediction accuracy. The models are evaluated on a held-out test set and compared using standard regression metrics, establishing a performance baseline that motivates the adoption of advanced ensemble models in subsequent project phases.

---

### 3.2 System Architecture

The system architecture, illustrated in **Figure 1**, follows an end-to-end supervised machine learning pipeline organised into five sequential processing modules. Raw data from the Olist platform enters the system at the Data Ingestion layer, passes through preprocessing and feature engineering stages, and finally reaches the model training and evaluation layer where predictions and performance metrics are generated.

```
┌─────────────────────────────────────────────────────────────────────┐
│                  PROPOSED SYSTEM ARCHITECTURE                       │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │  MODULE 1    │    │  MODULE 2    │    │      MODULE 3         │  │
│  │  Data        │───▶│  Data        │───▶│   Feature             │  │
│  │  Ingestion   │    │  Preprocessing│   │   Engineering         │  │
│  │  (8 CSVs)    │    │  & Merging   │    │   & Selection         │  │
│  └──────────────┘    └──────────────┘    └───────────┬───────────┘  │
│                                                       │             │
│  ┌──────────────┐    ┌──────────────┐                │             │
│  │  MODULE 5    │    │  MODULE 4    │◀───────────────┘             │
│  │  Model       │◀───│  Model       │                              │
│  │  Evaluation  │    │  Training    │                              │
│  │  & Comparison│    │  LR + DT     │                              │
│  └──────────────┘    └──────────────┘                              │
└─────────────────────────────────────────────────────────────────────┘
```
**Figure 1:** System Architecture of the Proposed Machine Learning-based Dynamic Pricing Framework (MLPF).

---

### 3.3 Module 1 — Data Ingestion and Integration

**Module Title:** Multi-Source Data Ingestion and Relational Integration

**Description:**
The first module is responsible for loading, parsing, and relationally joining eight separate CSV files sourced from the Olist Brazilian E-Commerce platform. Each file encapsulates a distinct domain of the e-commerce transaction lifecycle: order metadata, order items, payment records, customer reviews, product attributes, customer profiles, seller profiles, and category translations. The eight datasets are merged through a sequence of left joins on common key fields (`order_id`, `product_id`, `customer_id`, `seller_id`), producing a single denormalised master dataframe that serves as the input to all subsequent modules.

Timestamp fields are parsed into datetime objects to enable temporal feature derivation. Payment records for the same order are aggregated using summation for the total payment value and the mode for payment type. Review records are aggregated per order using the mean review score. Product categories are translated from Portuguese to English using the translation lookup table. After integration, a quality filter is applied to retain only rows corresponding to **delivered orders** with prices in the range (0, 5000] R$.

Let $D = \{d_1, d_2, \ldots, d_8\}$ denote the set of eight source datasets. The resulting master dataframe $M$ is defined as:

$$M = d_1 \bowtie_{order\_id} d_2 \bowtie_{order\_id} d_3' \bowtie_{order\_id} d_4' \bowtie_{product\_id} d_5' \bowtie_{customer\_id} d_6 \bowtie_{seller\_id} d_7 \tag{1}$$

where $d_3'$, $d_4'$, $d_5'$ denote the aggregated payment, review, and translated product tables respectively, and $\bowtie$ denotes a left equi-join operation.

The payment aggregation for order $o$ is:

$$P_{total}(o) = \sum_{i=1}^{n_o} p_i \tag{2}$$

$$P_{installments}(o) = \max_{i=1}^{n_o}(k_i) \tag{3}$$

where $p_i$ is the payment value and $k_i$ is the number of installments for the $i$-th payment record of order $o$, and $n_o$ is the total payment records for order $o$.

The review score aggregation is:

$$R(o) = \frac{1}{m_o} \sum_{j=1}^{m_o} r_j \tag{4}$$

where $r_j$ is the $j$-th review score for order $o$ and $m_o$ is the number of reviews.

The quality filter applied to the master dataframe is:

$$M_{filtered} = \{x \in M \mid status(x) = \text{"delivered"} \wedge 0 < price(x) \leq 5000\} \tag{5}$$

---

### 3.4 Module 2 — Data Preprocessing

**Module Title:** Temporal Parsing, Missing Value Handling, and Data Normalisation

**Description:**
Module 2 processes the raw master dataframe to prepare it for feature engineering. Three key preprocessing operations are performed: timestamp parsing, missing value imputation, and outlier-aware filtering.

Timestamp columns (`order_purchase_timestamp`, `order_delivered_customer_date`, `order_estimated_delivery_date`) are parsed from string representation into Python `datetime` objects using the `errors="coerce"` strategy, which converts unparseable values to `NaT` (Not a Time) rather than raising exceptions. This preserves data integrity while allowing downstream temporal arithmetic.

Missing values in the `delivery_delay` column (arising from undelivered orders already filtered out) are imputed with zero. Rows with missing values in the selected feature set are dropped during the modelling phase. The final filtered dataset retains **109,357 rows** out of 110,194 after removing records with NaN in any model feature.

The imputation strategy for delivery delay is:

$$\delta_{delay}(o) = \begin{cases} t_{delivered}(o) - t_{estimated}(o) & \text{if } t_{delivered}(o) \text{ is known} \\ 0 & \text{otherwise} \end{cases} \tag{6}$$

where $t_{delivered}(o)$ and $t_{estimated}(o)$ are the actual and estimated delivery timestamps for order $o$.

The freight ratio, a normalised measure of shipping cost relative to product price, is computed as:

$$f_{ratio}(o) = \frac{freight\_value(o)}{price(o) + \epsilon} \tag{7}$$

where $\epsilon = 10^{-3}$ is a small constant added to prevent division by zero.

---

### 3.5 Module 3 — Feature Engineering and Selection

**Module Title:** Temporal, Logistic, and Categorical Feature Derivation

**Description:**
Module 3 derives a rich set of explanatory features from the raw merged columns. Features are grouped into three categories: temporal features extracted from timestamps, logistic features derived from delivery information, and encoded categorical features.

**Temporal features** capture the time-based demand patterns known to influence pricing. The purchase month captures seasonality, day-of-week captures weekly demand cycles, and a binary weekend flag distinguishes weekday from weekend purchasing behaviour.

**Logistic features** capture the cost and efficiency of order fulfilment. The actual days to deliver is a proxy for distance and product type. The delivery delay captures fulfilment reliability.

**Categorical encoding** uses Label Encoding to convert the 72-class `category_en` feature into an integer representation. While ordinal encoding introduces implicit ordering, it allows tree-based models (which partition on thresholds) to use this feature effectively.

The temporal features are defined as:

$$month(o) = \text{month}(t_{purchase}(o)) \in \{1, 2, \ldots, 12\} \tag{8}$$

$$dow(o) = \text{dayofweek}(t_{purchase}(o)) \in \{0, 1, \ldots, 6\} \tag{9}$$

$$is\_weekend(o) = \mathbb{1}[dow(o) \in \{5, 6\}] \tag{10}$$

The logistic features are:

$$days\_to\_deliver(o) = \lfloor t_{delivered}(o) - t_{purchase}(o) \rfloor_{days} \tag{11}$$

The label encoding function for category $c_i$ is:

$$\text{encode}(c_i) = \text{rank}(c_i) \in \{0, 1, \ldots, C-1\} \tag{12}$$

where $C = 72$ is the total number of unique product categories and $\text{rank}(c_i)$ is the alphabetical rank of category $c_i$.

The final feature vector for each order $o$ is:

$$\mathbf{x}(o) = [f_{freight}, r_{score}, k_{install}, d_{deliver}, m_{purchase}, w_{end}, c_{encoded}]^T \in \mathbb{R}^7 \tag{13}$$

---

### 3.6 Module 4 — Model Training

**Module Title:** Supervised Regression Model Training — Linear Regression and Decision Tree

**Description:**
Module 4 trains two supervised regression models on the processed feature matrix to predict the target variable `price`. The dataset is partitioned into training (80%) and testing (20%) subsets using a fixed random seed for reproducibility. A helper `evaluate()` function computes three metrics after each model's predictions are obtained.

The train-test split is formalised as:

$$|\mathcal{D}_{train}| = \lfloor 0.8 \cdot N \rfloor, \quad |\mathcal{D}_{test}| = N - |\mathcal{D}_{train}| \tag{14}$$

where $N = 109{,}357$ is the total number of modelling samples.

#### Algorithm 1 — Linear Regression

**Algorithm 1: Ordinary Least Squares Linear Regression**
```
INPUT : Training set {(xᵢ, yᵢ)}ᵢ₌₁ᴺ, feature matrix X ∈ ℝᴺˣ⁷, target y ∈ ℝᴺ
OUTPUT: Coefficient vector β ∈ ℝ⁸ (7 features + intercept)

1. Augment X with a bias column: X̃ ← [1 | X]  (shape N × 8)
2. Compute the OLS closed-form solution:
       β̂ = (X̃ᵀ X̃)⁻¹ X̃ᵀ y
3. For each test sample xᵢ ∈ D_test:
       ŷᵢ ← β̂₀ + Σⱼ β̂ⱼ · xᵢⱼ
4. Compute evaluation metrics RMSE, MAE, R²
5. RETURN β̂, {ŷᵢ}
```

**Explanation:** Linear Regression models the target price as a linear combination of the 7 input features. The optimal weights $\hat{\beta}$ are obtained analytically by minimising the Residual Sum of Squares (RSS):

$$\hat{\beta} = \arg\min_{\beta} \sum_{i=1}^{N} (y_i - \mathbf{x}_i^T \beta)^2 = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y} \tag{15}$$

The predicted price for a new sample $\mathbf{x}^*$ is:

$$\hat{y}^* = \beta_0 + \sum_{j=1}^{7} \beta_j x_j^* \tag{16}$$

---

#### Algorithm 2 — Decision Tree Regressor

**Algorithm 2: Decision Tree Regression with CART (max_depth = 5)**
```
INPUT : Training set {(xᵢ, yᵢ)}, max_depth = 5, min_samples_split = 2
OUTPUT: Trained tree T with leaf predictions

FUNCTION BuildTree(S, depth):
1.  IF depth = max_depth OR |S| < min_samples_split:
        RETURN LeafNode( value = mean({yᵢ : i ∈ S}) )
2.  best_feature ← NULL;  best_threshold ← NULL;  best_MSE ← ∞
3.  FOR each feature j ∈ {1, …, 7}:
        FOR each threshold t in sorted unique values of xⱼ:
            S_L ← {i ∈ S : xᵢⱼ ≤ t}
            S_R ← {i ∈ S : xᵢⱼ > t}
            mse ← (|S_L| · MSE(S_L) + |S_R| · MSE(S_R)) / |S|
            IF mse < best_MSE:
                best_MSE ← mse;  best_feature ← j;  best_threshold ← t
4.  Create InternalNode(feature = best_feature, threshold = best_threshold)
5.  node.left  ← BuildTree(S_L, depth + 1)
6.  node.right ← BuildTree(S_R, depth + 1)
7.  RETURN node

MAIN:
T ← BuildTree(D_train, depth = 0)
FOR each xᵢ ∈ D_test:
    ŷᵢ ← Traverse(T, xᵢ)   // follow splits until leaf, return leaf mean
RETURN T, {ŷᵢ}
```

**Explanation:** The Decision Tree uses the CART (Classification and Regression Trees) algorithm. At each internal node, it selects the feature $j$ and threshold $t$ that minimises the weighted Mean Squared Error across the resulting child partitions:

$$\text{MSE}_{split} = \frac{|S_L|}{|S|} \cdot \text{MSE}(S_L) + \frac{|S_R|}{|S|} \cdot \text{MSE}(S_R) \tag{17}$$

where the MSE of a partition $S_k$ is:

$$\text{MSE}(S_k) = \frac{1}{|S_k|} \sum_{i \in S_k} (y_i - \bar{y}_{S_k})^2, \quad \bar{y}_{S_k} = \frac{1}{|S_k|} \sum_{i \in S_k} y_i \tag{18}$$

The prediction for a test sample is the mean price of all training samples that fall in the same leaf node as that sample. The feature importance $I_j$ for feature $j$ across all nodes $t$ in the tree is:

$$I_j = \sum_{t : \text{split on } j} \frac{|S_t|}{N} \left[ \text{MSE}(S_t) - \frac{|S_{t,L}|}{|S_t|} \text{MSE}(S_{t,L}) - \frac{|S_{t,R}|}{|S_t|} \text{MSE}(S_{t,R}) \right] \tag{19}$$

---

### 3.7 Module 5 — Model Evaluation and Comparison

**Module Title:** Performance Evaluation Using Regression Metrics

**Description:**
Module 5 evaluates and compares both trained models on the held-out test set $\mathcal{D}_{test}$ using three standard regression metrics. All metrics are computed on the same 21,872 test samples to ensure a fair comparison. The three metrics — RMSE, MAE, and R² — capture different aspects of model quality: RMSE penalises large errors more heavily, MAE gives the average absolute deviation in interpretable price units (R$), and R² measures what fraction of price variance the model explains.

The Root Mean Squared Error (RMSE) is:

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} \tag{20}$$

The Mean Absolute Error (MAE) is:

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| \tag{21}$$

The coefficient of determination $R^2$ is:

$$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2} \tag{22}$$

where
```


## License: MIT
https://github.com/mlouarra/Chicago_Crime/blob/48e85467cf8d124e7e1aa84cd7b83d00f52fa51f/docs/Cahier_des_charges_mlops.tex

```
Here is the complete content for your report sections. Copy these directly into your document.

---

## 3. PROPOSED METHODOLOGY

### 3.1 Overview of the Proposed System

The proposed system is a **Machine Learning-based Dynamic Pricing Framework (MLPF)** designed to predict optimal product prices in e-commerce environments using historical transaction data. Unlike static rule-based pricing systems, the proposed framework leverages supervised learning algorithms to model the complex, non-linear relationships between product characteristics, customer behaviour, logistics features, and observed market prices. The system ingests raw transactional data from multiple sources, processes and engineers meaningful features, and produces price predictions that can guide real-time pricing decisions for online sellers.

The framework is built on the Brazilian Olist E-Commerce dataset comprising over 110,000 delivered orders spanning September 2016 to August 2018. The data pipeline integrates eight relational CSV files into a unified master dataframe, applies feature engineering to derive pricing-relevant signals, and trains two baseline supervised learning models — Linear Regression and Decision Tree Regressor — to benchmark price prediction accuracy. The models are evaluated on a held-out test set and compared using standard regression metrics, establishing a performance baseline that motivates the adoption of advanced ensemble models in subsequent project phases.

---

### 3.2 System Architecture

The system architecture, illustrated in **Figure 1**, follows an end-to-end supervised machine learning pipeline organised into five sequential processing modules. Raw data from the Olist platform enters the system at the Data Ingestion layer, passes through preprocessing and feature engineering stages, and finally reaches the model training and evaluation layer where predictions and performance metrics are generated.

```
┌─────────────────────────────────────────────────────────────────────┐
│                  PROPOSED SYSTEM ARCHITECTURE                       │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │  MODULE 1    │    │  MODULE 2    │    │      MODULE 3         │  │
│  │  Data        │───▶│  Data        │───▶│   Feature             │  │
│  │  Ingestion   │    │  Preprocessing│   │   Engineering         │  │
│  │  (8 CSVs)    │    │  & Merging   │    │   & Selection         │  │
│  └──────────────┘    └──────────────┘    └───────────┬───────────┘  │
│                                                       │             │
│  ┌──────────────┐    ┌──────────────┐                │             │
│  │  MODULE 5    │    │  MODULE 4    │◀───────────────┘             │
│  │  Model       │◀───│  Model       │                              │
│  │  Evaluation  │    │  Training    │                              │
│  │  & Comparison│    │  LR + DT     │                              │
│  └──────────────┘    └──────────────┘                              │
└─────────────────────────────────────────────────────────────────────┘
```
**Figure 1:** System Architecture of the Proposed Machine Learning-based Dynamic Pricing Framework (MLPF).

---

### 3.3 Module 1 — Data Ingestion and Integration

**Module Title:** Multi-Source Data Ingestion and Relational Integration

**Description:**
The first module is responsible for loading, parsing, and relationally joining eight separate CSV files sourced from the Olist Brazilian E-Commerce platform. Each file encapsulates a distinct domain of the e-commerce transaction lifecycle: order metadata, order items, payment records, customer reviews, product attributes, customer profiles, seller profiles, and category translations. The eight datasets are merged through a sequence of left joins on common key fields (`order_id`, `product_id`, `customer_id`, `seller_id`), producing a single denormalised master dataframe that serves as the input to all subsequent modules.

Timestamp fields are parsed into datetime objects to enable temporal feature derivation. Payment records for the same order are aggregated using summation for the total payment value and the mode for payment type. Review records are aggregated per order using the mean review score. Product categories are translated from Portuguese to English using the translation lookup table. After integration, a quality filter is applied to retain only rows corresponding to **delivered orders** with prices in the range (0, 5000] R$.

Let $D = \{d_1, d_2, \ldots, d_8\}$ denote the set of eight source datasets. The resulting master dataframe $M$ is defined as:

$$M = d_1 \bowtie_{order\_id} d_2 \bowtie_{order\_id} d_3' \bowtie_{order\_id} d_4' \bowtie_{product\_id} d_5' \bowtie_{customer\_id} d_6 \bowtie_{seller\_id} d_7 \tag{1}$$

where $d_3'$, $d_4'$, $d_5'$ denote the aggregated payment, review, and translated product tables respectively, and $\bowtie$ denotes a left equi-join operation.

The payment aggregation for order $o$ is:

$$P_{total}(o) = \sum_{i=1}^{n_o} p_i \tag{2}$$

$$P_{installments}(o) = \max_{i=1}^{n_o}(k_i) \tag{3}$$

where $p_i$ is the payment value and $k_i$ is the number of installments for the $i$-th payment record of order $o$, and $n_o$ is the total payment records for order $o$.

The review score aggregation is:

$$R(o) = \frac{1}{m_o} \sum_{j=1}^{m_o} r_j \tag{4}$$

where $r_j$ is the $j$-th review score for order $o$ and $m_o$ is the number of reviews.

The quality filter applied to the master dataframe is:

$$M_{filtered} = \{x \in M \mid status(x) = \text{"delivered"} \wedge 0 < price(x) \leq 5000\} \tag{5}$$

---

### 3.4 Module 2 — Data Preprocessing

**Module Title:** Temporal Parsing, Missing Value Handling, and Data Normalisation

**Description:**
Module 2 processes the raw master dataframe to prepare it for feature engineering. Three key preprocessing operations are performed: timestamp parsing, missing value imputation, and outlier-aware filtering.

Timestamp columns (`order_purchase_timestamp`, `order_delivered_customer_date`, `order_estimated_delivery_date`) are parsed from string representation into Python `datetime` objects using the `errors="coerce"` strategy, which converts unparseable values to `NaT` (Not a Time) rather than raising exceptions. This preserves data integrity while allowing downstream temporal arithmetic.

Missing values in the `delivery_delay` column (arising from undelivered orders already filtered out) are imputed with zero. Rows with missing values in the selected feature set are dropped during the modelling phase. The final filtered dataset retains **109,357 rows** out of 110,194 after removing records with NaN in any model feature.

The imputation strategy for delivery delay is:

$$\delta_{delay}(o) = \begin{cases} t_{delivered}(o) - t_{estimated}(o) & \text{if } t_{delivered}(o) \text{ is known} \\ 0 & \text{otherwise} \end{cases} \tag{6}$$

where $t_{delivered}(o)$ and $t_{estimated}(o)$ are the actual and estimated delivery timestamps for order $o$.

The freight ratio, a normalised measure of shipping cost relative to product price, is computed as:

$$f_{ratio}(o) = \frac{freight\_value(o)}{price(o) + \epsilon} \tag{7}$$

where $\epsilon = 10^{-3}$ is a small constant added to prevent division by zero.

---

### 3.5 Module 3 — Feature Engineering and Selection

**Module Title:** Temporal, Logistic, and Categorical Feature Derivation

**Description:**
Module 3 derives a rich set of explanatory features from the raw merged columns. Features are grouped into three categories: temporal features extracted from timestamps, logistic features derived from delivery information, and encoded categorical features.

**Temporal features** capture the time-based demand patterns known to influence pricing. The purchase month captures seasonality, day-of-week captures weekly demand cycles, and a binary weekend flag distinguishes weekday from weekend purchasing behaviour.

**Logistic features** capture the cost and efficiency of order fulfilment. The actual days to deliver is a proxy for distance and product type. The delivery delay captures fulfilment reliability.

**Categorical encoding** uses Label Encoding to convert the 72-class `category_en` feature into an integer representation. While ordinal encoding introduces implicit ordering, it allows tree-based models (which partition on thresholds) to use this feature effectively.

The temporal features are defined as:

$$month(o) = \text{month}(t_{purchase}(o)) \in \{1, 2, \ldots, 12\} \tag{8}$$

$$dow(o) = \text{dayofweek}(t_{purchase}(o)) \in \{0, 1, \ldots, 6\} \tag{9}$$

$$is\_weekend(o) = \mathbb{1}[dow(o) \in \{5, 6\}] \tag{10}$$

The logistic features are:

$$days\_to\_deliver(o) = \lfloor t_{delivered}(o) - t_{purchase}(o) \rfloor_{days} \tag{11}$$

The label encoding function for category $c_i$ is:

$$\text{encode}(c_i) = \text{rank}(c_i) \in \{0, 1, \ldots, C-1\} \tag{12}$$

where $C = 72$ is the total number of unique product categories and $\text{rank}(c_i)$ is the alphabetical rank of category $c_i$.

The final feature vector for each order $o$ is:

$$\mathbf{x}(o) = [f_{freight}, r_{score}, k_{install}, d_{deliver}, m_{purchase}, w_{end}, c_{encoded}]^T \in \mathbb{R}^7 \tag{13}$$

---

### 3.6 Module 4 — Model Training

**Module Title:** Supervised Regression Model Training — Linear Regression and Decision Tree

**Description:**
Module 4 trains two supervised regression models on the processed feature matrix to predict the target variable `price`. The dataset is partitioned into training (80%) and testing (20%) subsets using a fixed random seed for reproducibility. A helper `evaluate()` function computes three metrics after each model's predictions are obtained.

The train-test split is formalised as:

$$|\mathcal{D}_{train}| = \lfloor 0.8 \cdot N \rfloor, \quad |\mathcal{D}_{test}| = N - |\mathcal{D}_{train}| \tag{14}$$

where $N = 109{,}357$ is the total number of modelling samples.

#### Algorithm 1 — Linear Regression

**Algorithm 1: Ordinary Least Squares Linear Regression**
```
INPUT : Training set {(xᵢ, yᵢ)}ᵢ₌₁ᴺ, feature matrix X ∈ ℝᴺˣ⁷, target y ∈ ℝᴺ
OUTPUT: Coefficient vector β ∈ ℝ⁸ (7 features + intercept)

1. Augment X with a bias column: X̃ ← [1 | X]  (shape N × 8)
2. Compute the OLS closed-form solution:
       β̂ = (X̃ᵀ X̃)⁻¹ X̃ᵀ y
3. For each test sample xᵢ ∈ D_test:
       ŷᵢ ← β̂₀ + Σⱼ β̂ⱼ · xᵢⱼ
4. Compute evaluation metrics RMSE, MAE, R²
5. RETURN β̂, {ŷᵢ}
```

**Explanation:** Linear Regression models the target price as a linear combination of the 7 input features. The optimal weights $\hat{\beta}$ are obtained analytically by minimising the Residual Sum of Squares (RSS):

$$\hat{\beta} = \arg\min_{\beta} \sum_{i=1}^{N} (y_i - \mathbf{x}_i^T \beta)^2 = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y} \tag{15}$$

The predicted price for a new sample $\mathbf{x}^*$ is:

$$\hat{y}^* = \beta_0 + \sum_{j=1}^{7} \beta_j x_j^* \tag{16}$$

---

#### Algorithm 2 — Decision Tree Regressor

**Algorithm 2: Decision Tree Regression with CART (max_depth = 5)**
```
INPUT : Training set {(xᵢ, yᵢ)}, max_depth = 5, min_samples_split = 2
OUTPUT: Trained tree T with leaf predictions

FUNCTION BuildTree(S, depth):
1.  IF depth = max_depth OR |S| < min_samples_split:
        RETURN LeafNode( value = mean({yᵢ : i ∈ S}) )
2.  best_feature ← NULL;  best_threshold ← NULL;  best_MSE ← ∞
3.  FOR each feature j ∈ {1, …, 7}:
        FOR each threshold t in sorted unique values of xⱼ:
            S_L ← {i ∈ S : xᵢⱼ ≤ t}
            S_R ← {i ∈ S : xᵢⱼ > t}
            mse ← (|S_L| · MSE(S_L) + |S_R| · MSE(S_R)) / |S|
            IF mse < best_MSE:
                best_MSE ← mse;  best_feature ← j;  best_threshold ← t
4.  Create InternalNode(feature = best_feature, threshold = best_threshold)
5.  node.left  ← BuildTree(S_L, depth + 1)
6.  node.right ← BuildTree(S_R, depth + 1)
7.  RETURN node

MAIN:
T ← BuildTree(D_train, depth = 0)
FOR each xᵢ ∈ D_test:
    ŷᵢ ← Traverse(T, xᵢ)   // follow splits until leaf, return leaf mean
RETURN T, {ŷᵢ}
```

**Explanation:** The Decision Tree uses the CART (Classification and Regression Trees) algorithm. At each internal node, it selects the feature $j$ and threshold $t$ that minimises the weighted Mean Squared Error across the resulting child partitions:

$$\text{MSE}_{split} = \frac{|S_L|}{|S|} \cdot \text{MSE}(S_L) + \frac{|S_R|}{|S|} \cdot \text{MSE}(S_R) \tag{17}$$

where the MSE of a partition $S_k$ is:

$$\text{MSE}(S_k) = \frac{1}{|S_k|} \sum_{i \in S_k} (y_i - \bar{y}_{S_k})^2, \quad \bar{y}_{S_k} = \frac{1}{|S_k|} \sum_{i \in S_k} y_i \tag{18}$$

The prediction for a test sample is the mean price of all training samples that fall in the same leaf node as that sample. The feature importance $I_j$ for feature $j$ across all nodes $t$ in the tree is:

$$I_j = \sum_{t : \text{split on } j} \frac{|S_t|}{N} \left[ \text{MSE}(S_t) - \frac{|S_{t,L}|}{|S_t|} \text{MSE}(S_{t,L}) - \frac{|S_{t,R}|}{|S_t|} \text{MSE}(S_{t,R}) \right] \tag{19}$$

---

### 3.7 Module 5 — Model Evaluation and Comparison

**Module Title:** Performance Evaluation Using Regression Metrics

**Description:**
Module 5 evaluates and compares both trained models on the held-out test set $\mathcal{D}_{test}$ using three standard regression metrics. All metrics are computed on the same 21,872 test samples to ensure a fair comparison. The three metrics — RMSE, MAE, and R² — capture different aspects of model quality: RMSE penalises large errors more heavily, MAE gives the average absolute deviation in interpretable price units (R$), and R² measures what fraction of price variance the model explains.

The Root Mean Squared Error (RMSE) is:

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} \tag{20}$$

The Mean Absolute Error (MAE) is:

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| \tag{21}$$

The coefficient of determination $R^2$ is:

$$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2} \tag{22}$$

where
```


## License: unknown
https://github.com/oracle/tribuo-site/blob/5fa2491d60dc4181efca3439883e19aa1134aec2/learn/4.0/tutorials/regression-tribuo-v4.html

```
Here is the complete content for your report sections. Copy these directly into your document.

---

## 3. PROPOSED METHODOLOGY

### 3.1 Overview of the Proposed System

The proposed system is a **Machine Learning-based Dynamic Pricing Framework (MLPF)** designed to predict optimal product prices in e-commerce environments using historical transaction data. Unlike static rule-based pricing systems, the proposed framework leverages supervised learning algorithms to model the complex, non-linear relationships between product characteristics, customer behaviour, logistics features, and observed market prices. The system ingests raw transactional data from multiple sources, processes and engineers meaningful features, and produces price predictions that can guide real-time pricing decisions for online sellers.

The framework is built on the Brazilian Olist E-Commerce dataset comprising over 110,000 delivered orders spanning September 2016 to August 2018. The data pipeline integrates eight relational CSV files into a unified master dataframe, applies feature engineering to derive pricing-relevant signals, and trains two baseline supervised learning models — Linear Regression and Decision Tree Regressor — to benchmark price prediction accuracy. The models are evaluated on a held-out test set and compared using standard regression metrics, establishing a performance baseline that motivates the adoption of advanced ensemble models in subsequent project phases.

---

### 3.2 System Architecture

The system architecture, illustrated in **Figure 1**, follows an end-to-end supervised machine learning pipeline organised into five sequential processing modules. Raw data from the Olist platform enters the system at the Data Ingestion layer, passes through preprocessing and feature engineering stages, and finally reaches the model training and evaluation layer where predictions and performance metrics are generated.

```
┌─────────────────────────────────────────────────────────────────────┐
│                  PROPOSED SYSTEM ARCHITECTURE                       │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │  MODULE 1    │    │  MODULE 2    │    │      MODULE 3         │  │
│  │  Data        │───▶│  Data        │───▶│   Feature             │  │
│  │  Ingestion   │    │  Preprocessing│   │   Engineering         │  │
│  │  (8 CSVs)    │    │  & Merging   │    │   & Selection         │  │
│  └──────────────┘    └──────────────┘    └───────────┬───────────┘  │
│                                                       │             │
│  ┌──────────────┐    ┌──────────────┐                │             │
│  │  MODULE 5    │    │  MODULE 4    │◀───────────────┘             │
│  │  Model       │◀───│  Model       │                              │
│  │  Evaluation  │    │  Training    │                              │
│  │  & Comparison│    │  LR + DT     │                              │
│  └──────────────┘    └──────────────┘                              │
└─────────────────────────────────────────────────────────────────────┘
```
**Figure 1:** System Architecture of the Proposed Machine Learning-based Dynamic Pricing Framework (MLPF).

---

### 3.3 Module 1 — Data Ingestion and Integration

**Module Title:** Multi-Source Data Ingestion and Relational Integration

**Description:**
The first module is responsible for loading, parsing, and relationally joining eight separate CSV files sourced from the Olist Brazilian E-Commerce platform. Each file encapsulates a distinct domain of the e-commerce transaction lifecycle: order metadata, order items, payment records, customer reviews, product attributes, customer profiles, seller profiles, and category translations. The eight datasets are merged through a sequence of left joins on common key fields (`order_id`, `product_id`, `customer_id`, `seller_id`), producing a single denormalised master dataframe that serves as the input to all subsequent modules.

Timestamp fields are parsed into datetime objects to enable temporal feature derivation. Payment records for the same order are aggregated using summation for the total payment value and the mode for payment type. Review records are aggregated per order using the mean review score. Product categories are translated from Portuguese to English using the translation lookup table. After integration, a quality filter is applied to retain only rows corresponding to **delivered orders** with prices in the range (0, 5000] R$.

Let $D = \{d_1, d_2, \ldots, d_8\}$ denote the set of eight source datasets. The resulting master dataframe $M$ is defined as:

$$M = d_1 \bowtie_{order\_id} d_2 \bowtie_{order\_id} d_3' \bowtie_{order\_id} d_4' \bowtie_{product\_id} d_5' \bowtie_{customer\_id} d_6 \bowtie_{seller\_id} d_7 \tag{1}$$

where $d_3'$, $d_4'$, $d_5'$ denote the aggregated payment, review, and translated product tables respectively, and $\bowtie$ denotes a left equi-join operation.

The payment aggregation for order $o$ is:

$$P_{total}(o) = \sum_{i=1}^{n_o} p_i \tag{2}$$

$$P_{installments}(o) = \max_{i=1}^{n_o}(k_i) \tag{3}$$

where $p_i$ is the payment value and $k_i$ is the number of installments for the $i$-th payment record of order $o$, and $n_o$ is the total payment records for order $o$.

The review score aggregation is:

$$R(o) = \frac{1}{m_o} \sum_{j=1}^{m_o} r_j \tag{4}$$

where $r_j$ is the $j$-th review score for order $o$ and $m_o$ is the number of reviews.

The quality filter applied to the master dataframe is:

$$M_{filtered} = \{x \in M \mid status(x) = \text{"delivered"} \wedge 0 < price(x) \leq 5000\} \tag{5}$$

---

### 3.4 Module 2 — Data Preprocessing

**Module Title:** Temporal Parsing, Missing Value Handling, and Data Normalisation

**Description:**
Module 2 processes the raw master dataframe to prepare it for feature engineering. Three key preprocessing operations are performed: timestamp parsing, missing value imputation, and outlier-aware filtering.

Timestamp columns (`order_purchase_timestamp`, `order_delivered_customer_date`, `order_estimated_delivery_date`) are parsed from string representation into Python `datetime` objects using the `errors="coerce"` strategy, which converts unparseable values to `NaT` (Not a Time) rather than raising exceptions. This preserves data integrity while allowing downstream temporal arithmetic.

Missing values in the `delivery_delay` column (arising from undelivered orders already filtered out) are imputed with zero. Rows with missing values in the selected feature set are dropped during the modelling phase. The final filtered dataset retains **109,357 rows** out of 110,194 after removing records with NaN in any model feature.

The imputation strategy for delivery delay is:

$$\delta_{delay}(o) = \begin{cases} t_{delivered}(o) - t_{estimated}(o) & \text{if } t_{delivered}(o) \text{ is known} \\ 0 & \text{otherwise} \end{cases} \tag{6}$$

where $t_{delivered}(o)$ and $t_{estimated}(o)$ are the actual and estimated delivery timestamps for order $o$.

The freight ratio, a normalised measure of shipping cost relative to product price, is computed as:

$$f_{ratio}(o) = \frac{freight\_value(o)}{price(o) + \epsilon} \tag{7}$$

where $\epsilon = 10^{-3}$ is a small constant added to prevent division by zero.

---

### 3.5 Module 3 — Feature Engineering and Selection

**Module Title:** Temporal, Logistic, and Categorical Feature Derivation

**Description:**
Module 3 derives a rich set of explanatory features from the raw merged columns. Features are grouped into three categories: temporal features extracted from timestamps, logistic features derived from delivery information, and encoded categorical features.

**Temporal features** capture the time-based demand patterns known to influence pricing. The purchase month captures seasonality, day-of-week captures weekly demand cycles, and a binary weekend flag distinguishes weekday from weekend purchasing behaviour.

**Logistic features** capture the cost and efficiency of order fulfilment. The actual days to deliver is a proxy for distance and product type. The delivery delay captures fulfilment reliability.

**Categorical encoding** uses Label Encoding to convert the 72-class `category_en` feature into an integer representation. While ordinal encoding introduces implicit ordering, it allows tree-based models (which partition on thresholds) to use this feature effectively.

The temporal features are defined as:

$$month(o) = \text{month}(t_{purchase}(o)) \in \{1, 2, \ldots, 12\} \tag{8}$$

$$dow(o) = \text{dayofweek}(t_{purchase}(o)) \in \{0, 1, \ldots, 6\} \tag{9}$$

$$is\_weekend(o) = \mathbb{1}[dow(o) \in \{5, 6\}] \tag{10}$$

The logistic features are:

$$days\_to\_deliver(o) = \lfloor t_{delivered}(o) - t_{purchase}(o) \rfloor_{days} \tag{11}$$

The label encoding function for category $c_i$ is:

$$\text{encode}(c_i) = \text{rank}(c_i) \in \{0, 1, \ldots, C-1\} \tag{12}$$

where $C = 72$ is the total number of unique product categories and $\text{rank}(c_i)$ is the alphabetical rank of category $c_i$.

The final feature vector for each order $o$ is:

$$\mathbf{x}(o) = [f_{freight}, r_{score}, k_{install}, d_{deliver}, m_{purchase}, w_{end}, c_{encoded}]^T \in \mathbb{R}^7 \tag{13}$$

---

### 3.6 Module 4 — Model Training

**Module Title:** Supervised Regression Model Training — Linear Regression and Decision Tree

**Description:**
Module 4 trains two supervised regression models on the processed feature matrix to predict the target variable `price`. The dataset is partitioned into training (80%) and testing (20%) subsets using a fixed random seed for reproducibility. A helper `evaluate()` function computes three metrics after each model's predictions are obtained.

The train-test split is formalised as:

$$|\mathcal{D}_{train}| = \lfloor 0.8 \cdot N \rfloor, \quad |\mathcal{D}_{test}| = N - |\mathcal{D}_{train}| \tag{14}$$

where $N = 109{,}357$ is the total number of modelling samples.

#### Algorithm 1 — Linear Regression

**Algorithm 1: Ordinary Least Squares Linear Regression**
```
INPUT : Training set {(xᵢ, yᵢ)}ᵢ₌₁ᴺ, feature matrix X ∈ ℝᴺˣ⁷, target y ∈ ℝᴺ
OUTPUT: Coefficient vector β ∈ ℝ⁸ (7 features + intercept)

1. Augment X with a bias column: X̃ ← [1 | X]  (shape N × 8)
2. Compute the OLS closed-form solution:
       β̂ = (X̃ᵀ X̃)⁻¹ X̃ᵀ y
3. For each test sample xᵢ ∈ D_test:
       ŷᵢ ← β̂₀ + Σⱼ β̂ⱼ · xᵢⱼ
4. Compute evaluation metrics RMSE, MAE, R²
5. RETURN β̂, {ŷᵢ}
```

**Explanation:** Linear Regression models the target price as a linear combination of the 7 input features. The optimal weights $\hat{\beta}$ are obtained analytically by minimising the Residual Sum of Squares (RSS):

$$\hat{\beta} = \arg\min_{\beta} \sum_{i=1}^{N} (y_i - \mathbf{x}_i^T \beta)^2 = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y} \tag{15}$$

The predicted price for a new sample $\mathbf{x}^*$ is:

$$\hat{y}^* = \beta_0 + \sum_{j=1}^{7} \beta_j x_j^* \tag{16}$$

---

#### Algorithm 2 — Decision Tree Regressor

**Algorithm 2: Decision Tree Regression with CART (max_depth = 5)**
```
INPUT : Training set {(xᵢ, yᵢ)}, max_depth = 5, min_samples_split = 2
OUTPUT: Trained tree T with leaf predictions

FUNCTION BuildTree(S, depth):
1.  IF depth = max_depth OR |S| < min_samples_split:
        RETURN LeafNode( value = mean({yᵢ : i ∈ S}) )
2.  best_feature ← NULL;  best_threshold ← NULL;  best_MSE ← ∞
3.  FOR each feature j ∈ {1, …, 7}:
        FOR each threshold t in sorted unique values of xⱼ:
            S_L ← {i ∈ S : xᵢⱼ ≤ t}
            S_R ← {i ∈ S : xᵢⱼ > t}
            mse ← (|S_L| · MSE(S_L) + |S_R| · MSE(S_R)) / |S|
            IF mse < best_MSE:
                best_MSE ← mse;  best_feature ← j;  best_threshold ← t
4.  Create InternalNode(feature = best_feature, threshold = best_threshold)
5.  node.left  ← BuildTree(S_L, depth + 1)
6.  node.right ← BuildTree(S_R, depth + 1)
7.  RETURN node

MAIN:
T ← BuildTree(D_train, depth = 0)
FOR each xᵢ ∈ D_test:
    ŷᵢ ← Traverse(T, xᵢ)   // follow splits until leaf, return leaf mean
RETURN T, {ŷᵢ}
```

**Explanation:** The Decision Tree uses the CART (Classification and Regression Trees) algorithm. At each internal node, it selects the feature $j$ and threshold $t$ that minimises the weighted Mean Squared Error across the resulting child partitions:

$$\text{MSE}_{split} = \frac{|S_L|}{|S|} \cdot \text{MSE}(S_L) + \frac{|S_R|}{|S|} \cdot \text{MSE}(S_R) \tag{17}$$

where the MSE of a partition $S_k$ is:

$$\text{MSE}(S_k) = \frac{1}{|S_k|} \sum_{i \in S_k} (y_i - \bar{y}_{S_k})^2, \quad \bar{y}_{S_k} = \frac{1}{|S_k|} \sum_{i \in S_k} y_i \tag{18}$$

The prediction for a test sample is the mean price of all training samples that fall in the same leaf node as that sample. The feature importance $I_j$ for feature $j$ across all nodes $t$ in the tree is:

$$I_j = \sum_{t : \text{split on } j} \frac{|S_t|}{N} \left[ \text{MSE}(S_t) - \frac{|S_{t,L}|}{|S_t|} \text{MSE}(S_{t,L}) - \frac{|S_{t,R}|}{|S_t|} \text{MSE}(S_{t,R}) \right] \tag{19}$$

---

### 3.7 Module 5 — Model Evaluation and Comparison

**Module Title:** Performance Evaluation Using Regression Metrics

**Description:**
Module 5 evaluates and compares both trained models on the held-out test set $\mathcal{D}_{test}$ using three standard regression metrics. All metrics are computed on the same 21,872 test samples to ensure a fair comparison. The three metrics — RMSE, MAE, and R² — capture different aspects of model quality: RMSE penalises large errors more heavily, MAE gives the average absolute deviation in interpretable price units (R$), and R² measures what fraction of price variance the model explains.

The Root Mean Squared Error (RMSE) is:

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} \tag{20}$$

The Mean Absolute Error (MAE) is:

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| \tag{21}$$

The coefficient of determination $R^2$ is:

$$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2} \tag{22}$$

where
```


## License: MIT
https://github.com/mlouarra/Chicago_Crime/blob/48e85467cf8d124e7e1aa84cd7b83d00f52fa51f/docs/Cahier_des_charges_mlops.tex

```
Here is the complete content for your report sections. Copy these directly into your document.

---

## 3. PROPOSED METHODOLOGY

### 3.1 Overview of the Proposed System

The proposed system is a **Machine Learning-based Dynamic Pricing Framework (MLPF)** designed to predict optimal product prices in e-commerce environments using historical transaction data. Unlike static rule-based pricing systems, the proposed framework leverages supervised learning algorithms to model the complex, non-linear relationships between product characteristics, customer behaviour, logistics features, and observed market prices. The system ingests raw transactional data from multiple sources, processes and engineers meaningful features, and produces price predictions that can guide real-time pricing decisions for online sellers.

The framework is built on the Brazilian Olist E-Commerce dataset comprising over 110,000 delivered orders spanning September 2016 to August 2018. The data pipeline integrates eight relational CSV files into a unified master dataframe, applies feature engineering to derive pricing-relevant signals, and trains two baseline supervised learning models — Linear Regression and Decision Tree Regressor — to benchmark price prediction accuracy. The models are evaluated on a held-out test set and compared using standard regression metrics, establishing a performance baseline that motivates the adoption of advanced ensemble models in subsequent project phases.

---

### 3.2 System Architecture

The system architecture, illustrated in **Figure 1**, follows an end-to-end supervised machine learning pipeline organised into five sequential processing modules. Raw data from the Olist platform enters the system at the Data Ingestion layer, passes through preprocessing and feature engineering stages, and finally reaches the model training and evaluation layer where predictions and performance metrics are generated.

```
┌─────────────────────────────────────────────────────────────────────┐
│                  PROPOSED SYSTEM ARCHITECTURE                       │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │  MODULE 1    │    │  MODULE 2    │    │      MODULE 3         │  │
│  │  Data        │───▶│  Data        │───▶│   Feature             │  │
│  │  Ingestion   │    │  Preprocessing│   │   Engineering         │  │
│  │  (8 CSVs)    │    │  & Merging   │    │   & Selection         │  │
│  └──────────────┘    └──────────────┘    └───────────┬───────────┘  │
│                                                       │             │
│  ┌──────────────┐    ┌──────────────┐                │             │
│  │  MODULE 5    │    │  MODULE 4    │◀───────────────┘             │
│  │  Model       │◀───│  Model       │                              │
│  │  Evaluation  │    │  Training    │                              │
│  │  & Comparison│    │  LR + DT     │                              │
│  └──────────────┘    └──────────────┘                              │
└─────────────────────────────────────────────────────────────────────┘
```
**Figure 1:** System Architecture of the Proposed Machine Learning-based Dynamic Pricing Framework (MLPF).

---

### 3.3 Module 1 — Data Ingestion and Integration

**Module Title:** Multi-Source Data Ingestion and Relational Integration

**Description:**
The first module is responsible for loading, parsing, and relationally joining eight separate CSV files sourced from the Olist Brazilian E-Commerce platform. Each file encapsulates a distinct domain of the e-commerce transaction lifecycle: order metadata, order items, payment records, customer reviews, product attributes, customer profiles, seller profiles, and category translations. The eight datasets are merged through a sequence of left joins on common key fields (`order_id`, `product_id`, `customer_id`, `seller_id`), producing a single denormalised master dataframe that serves as the input to all subsequent modules.

Timestamp fields are parsed into datetime objects to enable temporal feature derivation. Payment records for the same order are aggregated using summation for the total payment value and the mode for payment type. Review records are aggregated per order using the mean review score. Product categories are translated from Portuguese to English using the translation lookup table. After integration, a quality filter is applied to retain only rows corresponding to **delivered orders** with prices in the range (0, 5000] R$.

Let $D = \{d_1, d_2, \ldots, d_8\}$ denote the set of eight source datasets. The resulting master dataframe $M$ is defined as:

$$M = d_1 \bowtie_{order\_id} d_2 \bowtie_{order\_id} d_3' \bowtie_{order\_id} d_4' \bowtie_{product\_id} d_5' \bowtie_{customer\_id} d_6 \bowtie_{seller\_id} d_7 \tag{1}$$

where $d_3'$, $d_4'$, $d_5'$ denote the aggregated payment, review, and translated product tables respectively, and $\bowtie$ denotes a left equi-join operation.

The payment aggregation for order $o$ is:

$$P_{total}(o) = \sum_{i=1}^{n_o} p_i \tag{2}$$

$$P_{installments}(o) = \max_{i=1}^{n_o}(k_i) \tag{3}$$

where $p_i$ is the payment value and $k_i$ is the number of installments for the $i$-th payment record of order $o$, and $n_o$ is the total payment records for order $o$.

The review score aggregation is:

$$R(o) = \frac{1}{m_o} \sum_{j=1}^{m_o} r_j \tag{4}$$

where $r_j$ is the $j$-th review score for order $o$ and $m_o$ is the number of reviews.

The quality filter applied to the master dataframe is:

$$M_{filtered} = \{x \in M \mid status(x) = \text{"delivered"} \wedge 0 < price(x) \leq 5000\} \tag{5}$$

---

### 3.4 Module 2 — Data Preprocessing

**Module Title:** Temporal Parsing, Missing Value Handling, and Data Normalisation

**Description:**
Module 2 processes the raw master dataframe to prepare it for feature engineering. Three key preprocessing operations are performed: timestamp parsing, missing value imputation, and outlier-aware filtering.

Timestamp columns (`order_purchase_timestamp`, `order_delivered_customer_date`, `order_estimated_delivery_date`) are parsed from string representation into Python `datetime` objects using the `errors="coerce"` strategy, which converts unparseable values to `NaT` (Not a Time) rather than raising exceptions. This preserves data integrity while allowing downstream temporal arithmetic.

Missing values in the `delivery_delay` column (arising from undelivered orders already filtered out) are imputed with zero. Rows with missing values in the selected feature set are dropped during the modelling phase. The final filtered dataset retains **109,357 rows** out of 110,194 after removing records with NaN in any model feature.

The imputation strategy for delivery delay is:

$$\delta_{delay}(o) = \begin{cases} t_{delivered}(o) - t_{estimated}(o) & \text{if } t_{delivered}(o) \text{ is known} \\ 0 & \text{otherwise} \end{cases} \tag{6}$$

where $t_{delivered}(o)$ and $t_{estimated}(o)$ are the actual and estimated delivery timestamps for order $o$.

The freight ratio, a normalised measure of shipping cost relative to product price, is computed as:

$$f_{ratio}(o) = \frac{freight\_value(o)}{price(o) + \epsilon} \tag{7}$$

where $\epsilon = 10^{-3}$ is a small constant added to prevent division by zero.

---

### 3.5 Module 3 — Feature Engineering and Selection

**Module Title:** Temporal, Logistic, and Categorical Feature Derivation

**Description:**
Module 3 derives a rich set of explanatory features from the raw merged columns. Features are grouped into three categories: temporal features extracted from timestamps, logistic features derived from delivery information, and encoded categorical features.

**Temporal features** capture the time-based demand patterns known to influence pricing. The purchase month captures seasonality, day-of-week captures weekly demand cycles, and a binary weekend flag distinguishes weekday from weekend purchasing behaviour.

**Logistic features** capture the cost and efficiency of order fulfilment. The actual days to deliver is a proxy for distance and product type. The delivery delay captures fulfilment reliability.

**Categorical encoding** uses Label Encoding to convert the 72-class `category_en` feature into an integer representation. While ordinal encoding introduces implicit ordering, it allows tree-based models (which partition on thresholds) to use this feature effectively.

The temporal features are defined as:

$$month(o) = \text{month}(t_{purchase}(o)) \in \{1, 2, \ldots, 12\} \tag{8}$$

$$dow(o) = \text{dayofweek}(t_{purchase}(o)) \in \{0, 1, \ldots, 6\} \tag{9}$$

$$is\_weekend(o) = \mathbb{1}[dow(o) \in \{5, 6\}] \tag{10}$$

The logistic features are:

$$days\_to\_deliver(o) = \lfloor t_{delivered}(o) - t_{purchase}(o) \rfloor_{days} \tag{11}$$

The label encoding function for category $c_i$ is:

$$\text{encode}(c_i) = \text{rank}(c_i) \in \{0, 1, \ldots, C-1\} \tag{12}$$

where $C = 72$ is the total number of unique product categories and $\text{rank}(c_i)$ is the alphabetical rank of category $c_i$.

The final feature vector for each order $o$ is:

$$\mathbf{x}(o) = [f_{freight}, r_{score}, k_{install}, d_{deliver}, m_{purchase}, w_{end}, c_{encoded}]^T \in \mathbb{R}^7 \tag{13}$$

---

### 3.6 Module 4 — Model Training

**Module Title:** Supervised Regression Model Training — Linear Regression and Decision Tree

**Description:**
Module 4 trains two supervised regression models on the processed feature matrix to predict the target variable `price`. The dataset is partitioned into training (80%) and testing (20%) subsets using a fixed random seed for reproducibility. A helper `evaluate()` function computes three metrics after each model's predictions are obtained.

The train-test split is formalised as:

$$|\mathcal{D}_{train}| = \lfloor 0.8 \cdot N \rfloor, \quad |\mathcal{D}_{test}| = N - |\mathcal{D}_{train}| \tag{14}$$

where $N = 109{,}357$ is the total number of modelling samples.

#### Algorithm 1 — Linear Regression

**Algorithm 1: Ordinary Least Squares Linear Regression**
```
INPUT : Training set {(xᵢ, yᵢ)}ᵢ₌₁ᴺ, feature matrix X ∈ ℝᴺˣ⁷, target y ∈ ℝᴺ
OUTPUT: Coefficient vector β ∈ ℝ⁸ (7 features + intercept)

1. Augment X with a bias column: X̃ ← [1 | X]  (shape N × 8)
2. Compute the OLS closed-form solution:
       β̂ = (X̃ᵀ X̃)⁻¹ X̃ᵀ y
3. For each test sample xᵢ ∈ D_test:
       ŷᵢ ← β̂₀ + Σⱼ β̂ⱼ · xᵢⱼ
4. Compute evaluation metrics RMSE, MAE, R²
5. RETURN β̂, {ŷᵢ}
```

**Explanation:** Linear Regression models the target price as a linear combination of the 7 input features. The optimal weights $\hat{\beta}$ are obtained analytically by minimising the Residual Sum of Squares (RSS):

$$\hat{\beta} = \arg\min_{\beta} \sum_{i=1}^{N} (y_i - \mathbf{x}_i^T \beta)^2 = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y} \tag{15}$$

The predicted price for a new sample $\mathbf{x}^*$ is:

$$\hat{y}^* = \beta_0 + \sum_{j=1}^{7} \beta_j x_j^* \tag{16}$$

---

#### Algorithm 2 — Decision Tree Regressor

**Algorithm 2: Decision Tree Regression with CART (max_depth = 5)**
```
INPUT : Training set {(xᵢ, yᵢ)}, max_depth = 5, min_samples_split = 2
OUTPUT: Trained tree T with leaf predictions

FUNCTION BuildTree(S, depth):
1.  IF depth = max_depth OR |S| < min_samples_split:
        RETURN LeafNode( value = mean({yᵢ : i ∈ S}) )
2.  best_feature ← NULL;  best_threshold ← NULL;  best_MSE ← ∞
3.  FOR each feature j ∈ {1, …, 7}:
        FOR each threshold t in sorted unique values of xⱼ:
            S_L ← {i ∈ S : xᵢⱼ ≤ t}
            S_R ← {i ∈ S : xᵢⱼ > t}
            mse ← (|S_L| · MSE(S_L) + |S_R| · MSE(S_R)) / |S|
            IF mse < best_MSE:
                best_MSE ← mse;  best_feature ← j;  best_threshold ← t
4.  Create InternalNode(feature = best_feature, threshold = best_threshold)
5.  node.left  ← BuildTree(S_L, depth + 1)
6.  node.right ← BuildTree(S_R, depth + 1)
7.  RETURN node

MAIN:
T ← BuildTree(D_train, depth = 0)
FOR each xᵢ ∈ D_test:
    ŷᵢ ← Traverse(T, xᵢ)   // follow splits until leaf, return leaf mean
RETURN T, {ŷᵢ}
```

**Explanation:** The Decision Tree uses the CART (Classification and Regression Trees) algorithm. At each internal node, it selects the feature $j$ and threshold $t$ that minimises the weighted Mean Squared Error across the resulting child partitions:

$$\text{MSE}_{split} = \frac{|S_L|}{|S|} \cdot \text{MSE}(S_L) + \frac{|S_R|}{|S|} \cdot \text{MSE}(S_R) \tag{17}$$

where the MSE of a partition $S_k$ is:

$$\text{MSE}(S_k) = \frac{1}{|S_k|} \sum_{i \in S_k} (y_i - \bar{y}_{S_k})^2, \quad \bar{y}_{S_k} = \frac{1}{|S_k|} \sum_{i \in S_k} y_i \tag{18}$$

The prediction for a test sample is the mean price of all training samples that fall in the same leaf node as that sample. The feature importance $I_j$ for feature $j$ across all nodes $t$ in the tree is:

$$I_j = \sum_{t : \text{split on } j} \frac{|S_t|}{N} \left[ \text{MSE}(S_t) - \frac{|S_{t,L}|}{|S_t|} \text{MSE}(S_{t,L}) - \frac{|S_{t,R}|}{|S_t|} \text{MSE}(S_{t,R}) \right] \tag{19}$$

---

### 3.7 Module 5 — Model Evaluation and Comparison

**Module Title:** Performance Evaluation Using Regression Metrics

**Description:**
Module 5 evaluates and compares both trained models on the held-out test set $\mathcal{D}_{test}$ using three standard regression metrics. All metrics are computed on the same 21,872 test samples to ensure a fair comparison. The three metrics — RMSE, MAE, and R² — capture different aspects of model quality: RMSE penalises large errors more heavily, MAE gives the average absolute deviation in interpretable price units (R$), and R² measures what fraction of price variance the model explains.

The Root Mean Squared Error (RMSE) is:

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} \tag{20}$$

The Mean Absolute Error (MAE) is:

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| \tag{21}$$

The coefficient of determination $R^2$ is:

$$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2} \tag{22}$$

where
```


## License: unknown
https://github.com/oracle/tribuo-site/blob/5fa2491d60dc4181efca3439883e19aa1134aec2/learn/4.0/tutorials/regression-tribuo-v4.html

```
Here is the complete content for your report sections. Copy these directly into your document.

---

## 3. PROPOSED METHODOLOGY

### 3.1 Overview of the Proposed System

The proposed system is a **Machine Learning-based Dynamic Pricing Framework (MLPF)** designed to predict optimal product prices in e-commerce environments using historical transaction data. Unlike static rule-based pricing systems, the proposed framework leverages supervised learning algorithms to model the complex, non-linear relationships between product characteristics, customer behaviour, logistics features, and observed market prices. The system ingests raw transactional data from multiple sources, processes and engineers meaningful features, and produces price predictions that can guide real-time pricing decisions for online sellers.

The framework is built on the Brazilian Olist E-Commerce dataset comprising over 110,000 delivered orders spanning September 2016 to August 2018. The data pipeline integrates eight relational CSV files into a unified master dataframe, applies feature engineering to derive pricing-relevant signals, and trains two baseline supervised learning models — Linear Regression and Decision Tree Regressor — to benchmark price prediction accuracy. The models are evaluated on a held-out test set and compared using standard regression metrics, establishing a performance baseline that motivates the adoption of advanced ensemble models in subsequent project phases.

---

### 3.2 System Architecture

The system architecture, illustrated in **Figure 1**, follows an end-to-end supervised machine learning pipeline organised into five sequential processing modules. Raw data from the Olist platform enters the system at the Data Ingestion layer, passes through preprocessing and feature engineering stages, and finally reaches the model training and evaluation layer where predictions and performance metrics are generated.

```
┌─────────────────────────────────────────────────────────────────────┐
│                  PROPOSED SYSTEM ARCHITECTURE                       │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │  MODULE 1    │    │  MODULE 2    │    │      MODULE 3         │  │
│  │  Data        │───▶│  Data        │───▶│   Feature             │  │
│  │  Ingestion   │    │  Preprocessing│   │   Engineering         │  │
│  │  (8 CSVs)    │    │  & Merging   │    │   & Selection         │  │
│  └──────────────┘    └──────────────┘    └───────────┬───────────┘  │
│                                                       │             │
│  ┌──────────────┐    ┌──────────────┐                │             │
│  │  MODULE 5    │    │  MODULE 4    │◀───────────────┘             │
│  │  Model       │◀───│  Model       │                              │
│  │  Evaluation  │    │  Training    │                              │
│  │  & Comparison│    │  LR + DT     │                              │
│  └──────────────┘    └──────────────┘                              │
└─────────────────────────────────────────────────────────────────────┘
```
**Figure 1:** System Architecture of the Proposed Machine Learning-based Dynamic Pricing Framework (MLPF).

---

### 3.3 Module 1 — Data Ingestion and Integration

**Module Title:** Multi-Source Data Ingestion and Relational Integration

**Description:**
The first module is responsible for loading, parsing, and relationally joining eight separate CSV files sourced from the Olist Brazilian E-Commerce platform. Each file encapsulates a distinct domain of the e-commerce transaction lifecycle: order metadata, order items, payment records, customer reviews, product attributes, customer profiles, seller profiles, and category translations. The eight datasets are merged through a sequence of left joins on common key fields (`order_id`, `product_id`, `customer_id`, `seller_id`), producing a single denormalised master dataframe that serves as the input to all subsequent modules.

Timestamp fields are parsed into datetime objects to enable temporal feature derivation. Payment records for the same order are aggregated using summation for the total payment value and the mode for payment type. Review records are aggregated per order using the mean review score. Product categories are translated from Portuguese to English using the translation lookup table. After integration, a quality filter is applied to retain only rows corresponding to **delivered orders** with prices in the range (0, 5000] R$.

Let $D = \{d_1, d_2, \ldots, d_8\}$ denote the set of eight source datasets. The resulting master dataframe $M$ is defined as:

$$M = d_1 \bowtie_{order\_id} d_2 \bowtie_{order\_id} d_3' \bowtie_{order\_id} d_4' \bowtie_{product\_id} d_5' \bowtie_{customer\_id} d_6 \bowtie_{seller\_id} d_7 \tag{1}$$

where $d_3'$, $d_4'$, $d_5'$ denote the aggregated payment, review, and translated product tables respectively, and $\bowtie$ denotes a left equi-join operation.

The payment aggregation for order $o$ is:

$$P_{total}(o) = \sum_{i=1}^{n_o} p_i \tag{2}$$

$$P_{installments}(o) = \max_{i=1}^{n_o}(k_i) \tag{3}$$

where $p_i$ is the payment value and $k_i$ is the number of installments for the $i$-th payment record of order $o$, and $n_o$ is the total payment records for order $o$.

The review score aggregation is:

$$R(o) = \frac{1}{m_o} \sum_{j=1}^{m_o} r_j \tag{4}$$

where $r_j$ is the $j$-th review score for order $o$ and $m_o$ is the number of reviews.

The quality filter applied to the master dataframe is:

$$M_{filtered} = \{x \in M \mid status(x) = \text{"delivered"} \wedge 0 < price(x) \leq 5000\} \tag{5}$$

---

### 3.4 Module 2 — Data Preprocessing

**Module Title:** Temporal Parsing, Missing Value Handling, and Data Normalisation

**Description:**
Module 2 processes the raw master dataframe to prepare it for feature engineering. Three key preprocessing operations are performed: timestamp parsing, missing value imputation, and outlier-aware filtering.

Timestamp columns (`order_purchase_timestamp`, `order_delivered_customer_date`, `order_estimated_delivery_date`) are parsed from string representation into Python `datetime` objects using the `errors="coerce"` strategy, which converts unparseable values to `NaT` (Not a Time) rather than raising exceptions. This preserves data integrity while allowing downstream temporal arithmetic.

Missing values in the `delivery_delay` column (arising from undelivered orders already filtered out) are imputed with zero. Rows with missing values in the selected feature set are dropped during the modelling phase. The final filtered dataset retains **109,357 rows** out of 110,194 after removing records with NaN in any model feature.

The imputation strategy for delivery delay is:

$$\delta_{delay}(o) = \begin{cases} t_{delivered}(o) - t_{estimated}(o) & \text{if } t_{delivered}(o) \text{ is known} \\ 0 & \text{otherwise} \end{cases} \tag{6}$$

where $t_{delivered}(o)$ and $t_{estimated}(o)$ are the actual and estimated delivery timestamps for order $o$.

The freight ratio, a normalised measure of shipping cost relative to product price, is computed as:

$$f_{ratio}(o) = \frac{freight\_value(o)}{price(o) + \epsilon} \tag{7}$$

where $\epsilon = 10^{-3}$ is a small constant added to prevent division by zero.

---

### 3.5 Module 3 — Feature Engineering and Selection

**Module Title:** Temporal, Logistic, and Categorical Feature Derivation

**Description:**
Module 3 derives a rich set of explanatory features from the raw merged columns. Features are grouped into three categories: temporal features extracted from timestamps, logistic features derived from delivery information, and encoded categorical features.

**Temporal features** capture the time-based demand patterns known to influence pricing. The purchase month captures seasonality, day-of-week captures weekly demand cycles, and a binary weekend flag distinguishes weekday from weekend purchasing behaviour.

**Logistic features** capture the cost and efficiency of order fulfilment. The actual days to deliver is a proxy for distance and product type. The delivery delay captures fulfilment reliability.

**Categorical encoding** uses Label Encoding to convert the 72-class `category_en` feature into an integer representation. While ordinal encoding introduces implicit ordering, it allows tree-based models (which partition on thresholds) to use this feature effectively.

The temporal features are defined as:

$$month(o) = \text{month}(t_{purchase}(o)) \in \{1, 2, \ldots, 12\} \tag{8}$$

$$dow(o) = \text{dayofweek}(t_{purchase}(o)) \in \{0, 1, \ldots, 6\} \tag{9}$$

$$is\_weekend(o) = \mathbb{1}[dow(o) \in \{5, 6\}] \tag{10}$$

The logistic features are:

$$days\_to\_deliver(o) = \lfloor t_{delivered}(o) - t_{purchase}(o) \rfloor_{days} \tag{11}$$

The label encoding function for category $c_i$ is:

$$\text{encode}(c_i) = \text{rank}(c_i) \in \{0, 1, \ldots, C-1\} \tag{12}$$

where $C = 72$ is the total number of unique product categories and $\text{rank}(c_i)$ is the alphabetical rank of category $c_i$.

The final feature vector for each order $o$ is:

$$\mathbf{x}(o) = [f_{freight}, r_{score}, k_{install}, d_{deliver}, m_{purchase}, w_{end}, c_{encoded}]^T \in \mathbb{R}^7 \tag{13}$$

---

### 3.6 Module 4 — Model Training

**Module Title:** Supervised Regression Model Training — Linear Regression and Decision Tree

**Description:**
Module 4 trains two supervised regression models on the processed feature matrix to predict the target variable `price`. The dataset is partitioned into training (80%) and testing (20%) subsets using a fixed random seed for reproducibility. A helper `evaluate()` function computes three metrics after each model's predictions are obtained.

The train-test split is formalised as:

$$|\mathcal{D}_{train}| = \lfloor 0.8 \cdot N \rfloor, \quad |\mathcal{D}_{test}| = N - |\mathcal{D}_{train}| \tag{14}$$

where $N = 109{,}357$ is the total number of modelling samples.

#### Algorithm 1 — Linear Regression

**Algorithm 1: Ordinary Least Squares Linear Regression**
```
INPUT : Training set {(xᵢ, yᵢ)}ᵢ₌₁ᴺ, feature matrix X ∈ ℝᴺˣ⁷, target y ∈ ℝᴺ
OUTPUT: Coefficient vector β ∈ ℝ⁸ (7 features + intercept)

1. Augment X with a bias column: X̃ ← [1 | X]  (shape N × 8)
2. Compute the OLS closed-form solution:
       β̂ = (X̃ᵀ X̃)⁻¹ X̃ᵀ y
3. For each test sample xᵢ ∈ D_test:
       ŷᵢ ← β̂₀ + Σⱼ β̂ⱼ · xᵢⱼ
4. Compute evaluation metrics RMSE, MAE, R²
5. RETURN β̂, {ŷᵢ}
```

**Explanation:** Linear Regression models the target price as a linear combination of the 7 input features. The optimal weights $\hat{\beta}$ are obtained analytically by minimising the Residual Sum of Squares (RSS):

$$\hat{\beta} = \arg\min_{\beta} \sum_{i=1}^{N} (y_i - \mathbf{x}_i^T \beta)^2 = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y} \tag{15}$$

The predicted price for a new sample $\mathbf{x}^*$ is:

$$\hat{y}^* = \beta_0 + \sum_{j=1}^{7} \beta_j x_j^* \tag{16}$$

---

#### Algorithm 2 — Decision Tree Regressor

**Algorithm 2: Decision Tree Regression with CART (max_depth = 5)**
```
INPUT : Training set {(xᵢ, yᵢ)}, max_depth = 5, min_samples_split = 2
OUTPUT: Trained tree T with leaf predictions

FUNCTION BuildTree(S, depth):
1.  IF depth = max_depth OR |S| < min_samples_split:
        RETURN LeafNode( value = mean({yᵢ : i ∈ S}) )
2.  best_feature ← NULL;  best_threshold ← NULL;  best_MSE ← ∞
3.  FOR each feature j ∈ {1, …, 7}:
        FOR each threshold t in sorted unique values of xⱼ:
            S_L ← {i ∈ S : xᵢⱼ ≤ t}
            S_R ← {i ∈ S : xᵢⱼ > t}
            mse ← (|S_L| · MSE(S_L) + |S_R| · MSE(S_R)) / |S|
            IF mse < best_MSE:
                best_MSE ← mse;  best_feature ← j;  best_threshold ← t
4.  Create InternalNode(feature = best_feature, threshold = best_threshold)
5.  node.left  ← BuildTree(S_L, depth + 1)
6.  node.right ← BuildTree(S_R, depth + 1)
7.  RETURN node

MAIN:
T ← BuildTree(D_train, depth = 0)
FOR each xᵢ ∈ D_test:
    ŷᵢ ← Traverse(T, xᵢ)   // follow splits until leaf, return leaf mean
RETURN T, {ŷᵢ}
```

**Explanation:** The Decision Tree uses the CART (Classification and Regression Trees) algorithm. At each internal node, it selects the feature $j$ and threshold $t$ that minimises the weighted Mean Squared Error across the resulting child partitions:

$$\text{MSE}_{split} = \frac{|S_L|}{|S|} \cdot \text{MSE}(S_L) + \frac{|S_R|}{|S|} \cdot \text{MSE}(S_R) \tag{17}$$

where the MSE of a partition $S_k$ is:

$$\text{MSE}(S_k) = \frac{1}{|S_k|} \sum_{i \in S_k} (y_i - \bar{y}_{S_k})^2, \quad \bar{y}_{S_k} = \frac{1}{|S_k|} \sum_{i \in S_k} y_i \tag{18}$$

The prediction for a test sample is the mean price of all training samples that fall in the same leaf node as that sample. The feature importance $I_j$ for feature $j$ across all nodes $t$ in the tree is:

$$I_j = \sum_{t : \text{split on } j} \frac{|S_t|}{N} \left[ \text{MSE}(S_t) - \frac{|S_{t,L}|}{|S_t|} \text{MSE}(S_{t,L}) - \frac{|S_{t,R}|}{|S_t|} \text{MSE}(S_{t,R}) \right] \tag{19}$$

---

### 3.7 Module 5 — Model Evaluation and Comparison

**Module Title:** Performance Evaluation Using Regression Metrics

**Description:**
Module 5 evaluates and compares both trained models on the held-out test set $\mathcal{D}_{test}$ using three standard regression metrics. All metrics are computed on the same 21,872 test samples to ensure a fair comparison. The three metrics — RMSE, MAE, and R² — capture different aspects of model quality: RMSE penalises large errors more heavily, MAE gives the average absolute deviation in interpretable price units (R$), and R² measures what fraction of price variance the model explains.

The Root Mean Squared Error (RMSE) is:

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} \tag{20}$$

The Mean Absolute Error (MAE) is:

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| \tag{21}$$

The coefficient of determination $R^2$ is:

$$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2} \tag{22}$$

where
```


## License: MIT
https://github.com/mlouarra/Chicago_Crime/blob/48e85467cf8d124e7e1aa84cd7b83d00f52fa51f/docs/Cahier_des_charges_mlops.tex

```
Here is the complete content for your report sections. Copy these directly into your document.

---

## 3. PROPOSED METHODOLOGY

### 3.1 Overview of the Proposed System

The proposed system is a **Machine Learning-based Dynamic Pricing Framework (MLPF)** designed to predict optimal product prices in e-commerce environments using historical transaction data. Unlike static rule-based pricing systems, the proposed framework leverages supervised learning algorithms to model the complex, non-linear relationships between product characteristics, customer behaviour, logistics features, and observed market prices. The system ingests raw transactional data from multiple sources, processes and engineers meaningful features, and produces price predictions that can guide real-time pricing decisions for online sellers.

The framework is built on the Brazilian Olist E-Commerce dataset comprising over 110,000 delivered orders spanning September 2016 to August 2018. The data pipeline integrates eight relational CSV files into a unified master dataframe, applies feature engineering to derive pricing-relevant signals, and trains two baseline supervised learning models — Linear Regression and Decision Tree Regressor — to benchmark price prediction accuracy. The models are evaluated on a held-out test set and compared using standard regression metrics, establishing a performance baseline that motivates the adoption of advanced ensemble models in subsequent project phases.

---

### 3.2 System Architecture

The system architecture, illustrated in **Figure 1**, follows an end-to-end supervised machine learning pipeline organised into five sequential processing modules. Raw data from the Olist platform enters the system at the Data Ingestion layer, passes through preprocessing and feature engineering stages, and finally reaches the model training and evaluation layer where predictions and performance metrics are generated.

```
┌─────────────────────────────────────────────────────────────────────┐
│                  PROPOSED SYSTEM ARCHITECTURE                       │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │  MODULE 1    │    │  MODULE 2    │    │      MODULE 3         │  │
│  │  Data        │───▶│  Data        │───▶│   Feature             │  │
│  │  Ingestion   │    │  Preprocessing│   │   Engineering         │  │
│  │  (8 CSVs)    │    │  & Merging   │    │   & Selection         │  │
│  └──────────────┘    └──────────────┘    └───────────┬───────────┘  │
│                                                       │             │
│  ┌──────────────┐    ┌──────────────┐                │             │
│  │  MODULE 5    │    │  MODULE 4    │◀───────────────┘             │
│  │  Model       │◀───│  Model       │                              │
│  │  Evaluation  │    │  Training    │                              │
│  │  & Comparison│    │  LR + DT     │                              │
│  └──────────────┘    └──────────────┘                              │
└─────────────────────────────────────────────────────────────────────┘
```
**Figure 1:** System Architecture of the Proposed Machine Learning-based Dynamic Pricing Framework (MLPF).

---

### 3.3 Module 1 — Data Ingestion and Integration

**Module Title:** Multi-Source Data Ingestion and Relational Integration

**Description:**
The first module is responsible for loading, parsing, and relationally joining eight separate CSV files sourced from the Olist Brazilian E-Commerce platform. Each file encapsulates a distinct domain of the e-commerce transaction lifecycle: order metadata, order items, payment records, customer reviews, product attributes, customer profiles, seller profiles, and category translations. The eight datasets are merged through a sequence of left joins on common key fields (`order_id`, `product_id`, `customer_id`, `seller_id`), producing a single denormalised master dataframe that serves as the input to all subsequent modules.

Timestamp fields are parsed into datetime objects to enable temporal feature derivation. Payment records for the same order are aggregated using summation for the total payment value and the mode for payment type. Review records are aggregated per order using the mean review score. Product categories are translated from Portuguese to English using the translation lookup table. After integration, a quality filter is applied to retain only rows corresponding to **delivered orders** with prices in the range (0, 5000] R$.

Let $D = \{d_1, d_2, \ldots, d_8\}$ denote the set of eight source datasets. The resulting master dataframe $M$ is defined as:

$$M = d_1 \bowtie_{order\_id} d_2 \bowtie_{order\_id} d_3' \bowtie_{order\_id} d_4' \bowtie_{product\_id} d_5' \bowtie_{customer\_id} d_6 \bowtie_{seller\_id} d_7 \tag{1}$$

where $d_3'$, $d_4'$, $d_5'$ denote the aggregated payment, review, and translated product tables respectively, and $\bowtie$ denotes a left equi-join operation.

The payment aggregation for order $o$ is:

$$P_{total}(o) = \sum_{i=1}^{n_o} p_i \tag{2}$$

$$P_{installments}(o) = \max_{i=1}^{n_o}(k_i) \tag{3}$$

where $p_i$ is the payment value and $k_i$ is the number of installments for the $i$-th payment record of order $o$, and $n_o$ is the total payment records for order $o$.

The review score aggregation is:

$$R(o) = \frac{1}{m_o} \sum_{j=1}^{m_o} r_j \tag{4}$$

where $r_j$ is the $j$-th review score for order $o$ and $m_o$ is the number of reviews.

The quality filter applied to the master dataframe is:

$$M_{filtered} = \{x \in M \mid status(x) = \text{"delivered"} \wedge 0 < price(x) \leq 5000\} \tag{5}$$

---

### 3.4 Module 2 — Data Preprocessing

**Module Title:** Temporal Parsing, Missing Value Handling, and Data Normalisation

**Description:**
Module 2 processes the raw master dataframe to prepare it for feature engineering. Three key preprocessing operations are performed: timestamp parsing, missing value imputation, and outlier-aware filtering.

Timestamp columns (`order_purchase_timestamp`, `order_delivered_customer_date`, `order_estimated_delivery_date`) are parsed from string representation into Python `datetime` objects using the `errors="coerce"` strategy, which converts unparseable values to `NaT` (Not a Time) rather than raising exceptions. This preserves data integrity while allowing downstream temporal arithmetic.

Missing values in the `delivery_delay` column (arising from undelivered orders already filtered out) are imputed with zero. Rows with missing values in the selected feature set are dropped during the modelling phase. The final filtered dataset retains **109,357 rows** out of 110,194 after removing records with NaN in any model feature.

The imputation strategy for delivery delay is:

$$\delta_{delay}(o) = \begin{cases} t_{delivered}(o) - t_{estimated}(o) & \text{if } t_{delivered}(o) \text{ is known} \\ 0 & \text{otherwise} \end{cases} \tag{6}$$

where $t_{delivered}(o)$ and $t_{estimated}(o)$ are the actual and estimated delivery timestamps for order $o$.

The freight ratio, a normalised measure of shipping cost relative to product price, is computed as:

$$f_{ratio}(o) = \frac{freight\_value(o)}{price(o) + \epsilon} \tag{7}$$

where $\epsilon = 10^{-3}$ is a small constant added to prevent division by zero.

---

### 3.5 Module 3 — Feature Engineering and Selection

**Module Title:** Temporal, Logistic, and Categorical Feature Derivation

**Description:**
Module 3 derives a rich set of explanatory features from the raw merged columns. Features are grouped into three categories: temporal features extracted from timestamps, logistic features derived from delivery information, and encoded categorical features.

**Temporal features** capture the time-based demand patterns known to influence pricing. The purchase month captures seasonality, day-of-week captures weekly demand cycles, and a binary weekend flag distinguishes weekday from weekend purchasing behaviour.

**Logistic features** capture the cost and efficiency of order fulfilment. The actual days to deliver is a proxy for distance and product type. The delivery delay captures fulfilment reliability.

**Categorical encoding** uses Label Encoding to convert the 72-class `category_en` feature into an integer representation. While ordinal encoding introduces implicit ordering, it allows tree-based models (which partition on thresholds) to use this feature effectively.

The temporal features are defined as:

$$month(o) = \text{month}(t_{purchase}(o)) \in \{1, 2, \ldots, 12\} \tag{8}$$

$$dow(o) = \text{dayofweek}(t_{purchase}(o)) \in \{0, 1, \ldots, 6\} \tag{9}$$

$$is\_weekend(o) = \mathbb{1}[dow(o) \in \{5, 6\}] \tag{10}$$

The logistic features are:

$$days\_to\_deliver(o) = \lfloor t_{delivered}(o) - t_{purchase}(o) \rfloor_{days} \tag{11}$$

The label encoding function for category $c_i$ is:

$$\text{encode}(c_i) = \text{rank}(c_i) \in \{0, 1, \ldots, C-1\} \tag{12}$$

where $C = 72$ is the total number of unique product categories and $\text{rank}(c_i)$ is the alphabetical rank of category $c_i$.

The final feature vector for each order $o$ is:

$$\mathbf{x}(o) = [f_{freight}, r_{score}, k_{install}, d_{deliver}, m_{purchase}, w_{end}, c_{encoded}]^T \in \mathbb{R}^7 \tag{13}$$

---

### 3.6 Module 4 — Model Training

**Module Title:** Supervised Regression Model Training — Linear Regression and Decision Tree

**Description:**
Module 4 trains two supervised regression models on the processed feature matrix to predict the target variable `price`. The dataset is partitioned into training (80%) and testing (20%) subsets using a fixed random seed for reproducibility. A helper `evaluate()` function computes three metrics after each model's predictions are obtained.

The train-test split is formalised as:

$$|\mathcal{D}_{train}| = \lfloor 0.8 \cdot N \rfloor, \quad |\mathcal{D}_{test}| = N - |\mathcal{D}_{train}| \tag{14}$$

where $N = 109{,}357$ is the total number of modelling samples.

#### Algorithm 1 — Linear Regression

**Algorithm 1: Ordinary Least Squares Linear Regression**
```
INPUT : Training set {(xᵢ, yᵢ)}ᵢ₌₁ᴺ, feature matrix X ∈ ℝᴺˣ⁷, target y ∈ ℝᴺ
OUTPUT: Coefficient vector β ∈ ℝ⁸ (7 features + intercept)

1. Augment X with a bias column: X̃ ← [1 | X]  (shape N × 8)
2. Compute the OLS closed-form solution:
       β̂ = (X̃ᵀ X̃)⁻¹ X̃ᵀ y
3. For each test sample xᵢ ∈ D_test:
       ŷᵢ ← β̂₀ + Σⱼ β̂ⱼ · xᵢⱼ
4. Compute evaluation metrics RMSE, MAE, R²
5. RETURN β̂, {ŷᵢ}
```

**Explanation:** Linear Regression models the target price as a linear combination of the 7 input features. The optimal weights $\hat{\beta}$ are obtained analytically by minimising the Residual Sum of Squares (RSS):

$$\hat{\beta} = \arg\min_{\beta} \sum_{i=1}^{N} (y_i - \mathbf{x}_i^T \beta)^2 = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y} \tag{15}$$

The predicted price for a new sample $\mathbf{x}^*$ is:

$$\hat{y}^* = \beta_0 + \sum_{j=1}^{7} \beta_j x_j^* \tag{16}$$

---

#### Algorithm 2 — Decision Tree Regressor

**Algorithm 2: Decision Tree Regression with CART (max_depth = 5)**
```
INPUT : Training set {(xᵢ, yᵢ)}, max_depth = 5, min_samples_split = 2
OUTPUT: Trained tree T with leaf predictions

FUNCTION BuildTree(S, depth):
1.  IF depth = max_depth OR |S| < min_samples_split:
        RETURN LeafNode( value = mean({yᵢ : i ∈ S}) )
2.  best_feature ← NULL;  best_threshold ← NULL;  best_MSE ← ∞
3.  FOR each feature j ∈ {1, …, 7}:
        FOR each threshold t in sorted unique values of xⱼ:
            S_L ← {i ∈ S : xᵢⱼ ≤ t}
            S_R ← {i ∈ S : xᵢⱼ > t}
            mse ← (|S_L| · MSE(S_L) + |S_R| · MSE(S_R)) / |S|
            IF mse < best_MSE:
                best_MSE ← mse;  best_feature ← j;  best_threshold ← t
4.  Create InternalNode(feature = best_feature, threshold = best_threshold)
5.  node.left  ← BuildTree(S_L, depth + 1)
6.  node.right ← BuildTree(S_R, depth + 1)
7.  RETURN node

MAIN:
T ← BuildTree(D_train, depth = 0)
FOR each xᵢ ∈ D_test:
    ŷᵢ ← Traverse(T, xᵢ)   // follow splits until leaf, return leaf mean
RETURN T, {ŷᵢ}
```

**Explanation:** The Decision Tree uses the CART (Classification and Regression Trees) algorithm. At each internal node, it selects the feature $j$ and threshold $t$ that minimises the weighted Mean Squared Error across the resulting child partitions:

$$\text{MSE}_{split} = \frac{|S_L|}{|S|} \cdot \text{MSE}(S_L) + \frac{|S_R|}{|S|} \cdot \text{MSE}(S_R) \tag{17}$$

where the MSE of a partition $S_k$ is:

$$\text{MSE}(S_k) = \frac{1}{|S_k|} \sum_{i \in S_k} (y_i - \bar{y}_{S_k})^2, \quad \bar{y}_{S_k} = \frac{1}{|S_k|} \sum_{i \in S_k} y_i \tag{18}$$

The prediction for a test sample is the mean price of all training samples that fall in the same leaf node as that sample. The feature importance $I_j$ for feature $j$ across all nodes $t$ in the tree is:

$$I_j = \sum_{t : \text{split on } j} \frac{|S_t|}{N} \left[ \text{MSE}(S_t) - \frac{|S_{t,L}|}{|S_t|} \text{MSE}(S_{t,L}) - \frac{|S_{t,R}|}{|S_t|} \text{MSE}(S_{t,R}) \right] \tag{19}$$

---

### 3.7 Module 5 — Model Evaluation and Comparison

**Module Title:** Performance Evaluation Using Regression Metrics

**Description:**
Module 5 evaluates and compares both trained models on the held-out test set $\mathcal{D}_{test}$ using three standard regression metrics. All metrics are computed on the same 21,872 test samples to ensure a fair comparison. The three metrics — RMSE, MAE, and R² — capture different aspects of model quality: RMSE penalises large errors more heavily, MAE gives the average absolute deviation in interpretable price units (R$), and R² measures what fraction of price variance the model explains.

The Root Mean Squared Error (RMSE) is:

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} \tag{20}$$

The Mean Absolute Error (MAE) is:

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| \tag{21}$$

The coefficient of determination $R^2$ is:

$$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2} \tag{22}$$

where
```


## License: unknown
https://github.com/oracle/tribuo-site/blob/5fa2491d60dc4181efca3439883e19aa1134aec2/learn/4.0/tutorials/regression-tribuo-v4.html

```
Here is the complete content for your report sections. Copy these directly into your document.

---

## 3. PROPOSED METHODOLOGY

### 3.1 Overview of the Proposed System

The proposed system is a **Machine Learning-based Dynamic Pricing Framework (MLPF)** designed to predict optimal product prices in e-commerce environments using historical transaction data. Unlike static rule-based pricing systems, the proposed framework leverages supervised learning algorithms to model the complex, non-linear relationships between product characteristics, customer behaviour, logistics features, and observed market prices. The system ingests raw transactional data from multiple sources, processes and engineers meaningful features, and produces price predictions that can guide real-time pricing decisions for online sellers.

The framework is built on the Brazilian Olist E-Commerce dataset comprising over 110,000 delivered orders spanning September 2016 to August 2018. The data pipeline integrates eight relational CSV files into a unified master dataframe, applies feature engineering to derive pricing-relevant signals, and trains two baseline supervised learning models — Linear Regression and Decision Tree Regressor — to benchmark price prediction accuracy. The models are evaluated on a held-out test set and compared using standard regression metrics, establishing a performance baseline that motivates the adoption of advanced ensemble models in subsequent project phases.

---

### 3.2 System Architecture

The system architecture, illustrated in **Figure 1**, follows an end-to-end supervised machine learning pipeline organised into five sequential processing modules. Raw data from the Olist platform enters the system at the Data Ingestion layer, passes through preprocessing and feature engineering stages, and finally reaches the model training and evaluation layer where predictions and performance metrics are generated.

```
┌─────────────────────────────────────────────────────────────────────┐
│                  PROPOSED SYSTEM ARCHITECTURE                       │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │  MODULE 1    │    │  MODULE 2    │    │      MODULE 3         │  │
│  │  Data        │───▶│  Data        │───▶│   Feature             │  │
│  │  Ingestion   │    │  Preprocessing│   │   Engineering         │  │
│  │  (8 CSVs)    │    │  & Merging   │    │   & Selection         │  │
│  └──────────────┘    └──────────────┘    └───────────┬───────────┘  │
│                                                       │             │
│  ┌──────────────┐    ┌──────────────┐                │             │
│  │  MODULE 5    │    │  MODULE 4    │◀───────────────┘             │
│  │  Model       │◀───│  Model       │                              │
│  │  Evaluation  │    │  Training    │                              │
│  │  & Comparison│    │  LR + DT     │                              │
│  └──────────────┘    └──────────────┘                              │
└─────────────────────────────────────────────────────────────────────┘
```
**Figure 1:** System Architecture of the Proposed Machine Learning-based Dynamic Pricing Framework (MLPF).

---

### 3.3 Module 1 — Data Ingestion and Integration

**Module Title:** Multi-Source Data Ingestion and Relational Integration

**Description:**
The first module is responsible for loading, parsing, and relationally joining eight separate CSV files sourced from the Olist Brazilian E-Commerce platform. Each file encapsulates a distinct domain of the e-commerce transaction lifecycle: order metadata, order items, payment records, customer reviews, product attributes, customer profiles, seller profiles, and category translations. The eight datasets are merged through a sequence of left joins on common key fields (`order_id`, `product_id`, `customer_id`, `seller_id`), producing a single denormalised master dataframe that serves as the input to all subsequent modules.

Timestamp fields are parsed into datetime objects to enable temporal feature derivation. Payment records for the same order are aggregated using summation for the total payment value and the mode for payment type. Review records are aggregated per order using the mean review score. Product categories are translated from Portuguese to English using the translation lookup table. After integration, a quality filter is applied to retain only rows corresponding to **delivered orders** with prices in the range (0, 5000] R$.

Let $D = \{d_1, d_2, \ldots, d_8\}$ denote the set of eight source datasets. The resulting master dataframe $M$ is defined as:

$$M = d_1 \bowtie_{order\_id} d_2 \bowtie_{order\_id} d_3' \bowtie_{order\_id} d_4' \bowtie_{product\_id} d_5' \bowtie_{customer\_id} d_6 \bowtie_{seller\_id} d_7 \tag{1}$$

where $d_3'$, $d_4'$, $d_5'$ denote the aggregated payment, review, and translated product tables respectively, and $\bowtie$ denotes a left equi-join operation.

The payment aggregation for order $o$ is:

$$P_{total}(o) = \sum_{i=1}^{n_o} p_i \tag{2}$$

$$P_{installments}(o) = \max_{i=1}^{n_o}(k_i) \tag{3}$$

where $p_i$ is the payment value and $k_i$ is the number of installments for the $i$-th payment record of order $o$, and $n_o$ is the total payment records for order $o$.

The review score aggregation is:

$$R(o) = \frac{1}{m_o} \sum_{j=1}^{m_o} r_j \tag{4}$$

where $r_j$ is the $j$-th review score for order $o$ and $m_o$ is the number of reviews.

The quality filter applied to the master dataframe is:

$$M_{filtered} = \{x \in M \mid status(x) = \text{"delivered"} \wedge 0 < price(x) \leq 5000\} \tag{5}$$

---

### 3.4 Module 2 — Data Preprocessing

**Module Title:** Temporal Parsing, Missing Value Handling, and Data Normalisation

**Description:**
Module 2 processes the raw master dataframe to prepare it for feature engineering. Three key preprocessing operations are performed: timestamp parsing, missing value imputation, and outlier-aware filtering.

Timestamp columns (`order_purchase_timestamp`, `order_delivered_customer_date`, `order_estimated_delivery_date`) are parsed from string representation into Python `datetime` objects using the `errors="coerce"` strategy, which converts unparseable values to `NaT` (Not a Time) rather than raising exceptions. This preserves data integrity while allowing downstream temporal arithmetic.

Missing values in the `delivery_delay` column (arising from undelivered orders already filtered out) are imputed with zero. Rows with missing values in the selected feature set are dropped during the modelling phase. The final filtered dataset retains **109,357 rows** out of 110,194 after removing records with NaN in any model feature.

The imputation strategy for delivery delay is:

$$\delta_{delay}(o) = \begin{cases} t_{delivered}(o) - t_{estimated}(o) & \text{if } t_{delivered}(o) \text{ is known} \\ 0 & \text{otherwise} \end{cases} \tag{6}$$

where $t_{delivered}(o)$ and $t_{estimated}(o)$ are the actual and estimated delivery timestamps for order $o$.

The freight ratio, a normalised measure of shipping cost relative to product price, is computed as:

$$f_{ratio}(o) = \frac{freight\_value(o)}{price(o) + \epsilon} \tag{7}$$

where $\epsilon = 10^{-3}$ is a small constant added to prevent division by zero.

---

### 3.5 Module 3 — Feature Engineering and Selection

**Module Title:** Temporal, Logistic, and Categorical Feature Derivation

**Description:**
Module 3 derives a rich set of explanatory features from the raw merged columns. Features are grouped into three categories: temporal features extracted from timestamps, logistic features derived from delivery information, and encoded categorical features.

**Temporal features** capture the time-based demand patterns known to influence pricing. The purchase month captures seasonality, day-of-week captures weekly demand cycles, and a binary weekend flag distinguishes weekday from weekend purchasing behaviour.

**Logistic features** capture the cost and efficiency of order fulfilment. The actual days to deliver is a proxy for distance and product type. The delivery delay captures fulfilment reliability.

**Categorical encoding** uses Label Encoding to convert the 72-class `category_en` feature into an integer representation. While ordinal encoding introduces implicit ordering, it allows tree-based models (which partition on thresholds) to use this feature effectively.

The temporal features are defined as:

$$month(o) = \text{month}(t_{purchase}(o)) \in \{1, 2, \ldots, 12\} \tag{8}$$

$$dow(o) = \text{dayofweek}(t_{purchase}(o)) \in \{0, 1, \ldots, 6\} \tag{9}$$

$$is\_weekend(o) = \mathbb{1}[dow(o) \in \{5, 6\}] \tag{10}$$

The logistic features are:

$$days\_to\_deliver(o) = \lfloor t_{delivered}(o) - t_{purchase}(o) \rfloor_{days} \tag{11}$$

The label encoding function for category $c_i$ is:

$$\text{encode}(c_i) = \text{rank}(c_i) \in \{0, 1, \ldots, C-1\} \tag{12}$$

where $C = 72$ is the total number of unique product categories and $\text{rank}(c_i)$ is the alphabetical rank of category $c_i$.

The final feature vector for each order $o$ is:

$$\mathbf{x}(o) = [f_{freight}, r_{score}, k_{install}, d_{deliver}, m_{purchase}, w_{end}, c_{encoded}]^T \in \mathbb{R}^7 \tag{13}$$

---

### 3.6 Module 4 — Model Training

**Module Title:** Supervised Regression Model Training — Linear Regression and Decision Tree

**Description:**
Module 4 trains two supervised regression models on the processed feature matrix to predict the target variable `price`. The dataset is partitioned into training (80%) and testing (20%) subsets using a fixed random seed for reproducibility. A helper `evaluate()` function computes three metrics after each model's predictions are obtained.

The train-test split is formalised as:

$$|\mathcal{D}_{train}| = \lfloor 0.8 \cdot N \rfloor, \quad |\mathcal{D}_{test}| = N - |\mathcal{D}_{train}| \tag{14}$$

where $N = 109{,}357$ is the total number of modelling samples.

#### Algorithm 1 — Linear Regression

**Algorithm 1: Ordinary Least Squares Linear Regression**
```
INPUT : Training set {(xᵢ, yᵢ)}ᵢ₌₁ᴺ, feature matrix X ∈ ℝᴺˣ⁷, target y ∈ ℝᴺ
OUTPUT: Coefficient vector β ∈ ℝ⁸ (7 features + intercept)

1. Augment X with a bias column: X̃ ← [1 | X]  (shape N × 8)
2. Compute the OLS closed-form solution:
       β̂ = (X̃ᵀ X̃)⁻¹ X̃ᵀ y
3. For each test sample xᵢ ∈ D_test:
       ŷᵢ ← β̂₀ + Σⱼ β̂ⱼ · xᵢⱼ
4. Compute evaluation metrics RMSE, MAE, R²
5. RETURN β̂, {ŷᵢ}
```

**Explanation:** Linear Regression models the target price as a linear combination of the 7 input features. The optimal weights $\hat{\beta}$ are obtained analytically by minimising the Residual Sum of Squares (RSS):

$$\hat{\beta} = \arg\min_{\beta} \sum_{i=1}^{N} (y_i - \mathbf{x}_i^T \beta)^2 = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y} \tag{15}$$

The predicted price for a new sample $\mathbf{x}^*$ is:

$$\hat{y}^* = \beta_0 + \sum_{j=1}^{7} \beta_j x_j^* \tag{16}$$

---

#### Algorithm 2 — Decision Tree Regressor

**Algorithm 2: Decision Tree Regression with CART (max_depth = 5)**
```
INPUT : Training set {(xᵢ, yᵢ)}, max_depth = 5, min_samples_split = 2
OUTPUT: Trained tree T with leaf predictions

FUNCTION BuildTree(S, depth):
1.  IF depth = max_depth OR |S| < min_samples_split:
        RETURN LeafNode( value = mean({yᵢ : i ∈ S}) )
2.  best_feature ← NULL;  best_threshold ← NULL;  best_MSE ← ∞
3.  FOR each feature j ∈ {1, …, 7}:
        FOR each threshold t in sorted unique values of xⱼ:
            S_L ← {i ∈ S : xᵢⱼ ≤ t}
            S_R ← {i ∈ S : xᵢⱼ > t}
            mse ← (|S_L| · MSE(S_L) + |S_R| · MSE(S_R)) / |S|
            IF mse < best_MSE:
                best_MSE ← mse;  best_feature ← j;  best_threshold ← t
4.  Create InternalNode(feature = best_feature, threshold = best_threshold)
5.  node.left  ← BuildTree(S_L, depth + 1)
6.  node.right ← BuildTree(S_R, depth + 1)
7.  RETURN node

MAIN:
T ← BuildTree(D_train, depth = 0)
FOR each xᵢ ∈ D_test:
    ŷᵢ ← Traverse(T, xᵢ)   // follow splits until leaf, return leaf mean
RETURN T, {ŷᵢ}
```

**Explanation:** The Decision Tree uses the CART (Classification and Regression Trees) algorithm. At each internal node, it selects the feature $j$ and threshold $t$ that minimises the weighted Mean Squared Error across the resulting child partitions:

$$\text{MSE}_{split} = \frac{|S_L|}{|S|} \cdot \text{MSE}(S_L) + \frac{|S_R|}{|S|} \cdot \text{MSE}(S_R) \tag{17}$$

where the MSE of a partition $S_k$ is:

$$\text{MSE}(S_k) = \frac{1}{|S_k|} \sum_{i \in S_k} (y_i - \bar{y}_{S_k})^2, \quad \bar{y}_{S_k} = \frac{1}{|S_k|} \sum_{i \in S_k} y_i \tag{18}$$

The prediction for a test sample is the mean price of all training samples that fall in the same leaf node as that sample. The feature importance $I_j$ for feature $j$ across all nodes $t$ in the tree is:

$$I_j = \sum_{t : \text{split on } j} \frac{|S_t|}{N} \left[ \text{MSE}(S_t) - \frac{|S_{t,L}|}{|S_t|} \text{MSE}(S_{t,L}) - \frac{|S_{t,R}|}{|S_t|} \text{MSE}(S_{t,R}) \right] \tag{19}$$

---

### 3.7 Module 5 — Model Evaluation and Comparison

**Module Title:** Performance Evaluation Using Regression Metrics

**Description:**
Module 5 evaluates and compares both trained models on the held-out test set $\mathcal{D}_{test}$ using three standard regression metrics. All metrics are computed on the same 21,872 test samples to ensure a fair comparison. The three metrics — RMSE, MAE, and R² — capture different aspects of model quality: RMSE penalises large errors more heavily, MAE gives the average absolute deviation in interpretable price units (R$), and R² measures what fraction of price variance the model explains.

The Root Mean Squared Error (RMSE) is:

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} \tag{20}$$

The Mean Absolute Error (MAE) is:

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| \tag{21}$$

The coefficient of determination $R^2$ is:

$$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2} \tag{22}$$

where
```

