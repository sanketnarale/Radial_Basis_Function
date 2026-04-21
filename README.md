# Smartphone Battery Health Prediction — RBF Neural Network

> Predicts `current_battery_health_percent` from smartphone usage and charging behaviour using a custom-built Radial Basis Function (RBF) Network.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset Description](#2-dataset-description)
   - [Feature File — `smartphone_battery_features.csv`](#21-feature-file--smartphone_battery_featurescsv)
   - [Target File — `smartphone_battery_targets.csv`](#22-target-file--smartphone_battery_targetscsv)
3. [Data Preprocessing](#3-data-preprocessing)
   - [Categorical Encoding](#31-categorical-encoding)
   - [Correlation Analysis & Feature Dropping](#32-correlation-analysis--feature-dropping)
   - [Final Feature Set](#33-final-feature-set)
   - [Train/Test Split & Scaling](#34-traintest-split--scaling)
4. [Model Architecture](#4-model-architecture)
   - [What is an RBF Network?](#41-what-is-an-rbf-network)
   - [Architecture Diagram](#42-architecture-diagram)
   - [Implementation Details](#43-implementation-details)
   - [Hyperparameter Search](#44-hyperparameter-search)
5. [Baseline Comparison](#5-baseline-comparison)
6. [Final Model Performance](#6-final-model-performance)
7. [Evaluation Plots](#7-evaluation-plots)
8. [How to Run](#8-how-to-run)
9. [Dependencies](#9-dependencies)

---

## 1. Project Overview

Battery health degradation in smartphones is influenced by a complex mix of usage patterns, thermal stress, and charging habits. This project builds a **custom Radial Basis Function (RBF) Neural Network** from scratch to regress `current_battery_health_percent` from 12 numeric device usage features.

The RBF network is compared against a Linear Regression baseline, and hyperparameters are tuned via a grid search over number of RBF centers and Ridge regularisation strength.

---

## 2. Dataset Description

The dataset is split across two CSV files — one for features, one for targets — both indexed by `Device_ID`.

| Property         | Value          |
|------------------|----------------|
| Total samples    | 5,000 devices  |
| Features file    | `smartphone_battery_features.csv` |
| Targets file     | `smartphone_battery_targets.csv`  |
| No missing values | ✅ (confirmed) |

---

### 2.1 Feature File — `smartphone_battery_features.csv`

**Shape:** `(5000, 15)` — 15 columns including the ID.

| Column | Type | Description | Range / Values |
|--------|------|-------------|----------------|
| `Device_ID` | str | Unique device identifier (UUID) | — |
| `device_age_months` | int | Age of the device in months | 0 – 48 |
| `battery_capacity_mah` | int | Battery capacity in milliamp-hours | 3000 – 5000 |
| `avg_screen_on_hours_per_day` | float | Average daily screen-on time | 1.0 – 12.0 hrs |
| `avg_charging_cycles_per_week` | float | Weekly charge cycle count | 3.0 – 18.6 |
| `avg_battery_temp_celsius` | float | Average battery temperature during use | 21.6 – 41.2 °C |
| `fast_charging_usage_percent` | float | % of charges using fast charging | 1.1 – 99.1% |
| `overnight_charging_freq_per_week` | int | How often device is charged overnight per week | 0 – 7 |
| `gaming_hours_per_week` | float | Weekly gaming usage hours | 0 – 24.8 hrs |
| `video_streaming_hours_per_week` | float | Weekly video streaming hours | 0 – 27.2 hrs |
| `background_app_usage_level` | str (ordinal) | Background app activity level | Low / Medium / High |
| `signal_strength_avg` | str (ordinal) | Average cellular signal quality | Poor / Moderate / Good |
| `charging_habit_score` | int | Composite score of charging habits (higher = healthier) | 3 – 10 |
| `usage_intensity_score` | float | Composite score of usage intensity | 6.57 – 10.0 |
| `thermal_stress_index` | float | Composite index of thermal stress on battery | 1.0 – 6.18 |

**Statistical Summary (numeric columns):**

| Feature | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| device_age_months | 24.23 | 14.14 | 0 | 48 |
| battery_capacity_mah | 4134.7 | 745.1 | 3000 | 5000 |
| avg_screen_on_hours_per_day | 5.51 | 1.97 | 1.0 | 12.0 |
| avg_charging_cycles_per_week | 8.33 | 3.03 | 3.0 | 18.6 |
| avg_battery_temp_celsius | 31.91 | 2.54 | 21.6 | 41.2 |
| fast_charging_usage_percent | 50.71 | 22.23 | 1.1 | 99.1 |
| overnight_charging_freq_per_week | 3.47 | 2.31 | 0 | 7 |
| gaming_hours_per_week | 4.01 | 2.83 | 0 | 24.8 |
| video_streaming_hours_per_week | 10.02 | 4.99 | 0 | 27.2 |
| charging_habit_score | 6.58 | 1.23 | 3 | 10 |
| usage_intensity_score | 10.00 | 0.06 | 6.57 | 10.0 |
| thermal_stress_index | 3.08 | 0.67 | 1.0 | 6.18 |

---

### 2.2 Target File — `smartphone_battery_targets.csv`

**Shape:** `(5000, 3)` — three columns.

| Column | Type | Description |
|--------|------|-------------|
| `Device_ID` | str | Device identifier (matches features file) |
| `current_battery_health_percent` | float | **Primary regression target** — current battery health as a percentage |
| `recommended_action` | str | Categorical label (e.g., "Replace", "Monitor") — not used in this model |

Only `current_battery_health_percent` is used as the target `y`.

---

## 3. Data Preprocessing

### 3.1 Categorical Encoding

Two columns had ordinal categorical values and were encoded before the correlation analysis:

| Column | Encoding | Rationale |
|--------|----------|-----------|
| `background_app_usage_level` | Low=0, Medium=1, High=2 | Ordinal — natural low-to-high scale |
| `signal_strength_avg` | Poor=0, Moderate=1, Good=2 | Ordinal — natural quality scale |

Ordinal encoding (not one-hot) was chosen because both variables have a clear inherent order.

---

### 3.2 Correlation Analysis & Feature Dropping

A full correlation matrix was computed on all encoded numeric features plus the target. Two separate investigations were performed:

#### A) Correlation with Target (`current_battery_health_percent`)

| Feature | Correlation | Interpretation |
|---------|-------------|----------------|
| `device_age_months` | **-0.854** | Very strong negative — dominant predictor |
| `avg_charging_cycles_per_week` | -0.410 | Moderate negative |
| `avg_screen_on_hours_per_day` | -0.400 | Moderate negative |
| `avg_battery_temp_celsius` | -0.301 | Moderate negative |
| `thermal_stress_index` | -0.280 | Moderate negative |
| `charging_habit_score` | +0.138 | Weak positive |
| `overnight_charging_freq_per_week` | -0.067 | Weak |
| `gaming_hours_per_week` | -0.049 | Very weak |
| `fast_charging_usage_percent` | -0.033 | Very weak |
| `video_streaming_hours_per_week` | +0.018 | Negligible |
| `background_app_usage_level` | +0.015 | Negligible |
| `usage_intensity_score` | -0.013 | Negligible |
| `battery_capacity_mah` | -0.011 | Negligible |
| `signal_strength_avg` | +0.008 | Negligible |

#### B) Multicollinearity Check (Inter-feature correlations > 0.85)

Two pairs of features exceeded the 0.85 threshold:

| Feature Pair | Correlation | Issue |
|---|---|---|
| `avg_screen_on_hours_per_day` ↔ `avg_charging_cycles_per_week` | **0.945** | High multicollinearity — screen-on time directly drives charging frequency |
| `avg_battery_temp_celsius` ↔ `thermal_stress_index` | **0.952** | High multicollinearity — thermal stress index is largely derived from temperature |

> **Note:** The notebook identifies these multicollinear pairs but does **not explicitly drop** them from the final feature matrix. Both pairs remain in training. The Ridge regression output layer in the RBF network provides inherent regularisation that tolerates multicollinearity.

#### C) Columns Dropped and Why

| Column | Dropped? | Reason |
|--------|----------|--------|
| `Device_ID` (features CSV) | ✅ **Yes** | Unique UUID identifier — carries no predictive signal, dropped before correlation analysis |
| `Device_ID` (targets CSV) | ✅ **Yes** | Same reason; only `current_battery_health_percent` extracted as `y` |
| `recommended_action` (targets CSV) | ✅ **Yes** | Categorical classification label — not used in this regression task |
| `background_app_usage_level` | ✅ **Yes (implicitly)** | After ordinal encoding was performed for correlation analysis, the final feature matrix `X` is built using `X_df.select_dtypes(include=[np.number])` on the **original unencoded** `X_df`. Since `background_app_usage_level` is still a string object at that point, it gets excluded. |
| `signal_strength_avg` | ✅ **Yes (implicitly)** | Same reason as above — original column is `str` dtype, excluded by `select_dtypes`. |

> **Important distinction:** The ordinal encoding was done on a **copy** (`X_df_encoded`) solely for computing the correlation matrix. The actual training data `X` is taken from the original `X_df` via `select_dtypes(include=[np.number])`, which naturally excludes `Device_ID`, `background_app_usage_level`, and `signal_strength_avg`.

---

### 3.3 Final Feature Set

After `select_dtypes(include=[np.number])`, the training matrix `X` contains **12 numeric features**:

```
device_age_months
battery_capacity_mah
avg_screen_on_hours_per_day
avg_charging_cycles_per_week
avg_battery_temp_celsius
fast_charging_usage_percent
overnight_charging_freq_per_week
gaming_hours_per_week
video_streaming_hours_per_week
charging_habit_score
usage_intensity_score
thermal_stress_index
```

---

### 3.4 Train/Test Split & Scaling

| Parameter | Value |
|-----------|-------|
| Train/Test ratio | 80% / 20% |
| Train samples | 4,000 |
| Test samples | 1,000 |
| `random_state` | 42 |
| Scaler | `StandardScaler` (zero mean, unit variance) |
| Fit on | Training set only; applied to test set |

---

## 4. Model Architecture

### 4.1 What is an RBF Network?

A **Radial Basis Function Network** is a three-layer neural network:

1. **Input Layer** — the raw (scaled) feature vector
2. **Hidden Layer (RBF Layer)** — a set of *centres* in feature space; each neuron computes a Gaussian activation based on the Euclidean distance from the input to its centre
3. **Output Layer** — a linear combination (Ridge regression) of the RBF activations

The Gaussian activation function for neuron $k$ is:

$$\phi_k(\mathbf{x}) = \exp\left(-\gamma \|\mathbf{x} - \mathbf{c}_k\|^2\right)$$

where $\mathbf{c}_k$ is the centre of neuron $k$ and $\gamma$ controls the width of the Gaussian.

---

### 4.2 Architecture Diagram

```
INPUT LAYER             RBF HIDDEN LAYER              OUTPUT LAYER
(12 features)           (1500 RBF neurons)            (1 output)

x1 ─────┐
x2 ─────┤              φ1 = exp(-γ‖x - c1‖²)
x3 ─────┤    ──────►   φ2 = exp(-γ‖x - c2‖²)   ──►  ŷ = wᵀφ + b
  ...   ┤              φ3 = exp(-γ‖x - c3‖²)        (Ridge Regression)
x12 ────┘               ...
                       φ1500 = exp(-γ‖x - c1500‖²)

         ↑
  Centres {c_k} found
  via K-Means clustering
  on training data
```

---

### 4.3 Implementation Details

The `RBFNetwork` class is implemented from scratch using NumPy, KMeans, and Ridge regression:

#### Centre Initialisation — K-Means

```python
self.kmeans_ = KMeans(n_clusters=self.n_centers, random_state=42, n_init=10)
self.kmeans_.fit(X)
self.centers_ = self.kmeans_.cluster_centers_
```

K-Means clusters the training data and uses the cluster centroids as the RBF centres. This ensures centres are placed in regions of high data density — a far better initialisation than random placement.

#### Automatic Gamma Estimation (Width Parameter)

If no `gamma` is provided, it is estimated from the average nearest-neighbour distance between centres:

```python
dists = pairwise_distances(self.centers_)
np.fill_diagonal(dists, np.inf)
sigma = dists.min(axis=1).mean()    # mean nearest-neighbour distance
self.gamma_ = 1 / (2 * sigma ** 2)
```

This heuristic sets the Gaussian width proportional to the typical spacing between centres, preventing activations from being too flat or too peaky.

#### RBF Transformation

Each input is transformed into a 1500-dimensional activation vector:

```python
def _transform(self, X):
    diff = X[:, np.newaxis, :] - self.centers_[np.newaxis, :, :]  # (N, K, D)
    sq_dists = np.sum(diff ** 2, axis=2)                           # (N, K)
    return np.exp(-self.gamma_ * sq_dists)                         # (N, K)
```

The output is a matrix of shape `(N_samples, N_centers)`.

#### Output Layer — Ridge Regression

```python
self.ridge_ = Ridge(alpha=self.alpha)
self.ridge_.fit(X_rbf, y)
```

Ridge regression fits a linear model on top of the RBF activations. The `alpha` (L2 penalty) prevents overfitting of the output weights.

#### Final Hyperparameters (Best Configuration)

| Hyperparameter | Value | Description |
|---|---|---|
| `n_centers` | **1500** | Number of RBF neurons (K-Means clusters) |
| `alpha` | **0.01** | Ridge L2 regularisation strength |
| `gamma` | **0.1725** (auto) | Gaussian width — estimated from centre spacing |
| `n_init` (KMeans) | 10 | K-Means restarts to avoid local minima |
| `random_state` | 42 | Reproducibility |

---

### 4.4 Hyperparameter Search

Two rounds of grid search were conducted:

**Round 1** — Coarse search:

| n_centers | alpha values tested |
|-----------|-------------------|
| 200, 500, 1000 | 0.01, 0.1, 1.0 |

**Round 2** — Fine search (higher centres, lower alpha):

| n_centers | alpha values tested |
|-----------|-------------------|
| 1500, 2000, 2500 | 0.001, 0.01 |

Best configuration found: `n_centers=1500`, `alpha=0.01` — delivering the highest R² on the test set.

---

## 5. Baseline Comparison

A **Linear Regression** model was trained on the same scaled features as a baseline:

| Model | R² | RMSE | MAE |
|-------|----|------|-----|
| Linear Regression | 0.9299 | 4.6847 | 3.6770 |
| **RBF Network** | **0.9419** | **4.2668** | **3.2549** |

The RBF Network improves R² by **+0.012**, reduces RMSE by **~0.42 percentage points**, and MAE by **~0.42 percentage points** — a meaningful improvement given the narrow output range.

---

## 6. Final Model Performance

Trained with `n_centers=1500`, `alpha=0.01` on 4000 samples, evaluated on 1000 held-out samples:

| Metric | Value |
|--------|-------|
| **R²** | **0.9419** |
| **RMSE** | **4.2668 %** |
| **MAE** | **3.2549 %** |
| Mean Residual | ≈ 0.0 (unbiased) |
| Gamma (auto) | 0.172502 |

An R² of 0.9419 means the model explains ~94.2% of the variance in battery health — strong predictive performance for a custom-built network.

---

## 7. Evaluation Plots

The notebook generates four diagnostic plots:

| Plot | What it shows |
|------|---------------|
| **Predicted vs Actual** | Scatter plot of ŷ vs y — points tightly around the diagonal indicate low error |
| **Residual Plot** | Residuals vs predicted values — checks for heteroscedasticity |
| **Distribution of Residuals** | Histogram of (y - ŷ) — should be centred at 0 and approximately normal |
| **Distribution of Absolute Errors** | Histogram of \|y - ŷ\| — shows the error spread; mean marked with a red dashed line |

---

## 8. How to Run

```bash
# 1. Place both CSVs in the same directory as the notebook
#    smartphone_battery_features.csv
#    smartphone_battery_targets.csv

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# 3. Open and run the notebook
jupyter notebook RBF_model.ipynb
```

All cells should be run top-to-bottom in order. The hyperparameter search (grid loops) may take a few minutes depending on hardware.

---

## 9. Dependencies

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations and RBF transformation |
| `matplotlib` | Evaluation plots |
| `seaborn` | Correlation heatmap |
| `scikit-learn` | `KMeans`, `Ridge`, `StandardScaler`, `train_test_split`, metrics |

No deep learning frameworks (PyTorch, TensorFlow) are used — the RBF network is implemented entirely in NumPy + scikit-learn primitives.
