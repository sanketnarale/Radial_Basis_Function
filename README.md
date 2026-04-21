# Smartphone Battery Health Prediction with a Custom RBF Network

This project predicts `current_battery_health_percent` from smartphone usage, charging, and thermal-behavior features using a custom Radial Basis Function (RBF) regression model implemented in `RBF_model.ipynb`.

The notebook also compares the RBF model against a linear regression baseline, performs correlation analysis, and evaluates the final model with residual diagnostics.

## Project Summary

| Item | Value |
| --- | --- |
| Model type | Custom RBF Network for regression |
| Notebook | `RBF_model.ipynb` |
| Feature file | `data/smartphone_battery_features.csv` |
| Target file | `data/smartphone_battery_targets.csv` |
| Total samples | 5,000 |
| Final training features | 12 numeric columns |
| Target | `current_battery_health_percent` |
| Best configuration | `n_centers=1500`, `alpha=0.01`, `gamma=auto` |
| Final test R2 | `0.9419` |
| Final test RMSE | `4.2668` |
| Final test MAE | `3.2549` |

## Dataset Description

The dataset is split into two CSV files connected by `Device_ID`.

### 1. Features Dataset

File: `data/smartphone_battery_features.csv`

Shape: `5000 x 15`

| Column | Type | Description |
| --- | --- | --- |
| `Device_ID` | object | Unique identifier for each device |
| `device_age_months` | int | Age of the phone in months |
| `battery_capacity_mah` | int | Battery capacity in mAh |
| `avg_screen_on_hours_per_day` | float | Average daily screen-on usage |
| `avg_charging_cycles_per_week` | float | Average weekly charge cycles |
| `avg_battery_temp_celsius` | float | Average battery temperature |
| `fast_charging_usage_percent` | float | Share of charging done via fast charging |
| `overnight_charging_freq_per_week` | int | Weekly overnight charging frequency |
| `gaming_hours_per_week` | float | Weekly gaming time |
| `video_streaming_hours_per_week` | float | Weekly video streaming time |
| `background_app_usage_level` | object | Background usage level: `Low`, `Medium`, `High` |
| `signal_strength_avg` | object | Signal quality: `Poor`, `Moderate`, `Good` |
| `charging_habit_score` | int | Aggregate charging-habit score |
| `usage_intensity_score` | float | Aggregate device-usage intensity score |
| `thermal_stress_index` | float | Aggregate thermal-stress indicator |

### 2. Target Dataset

File: `data/smartphone_battery_targets.csv`

Shape: `5000 x 3`

| Column | Type | Description |
| --- | --- | --- |
| `Device_ID` | object | Device identifier matching the features file |
| `current_battery_health_percent` | float | Regression target used in the notebook |
| `recommended_action` | object | Action label such as keep using or replace battery |

### 3. Basic Data Quality

The notebook checks for missing values and the dataset is complete:

- No null values in the features file
- No null values in the target file

### 4. Value Distributions Noted in the Notebook

The notebook explicitly inspects the two ordinal categorical columns before encoding:

| Column | Values observed |
| --- | --- |
| `background_app_usage_level` | `Low` (1707), `Medium` (1684), `High` (1609) |
| `signal_strength_avg` | `Good` (3025), `Moderate` (1509), `Poor` (466) |

The target file also includes a categorical recommendation column that is not used for regression:

| `recommended_action` distribution | Count |
| --- | --- |
| `Replace Battery` | 2325 |
| `Keep Using` | 1440 |
| `Change Phone` | 1235 |

## Numeric Dataset Summary

| Feature | Mean | Std | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| `device_age_months` | 24.2324 | 14.1380 | 0.0 | 48.0 |
| `battery_capacity_mah` | 4134.7000 | 745.0617 | 3000.0 | 5000.0 |
| `avg_screen_on_hours_per_day` | 5.5141 | 1.9747 | 1.0 | 12.0 |
| `avg_charging_cycles_per_week` | 8.3322 | 3.0345 | 3.0 | 18.6 |
| `avg_battery_temp_celsius` | 31.9074 | 2.5445 | 21.6 | 41.2 |
| `fast_charging_usage_percent` | 50.7124 | 22.2274 | 1.1 | 99.1 |
| `overnight_charging_freq_per_week` | 3.4652 | 2.3063 | 0.0 | 7.0 |
| `gaming_hours_per_week` | 4.0129 | 2.8343 | 0.0 | 24.8 |
| `video_streaming_hours_per_week` | 10.0207 | 4.9906 | 0.0 | 27.2 |
| `charging_habit_score` | 6.5820 | 1.2252 | 3.0 | 10.0 |
| `usage_intensity_score` | 9.9984 | 0.0585 | 6.57 | 10.0 |
| `thermal_stress_index` | 3.0794 | 0.6743 | 1.0 | 6.18 |
| `current_battery_health_percent` | 62.5956 | 17.7235 | 10.0 | 100.0 |

## Preprocessing Pipeline

### 1. Target Selection

The notebook extracts the regression target as:

```python
y = y_df["current_battery_health_percent"]
```

`recommended_action` is not used in modeling.

### 2. Ordinal Encoding Used for Correlation Analysis

The notebook creates a copy of the feature dataframe called `X_df_encoded` and applies ordinal mappings:

```python
background_mapping = {"Low": 0, "Medium": 1, "High": 2}
signal_mapping = {"Poor": 0, "Moderate": 1, "Good": 2}
```

This encoding is only used for the correlation matrix section.

### 3. Correlation Matrix Setup

Before computing correlations, the notebook removes only `Device_ID` from the encoded feature copy:

```python
X_for_analysis = X_df_encoded.drop(columns=["Device_ID"])
df_complete = pd.concat([X_for_analysis, y], axis=1)
correlation_matrix = df_complete.corr()
```

Reason:

- `Device_ID` is a unique identifier and carries no predictive pattern for battery health
- It must be removed because correlations are meaningful only for actual measured or engineered features

## Correlation Matrix Findings

The notebook performs two main correlation checks.

### 1. Correlation with the Target

`current_battery_health_percent` correlations sorted by absolute magnitude:

| Feature | Correlation with target |
| --- | ---: |
| `device_age_months` | -0.854140 |
| `avg_charging_cycles_per_week` | -0.409962 |
| `avg_screen_on_hours_per_day` | -0.399347 |
| `avg_battery_temp_celsius` | -0.300765 |
| `thermal_stress_index` | -0.280187 |
| `charging_habit_score` | 0.137861 |
| `overnight_charging_freq_per_week` | -0.067161 |
| `gaming_hours_per_week` | -0.049027 |
| `fast_charging_usage_percent` | -0.033425 |
| `video_streaming_hours_per_week` | 0.018082 |
| `background_app_usage_level` | 0.014968 |
| `usage_intensity_score` | -0.013157 |
| `battery_capacity_mah` | -0.011077 |
| `signal_strength_avg` | 0.008209 |

Interpretation from the notebook results:

- `device_age_months` is by far the strongest predictor and is strongly negatively associated with battery health
- charging frequency, screen-on time, temperature, and thermal stress all have meaningful negative relationships
- battery capacity and signal-strength-related effects are negligible in this dataset

### 2. Multicollinearity Check

The notebook flags feature pairs with absolute correlation greater than `0.85`.

| Feature 1 | Feature 2 | Correlation |
| --- | --- | ---: |
| `avg_screen_on_hours_per_day` | `avg_charging_cycles_per_week` | 0.945 |
| `avg_battery_temp_celsius` | `thermal_stress_index` | 0.952 |

These pairs are important because they carry overlapping information:

- more screen time naturally drives more charging cycles
- thermal stress index is strongly tied to battery temperature

## Columns Dropped and Why

This is the exact column-dropping story reflected by the notebook.

### Dropped for Correlation Analysis

| Column | Dropped | Why |
| --- | --- | --- |
| `Device_ID` from features | Yes | Identifier only; not a behavioral or physical predictor |

### Dropped from the Modeling Target Side

| Column | Dropped | Why |
| --- | --- | --- |
| `Device_ID` from targets | Yes | The model only uses the regression target column |
| `recommended_action` | Yes | This is a categorical label and is not part of the regression target |

### Dropped When Building the Final Training Matrix

Later, the notebook creates the actual model input with:

```python
X_df = X_df.select_dtypes(include=[np.number])
X = X_df.copy()
```

This means the final training data keeps only numeric columns from the original `X_df`.

| Column | Dropped | Why |
| --- | --- | --- |
| `Device_ID` | Yes | Object/string identifier |
| `background_app_usage_level` | Yes | Object/string column in the original dataframe |
| `signal_strength_avg` | Yes | Object/string column in the original dataframe |

Important note:

- The notebook does encode `background_app_usage_level` and `signal_strength_avg`, but that encoding lives in `X_df_encoded`, a separate copy used only for correlation analysis
- those encoded categorical columns are not fed into the final trained model

### Were High-Correlation Columns Dropped?

No. Even though the correlation matrix identifies two highly correlated feature pairs, the notebook does not remove any of them before training.

That means all four of these columns remain in the model input:

- `avg_screen_on_hours_per_day`
- `avg_charging_cycles_per_week`
- `avg_battery_temp_celsius`
- `thermal_stress_index`

## Final Feature Set Used by the Model

After `select_dtypes(include=[np.number])`, the model is trained on these 12 features:

1. `device_age_months`
2. `battery_capacity_mah`
3. `avg_screen_on_hours_per_day`
4. `avg_charging_cycles_per_week`
5. `avg_battery_temp_celsius`
6. `fast_charging_usage_percent`
7. `overnight_charging_freq_per_week`
8. `gaming_hours_per_week`
9. `video_streaming_hours_per_week`
10. `charging_habit_score`
11. `usage_intensity_score`
12. `thermal_stress_index`

## Train-Test Split and Scaling

The notebook uses:

```python
train_test_split(X, y, test_size=0.2, random_state=42)
```

So the split is:

- Training samples: 4,000
- Test samples: 1,000

Feature scaling is performed with `StandardScaler`:

- scaler is fit on the training set only
- the learned transformation is applied to both training and test sets

This is necessary because the RBF distance computation is scale-sensitive.

## Fully Detailed Model Architecture

The model is a custom RBF network implemented as a Python class named `RBFNetwork`.

### Architecture Overview

The network has three conceptual stages:

1. Input layer
2. Hidden RBF layer
3. Linear output layer with Ridge regularization

### 1. Input Layer

- Input dimension: 12
- Input data: standardized numeric features
- Shape entering the model: `(n_samples, 12)`

### 2. Hidden RBF Layer

The hidden layer is not a dense neural layer with learned backpropagation weights. Instead, it is built from cluster centers derived from the training data.

#### Center Selection

The notebook uses K-Means:

```python
self.kmeans_ = KMeans(n_clusters=self.n_centers, random_state=42, n_init=10)
self.kmeans_.fit(X)
self.centers_ = self.kmeans_.cluster_centers_
```

For the final model:

- number of centers: `1500`
- each center has dimension `12`
- center matrix shape: `(1500, 12)`

Each center acts like one RBF neuron.

#### Gaussian Activation

For each input vector `x` and center `c_k`, the hidden neuron output is:

```text
phi_k(x) = exp(-gamma * ||x - c_k||^2)
```

Where:

- `||x - c_k||^2` is the squared Euclidean distance between the input and center
- `gamma` controls the width of the Gaussian response

The implementation computes:

```python
diff = X[:, np.newaxis, :] - self.centers_[np.newaxis, :, :]
sq_dists = np.sum(diff ** 2, axis=2)
X_rbf = np.exp(-self.gamma_ * sq_dists)
```

Output of this layer:

- shape: `(n_samples, 1500)`
- each row contains similarity scores to all 1500 centers

#### Automatic Gamma Estimation

If `gamma` is not supplied manually, the notebook estimates it from center spacing:

```python
dists = pairwise_distances(self.centers_)
np.fill_diagonal(dists, np.inf)
sigma = dists.min(axis=1).mean()
self.gamma_ = 1 / (2 * sigma ** 2)
```

Meaning:

- compute pairwise distances between all centers
- ignore each center's distance to itself
- find the nearest neighboring center for each center
- average those nearest-neighbor distances to get `sigma`
- convert `sigma` into `gamma`

Final fitted value in this notebook run:

- `gamma = 0.1725022815344326`

### 3. Output Layer

The output layer is a Ridge regression model trained on the RBF-transformed features:

```python
self.ridge_ = Ridge(alpha=self.alpha)
self.ridge_.fit(X_rbf, y)
```

This means the model learns:

- one linear weight per RBF neuron
- plus the intercept term learned by Ridge regression

For the final model:

- output dimension: 1
- regularization type: L2
- `alpha = 0.01`

### End-to-End Flow

The full prediction pipeline is:

1. Start with 12 raw numeric input features
2. Standardize them with `StandardScaler`
3. Measure distance from each sample to 1500 K-Means centers
4. Convert distances into Gaussian activations
5. Feed the 1500-dimensional RBF representation into Ridge regression
6. Output predicted battery health percentage

### Architecture in Shape Form

```text
Input samples                  : (N, 12)
After StandardScaler           : (N, 12)
K-Means centers                : (1500, 12)
RBF-transformed representation : (N, 1500)
Ridge output                   : (N,)
```

## Hyperparameter Search in the Notebook

The notebook performs two search loops.

### Round 1

Tested:

- `n_centers`: `200`, `500`, `1000`
- `alpha`: `0.01`, `0.1`, `1.0`

Results:

| `n_centers` | `alpha` | Test R2 |
| ---: | ---: | ---: |
| 200 | 0.01 | 0.9312 |
| 200 | 0.1 | 0.9246 |
| 200 | 1.0 | 0.9069 |
| 500 | 0.01 | 0.9357 |
| 500 | 0.1 | 0.9316 |
| 500 | 1.0 | 0.9162 |
| 1000 | 0.01 | 0.9414 |
| 1000 | 0.1 | 0.9385 |
| 1000 | 1.0 | 0.9267 |

Best result from round 1:

- `n_centers=1000`
- `alpha=0.01`
- `R2=0.9414`

### Round 2

Tested:

- `n_centers`: `1500`, `2000`, `2500`
- `alpha`: `0.001`, `0.01`

Results:

| `n_centers` | `alpha` | Test R2 |
| ---: | ---: | ---: |
| 1500 | 0.001 | 0.9417 |
| 1500 | 0.01 | 0.9419 |
| 2000 | 0.001 | 0.9407 |
| 2000 | 0.01 | 0.9419 |
| 2500 | 0.001 | 0.9366 |
| 2500 | 0.01 | 0.9385 |

Best configuration selected in the notebook:

- `n_centers=1500`
- `alpha=0.01`

## Baseline Model

The notebook trains a linear regression baseline on the same scaled features.

Baseline metrics:

| Model | RMSE | MAE | R2 |
| --- | ---: | ---: | ---: |
| Linear Regression | 4.6847 | 3.6770 | 0.9299 |
| Final RBF Network | 4.2668 | 3.2549 | 0.9419 |

The RBF model outperforms the linear baseline across all reported metrics.

## Final Model Performance

Final fitted model:

```python
rbf_true = RBFNetwork(n_centers=1500, alpha=0.01)
rbf_true.fit(X_train_scaled, y_train)
```

Measured on the held-out test set:

| Metric | Value |
| --- | ---: |
| R2 | 0.9418546534 |
| RMSE | 4.2668144007 |
| MAE | 3.2548941381 |
| Mean residual | -0.0500630238 |
| Residual std | 4.2686555546 |
| Minimum absolute error | 0.0032498679 |
| Maximum absolute error | 29.8629651249 |

Interpretation:

- the model explains about 94.19% of the variance in battery health
- average prediction error is about 3.25 percentage points
- the residual mean is very close to zero, which suggests low overall bias

## Evaluation Visualizations Generated by the Notebook

The notebook creates four plots for final evaluation:

1. Predicted vs actual battery health
2. Residuals vs predicted values
3. Residual distribution histogram
4. Absolute error distribution histogram

These plots help check:

- overall fit quality
- bias and spread of residuals
- whether errors are centered near zero
- whether a few large misses dominate the error profile

## Important Implementation Notes

### 1. Encoded Categorical Columns Are Not Used in Final Training

This is the most important detail to know when reading the notebook:

- `background_app_usage_level` and `signal_strength_avg` are encoded only in `X_df_encoded`
- the actual training matrix is rebuilt later from the original dataframe using numeric-only selection
- therefore those two categorical variables do not participate in the final trained model

### 2. Multicollinearity Is Diagnosed, Not Resolved

The correlation matrix clearly identifies two strongly correlated feature pairs, but the notebook does not drop any of them before model fitting.

### 3. Notebook File Paths

The notebook reads:

```python
pd.read_csv("smartphone_battery_features.csv")
pd.read_csv("smartphone_battery_targets.csv")
```

In this repository, the CSV files are stored under `data/`, so running the notebook may require either:

- launching the notebook from the `RBF/data` context after adjusting paths, or
- updating the file paths inside the notebook to `data/smartphone_battery_features.csv` and `data/smartphone_battery_targets.csv`

## Dependencies

The notebook uses:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

## Conclusion

This notebook builds a strong battery-health regressor using a custom RBF architecture with:

- 12 standardized numeric input features
- 1500 Gaussian basis functions centered by K-Means
- an automatically estimated Gaussian width
- a Ridge regression output layer

The final model improves on linear regression and reaches an R2 of `0.9419` on the test set, with device age emerging as the strongest single predictor of battery health.
