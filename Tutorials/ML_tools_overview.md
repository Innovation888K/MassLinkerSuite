# Machine-learning utilities in `ML_tools.py`

This document provides an overview of the machine-learning utility functions implemented in `ML_tools.py`.

The `ML_tools.py` module provides helper tools for training conventional machine-learning models on MassLinker token datasets, saving fold-level prediction results, calculating validation metrics, plotting model-performance summaries, generating ROC curves, and performing SHAP-based feature interpretation.

This module is designed for downstream analysis after MassLinker metabolic tokens have been converted into Python dataset objects.

---

## Overview

The `ML_tools.py` module provides utilities for the following tasks:

1. Training conventional machine-learning classifiers on MassLinker token features.
2. Flattening MassLinker token tensors for model input.
3. Running fold-based model training and prediction.
4. Saving prediction results for each validation fold.
5. Calculating accuracy, precision, recall, F1-score, and AUC.
6. Visualizing cross-validation performance.
7. Plotting ROC curves for multiple models.
8. Importing additional external ROC results.
9. Preparing cross-dataset training and testing splits.
10. Performing SHAP-based feature-importance analysis.

---

## Main function groups

| Function group | Functions / Classes | Purpose |
|---|---|---|
| Machine-learning model wrapper | `ML_models` | Train and evaluate SVM, XGBoost, Random Forest, and LightGBM models |
| Fold handling | `fold_n`, `cross_dataset_data_prepare` | Prepare train/test indices or cross-dataset splits |
| Result saving | `save_model_results` | Save model predictions and true labels for each fold |
| Metric calculation | `init_indexes`, `cal_all_index` | Calculate fold-level metrics across models |
| Performance visualization | `box_plot`, `prepare_barplot_data`, `bar_plot`, `visualization` | Generate summary plots for model metrics |
| ROC utilities | `addition_roc`, `ML_models.plot_combined_roc` | Load external ROC results and plot combined ROC curves |
| SHAP interpretation | `UniversalSHAPAnalyzer` | Perform model-agnostic or model-specific SHAP feature interpretation |

---

## Machine-learning model wrapper

### `ML_models`

The `ML_models` class is the core machine-learning wrapper in `ML_tools.py`.

It initializes and manages four conventional machine-learning models:

| Model | Implementation |
|---|---|
| SVM | `sklearn.svm.SVC` |
| XGBoost | `xgboost.XGBClassifier` |
| Random Forest | `sklearn.ensemble.RandomForestClassifier` |
| LightGBM | `lightgbm.LGBMClassifier` |

The class provides methods for:

- loading train/test data;
- initializing models;
- fitting models;
- generating predictions;
- calculating validation metrics;
- generating prediction probabilities;
- plotting combined ROC curves.

MassLinker token tensors are flattened using `data_transform()` before being used as machine-learning input features.

---

### Input data format for `ML_models`

The `ML_models` class expects training and testing data derived from MassLinker dataset indexing.

In the current implementation, the data structure follows the output style of the original dataset object, where indexed data may contain:

| Index | Content |
|---:|---|
| `0` | Sample tensors |
| `1` | Labels |
| `2` | Sample names |
| `3` | Positive/negative indicators |
| `4` | Generated class IDs |

The training target depends on the selected fitting mode.

| Fit mode | Target index | Meaning |
|---|---:|---|
| `single` | `3` | Binary positive/negative classification |
| Other modes | `4` | Multi-class classification using generated class IDs |

---

## Model initialization

### `init_SVM`

Initializes an SVM classifier.

Default configuration:

| Parameter | Default |
|---|---|
| `kernel` | `rbf` |
| `C` | `1.0` |
| `gamma` | `scale` |

A `StandardScaler` object is also initialized for standardization-related workflows.

---

### `init_XGB`

Initializes an XGBoost classifier.

Default configuration includes:

| Parameter | Default |
|---|---|
| `max_depth` | `50` |
| `eta` | `0.001` |
| `num_round` | `200` |
| `objective` | `multi:softmax` |

The number of classes is determined from the training data.

---

### `init_RF`

Initializes a Random Forest classifier.

Default configuration:

| Parameter | Default |
|---|---|
| `n_estimators` | `1000` |
| `max_depth` | `None` |

---

### `init_lgb`

Initializes a LightGBM classifier for multiclass classification.

Default configuration includes:

| Parameter | Default |
|---|---|
| `objective` | `multiclass` |
| `metric` | `multi_logloss` |
| `n_estimators` | `200` |
| `learning_rate` | `0.0001` |
| `num_leaves` | `50` |
| `random_state` | `42` |

---

## Model training and prediction

### `fit_models`

The `fit_models()` method trains all four models:

1. Random Forest;
2. SVM;
3. XGBoost;
4. LightGBM.

Before fitting, MassLinker token tensors are flattened using:

```python
data_transform(...)
```

The target label is selected according to `fit_mode`.

| `fit_mode` | Target |
|---|---|
| `"single"` | `is_positive` |
| Other values | generated class IDs |

---

### `prediction`

The `prediction()` method generates predictions from the trained models.

Prediction results are stored in:

```text
self.predict_result
```

The order of stored predictions is:

1. SVM;
2. XGBoost;
3. Random Forest;
4. LightGBM.

---

### `validation`

The `validation()` method calculates and prints common classification metrics for each model.

Metrics include:

| Metric | Description |
|---|---|
| Accuracy | Overall prediction accuracy |
| Precision | Weighted precision |
| Recall | Weighted recall |
| F1-score | Weighted F1-score |
| Confusion matrix | Class-level prediction summary |

This function is mainly used for quick terminal-based validation.

---

## ROC analysis

### `predict_proba_all`

The `predict_proba_all()` method generates prediction probabilities for models that support probability output.

The returned probability dictionary includes:

| Key | Model |
|---|---|
| `XGB` | XGBoost |
| `RF` | Random Forest |
| `LGB` | LightGBM |

SVM is not included in this method because the current SVM initialization does not enable probability prediction.

---

### `plot_combined_roc`

The `plot_combined_roc()` method plots ROC curves for multiple models in one figure.

It supports:

- XGBoost ROC curve;
- Random Forest ROC curve;
- LightGBM ROC curve;
- optional external ROC results through `addition_ROCs`.

For binary classification, labels are binarized before ROC calculation.

The figure can either be shown directly or saved as a PDF file.

---

### `addition_roc`

The `addition_roc()` function loads external prediction results from CSV files and converts them into the format required by `plot_combined_roc()`.

Each input CSV file is expected to contain two columns:

| Column | Description |
|---|---|
| first column | True binary label |
| second column | Predicted probability for the positive class |

The returned object has the format:

```text
[(y_trues, y_scores, color, model_name), ...]
```

This is useful when comparing MassLinker machine-learning models with external models or previously saved prediction results.

---

## Fold handling and result saving

### `fold_n`

The `fold_n()` function prepares training and testing indices for fold-based validation.

Given a list of split indices and the selected fold number, it returns:

| Output | Description |
|---|---|
| first output | Test indices for the selected fold |
| second output | Training indices from all remaining folds |

This function is used to support n-fold cross-validation workflows.

---

### `save_model_results`

The `save_model_results()` function saves model predictions and true labels for one validation fold.

It writes an Excel file named:

```text
fold{fold_number}.xlsx
```

under a model-result directory.

The saved table contains:

- predictions from different models;
- true labels for the selected validation target.

These saved fold-level files can later be used for metric calculation and visualization.

---

### `cross_dataset_data_prepare`

The `cross_dataset_data_prepare()` function prepares cross-dataset training and testing data.

It takes a list-like object of loaded datasets and selects one part as the test set while concatenating the remaining parts as the training set.

It returns:

| Output | Description |
|---|---|
| `x_train` | Training sample tensors |
| `y_train` | Training labels |
| `x_test` | Testing sample tensors |
| `y_test` | Testing labels |

This function is useful for cohort-level validation or cross-dataset evaluation.

---

## Metric calculation

### `init_indexes`

The `init_indexes()` function initializes empty metric containers for the following models:

1. SVM;
2. XGBoost;
3. Random Forest;
4. LightGBM.

The initialized metrics include:

- accuracy;
- precision;
- recall;
- F1-score;
- ROC-related data.

---

### `cal_all_index`

The `cal_all_index()` function calculates metrics from fold-level prediction result tables.

For each fold and each model, it calculates:

| Metric | Description |
|---|---|
| Accuracy | Overall prediction accuracy |
| Precision | Weighted precision |
| Recall | Weighted recall |
| F1-score | Weighted F1-score |
| AUC | ROC-AUC calculated from predictions |

The output includes:

| Output | Description |
|---|---|
| `metrics_data` | Dictionary of model metrics across folds |
| `roc_data` | ROC-related data across folds |

---

## Performance visualization

### `box_plot`

The `box_plot()` function visualizes model metrics across folds using box plots.

Each metric is plotted separately.

This function is useful for inspecting performance variation across validation folds.

---

### `prepare_barplot_data`

The `prepare_barplot_data()` function converts a nested metric dictionary into a tidy DataFrame suitable for plotting.

The output DataFrame contains:

| Column | Description |
|---|---|
| `Model` | Model name |
| `Metric` | Metric name |
| `Value` | Metric value |

---

### `bar_plot`

The `bar_plot()` function generates bar plots for each metric.

It saves PDF figures under:

```text
barplots/{id}/
```

Each bar plot summarizes the average model performance for one metric.

In the current implementation, output file names include the suffix:

```text
_enhance.pdf
```

This reflects the current code behavior.

---

### `visualization`

The `visualization()` function provides a compact workflow for reading fold-level result files from a directory, calculating metrics, and generating bar plots.

It expects a directory containing fold-level Excel files.

The workflow is:

1. Read prediction result files.
2. Calculate metrics using `cal_all_index()`.
3. Prepare plotting data.
4. Generate bar plots using `bar_plot()`.

---

## SHAP feature interpretation

### `UniversalSHAPAnalyzer`

The `UniversalSHAPAnalyzer` class provides a general interface for SHAP-based feature interpretation.

It supports different model types by trying multiple SHAP explainers:

1. `TreeExplainer`;
2. `LinearExplainer`;
3. `KernelExplainer`.

This design allows the same interface to be used for tree-based models, linear models, and other black-box models.

---

### Main SHAP methods

| Method | Purpose |
|---|---|
| `create_explainer` | Create a suitable SHAP explainer for the model |
| `calculate_shap_values` | Calculate SHAP values for the input data |
| `plot_importance` | Plot stacked feature-importance bars |
| `get_feature_importance` | Return mean absolute SHAP values |
| `plot_waterfall` | Generate a SHAP waterfall plot for one sample |

---

### `create_explainer`

This method tries to create a SHAP explainer in the following order:

1. Tree-based SHAP explainer;
2. Linear SHAP explainer;
3. Kernel SHAP explainer.

If tree and linear explainers are not suitable, a sampled background dataset is used for `KernelExplainer`.

---

### `calculate_shap_values`

This method calculates SHAP values for the training data.

In the current implementation, if SHAP returns a two-dimensional value matrix, it is converted into a three-dimensional binary-class style representation.

This allows downstream importance plotting to handle class-specific feature contributions.

---

### `plot_importance`

This method generates a stacked horizontal bar plot of feature importance.

Feature importance is calculated from mean absolute SHAP values.

The plot is saved as:

```text
{model_name}_feature_importance.pdf
```

under the specified output directory.

---

### `get_feature_importance`

This method returns mean absolute SHAP values for features.

It can be used for downstream feature ranking, interpretation, or export.

---

### `plot_waterfall`

This method generates a SHAP waterfall plot for a selected sample.

The waterfall plot shows how individual features contribute to the model output for that sample.

---

## Dependencies

The `ML_tools.py` module depends on packages for machine learning, statistics, visualization, and model interpretation.

| Package | Purpose |
|---|---|
| `numpy` | Numerical operations |
| `pandas` | Reading, writing, and organizing tabular results |
| `torch` | Tensor operations and dataset handling |
| `scikit-learn` | SVM, Random Forest, metrics, scaling, ROC analysis |
| `xgboost` | XGBoost classifier |
| `lightgbm` | LightGBM classifier |
| `shap` | Model interpretation and feature attribution |
| `matplotlib` | Figure generation |
| `seaborn` | Statistical plotting |
| `joblib` | Loading and saving Python objects |

A typical installation command is:

```bash
pip install numpy pandas torch scikit-learn xgboost lightgbm shap matplotlib seaborn joblib openpyxl
```

---

## Notes

1. `ML_tools.py` is designed for downstream machine-learning analysis using MassLinker token datasets.
2. MassLinker token tensors are flattened using `data_transform()` before conventional machine-learning models are trained.
3. `ML_models` currently initializes SVM, XGBoost, Random Forest, and LightGBM classifiers.
4. The default SVM does not enable probability prediction, so it is not included in `predict_proba_all()`.
5. `fit_models(fit_mode="single")` uses `is_positive` as the classification target.
6. Other fitting modes use generated class IDs as the classification target.
7. Fold-level prediction results can be saved as Excel files using `save_model_results()`.
8. The visualization functions are mainly designed for fold-level validation result summaries.
9. `UniversalSHAPAnalyzer` provides a general SHAP interpretation interface for different model types.
10. Some output paths and filename suffixes, such as `barplots/{id}/` and `_enhance.pdf`, are fixed by the current implementation.
11. Detailed model-training case studies are provided in separate tutorials.
