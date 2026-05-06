# Machine-learning validation examples using MassLinker tokens

This tutorial demonstrates how to use `ML_tools.py` for machine-learning validation with MassLinker datasets.

The workflow covers fold-based validation, model-result visualization, external ROC comparison, saved-model reuse, and SHAP-based feature-importance analysis. It is designed for downstream supervised learning after MassLinker metabolic tokens have been converted into a Python dataset object.

---

## Overview

This tutorial focuses on practical usage examples for the machine-learning utilities provided in `ML_tools.py`.

The examples include:

1. Loading a processed MassLinker dataset.
2. Running n-fold validation with conventional machine-learning models.
3. Saving fold-level prediction results.
4. Summarizing validation metrics across folds.
5. Comparing MassLinker models with external model ROC results.
6. Training models on all samples and saving the trained model object.
7. Performing SHAP feature-importance analysis for RF, XGB, and LightGBM models.

The main machine-learning models used in this workflow are:

| Model | Implementation |
|---|---|
| SVM | `sklearn.svm.SVC` |
| XGBoost | `xgboost.XGBClassifier` |
| Random Forest | `sklearn.ensemble.RandomForestClassifier` |
| LightGBM | `lightgbm.LGBMClassifier` |

---

## Recommended project structure

A recommended project structure is:

```text
ml_analysis_project/
├── data/
│   └── processed_dataset.joblib
├── models/
│   └── saved_ml_model.joblib
├── results/
│   ├── fold_validation/
│   │   └── example_binary_classification/
│   │       ├── fold0.xlsx
│   │       ├── fold1.xlsx
│   │       ├── fold2.xlsx
│   │       ├── fold3.xlsx
│   │       └── fold4.xlsx
│   ├── figures/
│   │   ├── fold5_validation_summary.pdf
│   │   └── fold5_validation_with_external_models.pdf
│   └── shap/
│       ├── RF_feature_importance.pdf
│       ├── XGB_feature_importance.pdf
│       ├── LGB_feature_importance.pdf
│       └── feature_importances.joblib
├── external_results/
│   ├── external_model_1_roc.csv
│   ├── external_model_2_roc.csv
│   └── transformer_result.csv
└── scripts/
    └── ml_model_validation_examples.py
```

The role of each directory is summarized below.

| Directory | Description |
|---|---|
| `data/` | Stores processed MassLinker dataset files |
| `models/` | Stores trained machine-learning model objects |
| `results/fold_validation/` | Stores fold-level prediction result files |
| `results/figures/` | Stores validation summary and ROC comparison figures |
| `results/shap/` | Stores SHAP feature-importance plots and exported importance values |
| `external_results/` | Stores prediction results from external models |
| `scripts/` | Stores analysis scripts |

---

## Required input

The core input is one processed MassLinker dataset file:

```text
data/processed_dataset.joblib
```

This file should contain an `ExcelDataset`-like object or a merged MassLinker dataset object.

The dataset should be compatible with the following functions:

```python
from data import load_data, data_transform, gen_feature_names
```

In this workflow:

| Function | Purpose |
|---|---|
| `load_data()` | Load the dataset and generate fold split indices |
| `data_transform()` | Convert MassLinker token tensors into conventional ML feature matrices |
| `gen_feature_names()` | Generate feature names for SHAP interpretation |

---

## Required output from `load_data()`

The examples assume that:

```python
loaded_data, split_idx = load_data(data_path, n_part=5)
```

returns:

| Object | Description |
|---|---|
| `loaded_data` | Dataset-like object that supports indexed access |
| `split_idx` | Fold split indices used for cross-validation |

The indexed `loaded_data` object should provide data fields compatible with `ML_tools.ML_models`.

In the current implementation, the dataset indexing structure is used as follows:

| Index | Content |
|---:|---|
| `0` | MassLinker token tensors |
| `3` | Binary positive/negative labels |
| `4` | Generated class IDs or multi-class labels |

When `fit_mode="single"` is used, the binary label at index `3` is used as the classification target.

---

## External ROC result format

External model results can be added to ROC comparison plots using:

```python
ML_tools.addition_roc(...)
```

Each external ROC CSV file should contain two columns:

| Column | Description |
|---|---|
| First column | True binary label |
| Second column | Predicted probability for the positive class |

Example file:

```text
external_results/external_model_1_roc.csv
```

Example content:

```text
label,score
0,0.123
1,0.876
0,0.235
1,0.912
```

These external results can represent other models, such as image-based models, transformer models, or previously trained classifiers.

---

## Complete example script

Save the following script as:

```text
scripts/ml_model_validation_examples.py
```

```python
import os
import joblib

from tqdm import tqdm

from data import (
    load_data,
    data_transform,
    gen_feature_names
)

import ML_tools

from utils import visualization_foldn_valid


def fold_n_valid(
    data_path,
    task_name,
    save_dir,
    n_part=5,
    fit_mode="single"
):
    """
    Run n-fold validation using conventional machine-learning models.

    Parameters
    ----------
    data_path : str
        Path to the processed MassLinker dataset joblib file.
    task_name : str
        Name of the validation task. This name is used as the output subdirectory.
    save_dir : str
        Directory for saving fold-level prediction result files.
    n_part : int
        Number of folds.
    fit_mode : str
        Training target mode. Use "single" for binary classification.
    """

    loaded_data, split_idx = load_data(
        data_path,
        n_part=n_part
    )

    for fold in tqdm(range(n_part)):
        test_idx, train_idx = ML_tools.fold_n(
            split_idx,
            fold
        )

        models = ML_tools.ML_models(
            loaded_data[train_idx],
            loaded_data[train_idx],
            loaded_data[test_idx],
            loaded_data[test_idx]
        )

        models.fit_models(
            fit_mode=fit_mode
        )

        models.prediction()
        models.validation()

        ML_tools.save_model_results(
            models,
            fold,
            task_name,
            save_dir=save_dir
        )


def shap_ana(
    data_path,
    fit_mode="single",
    models=None,
    max_display=30,
    save_dir="results/shap",
    n_part=9
):
    """
    Perform SHAP feature-importance analysis for RF, XGB, and LightGBM models.

    Parameters
    ----------
    data_path : str
        Path to the processed MassLinker dataset joblib file.
    fit_mode : str
        Training target mode. Use "single" for binary classification.
    models : ML_tools.ML_models or None
        Pretrained ML_models object. If None, models will be trained on all samples.
    max_display : int
        Maximum number of features displayed in SHAP importance plots.
    save_dir : str
        Directory for saving SHAP output figures.
    n_part : int
        Number of parts used by load_data. This only affects data loading and splitting.

    Returns
    -------
    feature_importances : list
        Feature importance arrays from RF, XGB, and LightGBM.
    models : ML_tools.ML_models
        Trained or provided model wrapper.
    """

    os.makedirs(save_dir, exist_ok=True)

    feature_importances = []

    loaded_data, split_idx = load_data(
        data_path,
        n_part=n_part
    )

    all_indices = [
        i
        for i in range(len(loaded_data))
    ]

    all_data = loaded_data[all_indices]

    if models is None:
        models = ML_tools.ML_models(
            all_data,
            all_data,
            None,
            None
        )

        models.fit_models(
            fit_mode=fit_mode
        )

    X_all = data_transform(
        all_data[0]
    ).numpy()

    feature_names = gen_feature_names()

    analyzers = [
        ML_tools.UniversalSHAPAnalyzer(
            model=models.RF,
            X_train=X_all,
            X_test=X_all,
            max_display=max_display,
            feature_names=feature_names
        ),
        ML_tools.UniversalSHAPAnalyzer(
            model=models.XGB,
            X_train=X_all,
            X_test=X_all,
            max_display=max_display,
            feature_names=feature_names
        ),
        ML_tools.UniversalSHAPAnalyzer(
            model=models.lgb,
            X_train=X_all,
            X_test=X_all,
            max_display=max_display,
            feature_names=feature_names
        )
    ]

    model_names = [
        "RF",
        "XGB",
        "LGB"
    ]

    for analyzer, model_name in zip(analyzers, model_names):
        analyzer.plot_importance(
            model_name,
            save_dir=save_dir
        )

        feature_importances.append(
            analyzer.get_feature_importance()
        )

    return feature_importances, models


if __name__ == "__main__":

    # ------------------------------------------------------------
    # Define project paths
    # ------------------------------------------------------------

    data_path = "data/processed_dataset.joblib"

    model_dir = "models"
    fold_result_dir = "results/fold_validation"
    figure_dir = "results/figures"
    shap_dir = "results/shap"

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(fold_result_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)
    os.makedirs(shap_dir, exist_ok=True)

    task_name = "example_binary_classification"

    # ------------------------------------------------------------
    # Example 1: Run 5-fold validation
    # ------------------------------------------------------------

    fold_n_valid(
        data_path=data_path,
        task_name=task_name,
        save_dir=fold_result_dir,
        n_part=5,
        fit_mode="single"
    )

    # ------------------------------------------------------------
    # Example 2: Visualize fold-validation results
    # ------------------------------------------------------------

    fold_result_path = os.path.join(
        fold_result_dir,
        task_name
    )

    visualization_foldn_valid(
        fold_result_path,
        figure_dir,
        ["SVM", "XGB", "RF", "LGB"],
        "fold5_validation_summary.pdf",
        title="Model validation summary"
    )

    # ------------------------------------------------------------
    # Example 3: Compare with external ROC results
    # ------------------------------------------------------------

    external_rocs = ML_tools.addition_roc(
        paths=[
            "external_results/external_model_1_roc.csv",
            "external_results/external_model_2_roc.csv",
            "external_results/transformer_result.csv"
        ],
        model_names=[
            "External_Model_1",
            "External_Model_2",
            "Transformer"
        ]
    )

    visualization_foldn_valid(
        fold_result_path,
        figure_dir,
        ["SVM", "XGB", "RF", "LGB"],
        "fold5_validation_with_external_models.pdf",
        addition_ROCs=external_rocs,
        title="Model validation with external ROC comparison"
    )

    # ------------------------------------------------------------
    # Example 4: Train models on all samples and save model object
    # ------------------------------------------------------------

    loaded_data, split_idx = load_data(
        data_path,
        n_part=5
    )

    all_indices = [
        i
        for i in range(len(loaded_data))
    ]

    all_data = loaded_data[all_indices]

    models = ML_tools.ML_models(
        all_data,
        all_data,
        None,
        None
    )

    models.fit_models(
        fit_mode="single"
    )

    model_path = os.path.join(
        model_dir,
        "saved_ml_model.joblib"
    )

    joblib.dump(
        models,
        model_path
    )

    # ------------------------------------------------------------
    # Example 5: Load saved model and plot ROC curve
    # ------------------------------------------------------------

    loaded_models = joblib.load(
        model_path
    )

    loaded_models.plot_combined_roc(
        save_path=os.path.join(
            figure_dir,
            "combined_roc_from_saved_model.pdf"
        ),
        addition_ROCs=external_rocs,
        titles="ROC curves comparison"
    )

    # ------------------------------------------------------------
    # Example 6: SHAP feature-importance analysis
    # ------------------------------------------------------------

    feature_importances, models = shap_ana(
        data_path=data_path,
        fit_mode="single",
        models=models,
        max_display=20,
        save_dir=shap_dir,
        n_part=9
    )

    joblib.dump(
        feature_importances,
        os.path.join(
            shap_dir,
            "feature_importances.joblib"
        )
    )
```

---

## Example 1: Run fold-based validation

The function:

```python
fold_n_valid(
    data_path=data_path,
    task_name=task_name,
    save_dir=fold_result_dir,
    n_part=5,
    fit_mode="single"
)
```

runs n-fold machine-learning validation.

The workflow is:

```text
load_data()
    ↓
ML_tools.fold_n()
    ↓
ML_tools.ML_models()
    ↓
fit_models()
    ↓
prediction()
    ↓
validation()
    ↓
save_model_results()
```

For each fold, four models are trained:

1. SVM;
2. XGBoost;
3. Random Forest;
4. LightGBM.

The fold-level prediction results are saved under:

```text
results/fold_validation/example_binary_classification/
```

Example output:

```text
results/fold_validation/example_binary_classification/
├── fold0.xlsx
├── fold1.xlsx
├── fold2.xlsx
├── fold3.xlsx
└── fold4.xlsx
```

Each fold file stores predictions from different models and the corresponding true labels.

---

## Example 2: Visualize fold-validation results

After fold-level prediction files have been generated, use:

```python
visualization_foldn_valid(
    fold_result_path,
    figure_dir,
    ["SVM", "XGB", "RF", "LGB"],
    "fold5_validation_summary.pdf",
    title="Model validation summary"
)
```

This function reads fold-level result files, calculates model metrics, and generates a validation summary figure.

The output figure is saved as:

```text
results/figures/fold5_validation_summary.pdf
```

The summarized models are:

| Model label | Meaning |
|---|---|
| `SVM` | Support Vector Machine |
| `XGB` | XGBoost |
| `RF` | Random Forest |
| `LGB` | LightGBM |

---

## Example 3: Compare with external ROC results

External model results can be loaded using:

```python
external_rocs = ML_tools.addition_roc(
    paths=[
        "external_results/external_model_1_roc.csv",
        "external_results/external_model_2_roc.csv",
        "external_results/transformer_result.csv"
    ],
    model_names=[
        "External_Model_1",
        "External_Model_2",
        "Transformer"
    ]
)
```

Then pass `external_rocs` to:

```python
visualization_foldn_valid(
    fold_result_path,
    figure_dir,
    ["SVM", "XGB", "RF", "LGB"],
    "fold5_validation_with_external_models.pdf",
    addition_ROCs=external_rocs,
    title="Model validation with external ROC comparison"
)
```

This produces a combined visualization that includes:

- MassLinker conventional ML models;
- external model ROC curves;
- validation metrics from fold-level results.

The output file is:

```text
results/figures/fold5_validation_with_external_models.pdf
```

This is useful when comparing MassLinker token-based conventional machine-learning models with other model families, such as image-based models or transformer-based models.

---

## Example 4: Train models on all samples and save the model object

To train models using all available samples:

```python
loaded_data, split_idx = load_data(
    data_path,
    n_part=5
)

all_indices = [
    i
    for i in range(len(loaded_data))
]

all_data = loaded_data[all_indices]

models = ML_tools.ML_models(
    all_data,
    all_data,
    None,
    None
)

models.fit_models(
    fit_mode="single"
)
```

Then save the trained model wrapper:

```python
joblib.dump(
    models,
    "models/saved_ml_model.joblib"
)
```

The saved model object contains the trained classifiers:

| Attribute | Model |
|---|---|
| `models.SVM` | SVM classifier |
| `models.XGB` | XGBoost classifier |
| `models.RF` | Random Forest classifier |
| `models.lgb` | LightGBM classifier |

This is useful for reusing trained models without retraining.

---

## Example 5: Load a saved model and plot ROC curves

A saved model wrapper can be loaded using:

```python
loaded_models = joblib.load(
    "models/saved_ml_model.joblib"
)
```

Then generate a combined ROC plot:

```python
loaded_models.plot_combined_roc(
    save_path="results/figures/combined_roc_from_saved_model.pdf",
    addition_ROCs=external_rocs,
    titles="ROC curves comparison"
)
```

The output file is:

```text
results/figures/combined_roc_from_saved_model.pdf
```

Note that `plot_combined_roc()` uses the test data stored inside the `ML_models` object. Therefore, this method is most appropriate when the saved model object was created with a valid test set.

If the model was trained using all samples with `x_test=None` and `y_test=None`, this ROC plotting method cannot evaluate a held-out test set unless test data are provided in the model object.

---

## Example 6: SHAP feature-importance analysis

The function:

```python
feature_importances, models = shap_ana(
    data_path=data_path,
    fit_mode="single",
    models=models,
    max_display=20,
    save_dir=shap_dir,
    n_part=9
)
```

performs SHAP analysis for:

1. Random Forest;
2. XGBoost;
3. LightGBM.

It creates one `UniversalSHAPAnalyzer` for each model.

The output files are saved under:

```text
results/shap/
```

Example outputs:

```text
results/shap/
├── RF_feature_importance.pdf
├── XGB_feature_importance.pdf
├── LGB_feature_importance.pdf
└── feature_importances.joblib
```

The feature-importance values can be saved with:

```python
joblib.dump(
    feature_importances,
    "results/shap/feature_importances.joblib"
)
```

---

## Explanation of `fold_n_valid()`

The function:

```python
def fold_n_valid(data_path, task_name, save_dir, n_part=5, fit_mode="single"):
    ...
```

is the main validation loop.

It first loads the dataset:

```python
loaded_data, split_idx = load_data(
    data_path,
    n_part=n_part
)
```

Then for each fold:

```python
test_idx, train_idx = ML_tools.fold_n(
    split_idx,
    fold
)
```

The selected training and testing data are passed into `ML_models`:

```python
models = ML_tools.ML_models(
    loaded_data[train_idx],
    loaded_data[train_idx],
    loaded_data[test_idx],
    loaded_data[test_idx]
)
```

The same indexed dataset object is passed as both the feature source and label source because `ML_models` extracts features and labels internally by fixed indices.

In the current implementation:

| Internal access | Meaning |
|---|---|
| `x_train[0]` | Training feature tensor |
| `y_train[3]` | Binary label when `fit_mode="single"` |
| `y_train[4]` | Multi-class or generated label when `fit_mode` is not `"single"` |
| `x_test[0]` | Testing feature tensor |
| `y_test[3]` | Binary test label |
| `y_test[4]` | Multi-class or generated test label |

---

## Explanation of `fit_mode`

The argument:

```python
fit_mode="single"
```

controls which label field is used for model training.

| `fit_mode` | Label index | Use case |
|---|---:|---|
| `"single"` | `3` | Binary classification using positive/negative labels |
| Other values | `4` | Multi-class classification using generated class IDs |

For binary disease/control or case/control classification, use:

```python
fit_mode="single"
```

---

## Feature transformation for conventional ML models

MassLinker token tensors are not directly used by conventional machine-learning models.

Before training, the tensors are transformed by:

```python
data_transform(...)
```

Inside `ML_models.fit_models()`, the model input is prepared as:

```python
data_transform(self.x_train[0])
```

This converts MassLinker token tensors into a two-dimensional feature matrix:

```text
n_samples × n_features
```

This transformed matrix is then used by:

- SVM;
- XGBoost;
- Random Forest;
- LightGBM.

---

## Explanation of `shap_ana()`

The function:

```python
shap_ana(...)
```

provides a compact workflow for model interpretation.

It performs the following steps:

1. Load the full dataset.
2. Select all samples.
3. Train models on all samples if no pretrained model wrapper is provided.
4. Transform MassLinker token tensors using `data_transform()`.
5. Generate feature names using `gen_feature_names()`.
6. Create SHAP analyzers for RF, XGB, and LightGBM.
7. Plot SHAP feature-importance figures.
8. Return feature-importance arrays.

The analyzed models are:

| Model name | Model object |
|---|---|
| `RF` | `models.RF` |
| `XGB` | `models.XGB` |
| `LGB` | `models.lgb` |

SVM is not included in this SHAP workflow by default. Tree-based models are usually more efficient for SHAP analysis with `TreeExplainer`.

---

## Output files

This workflow may generate the following files:

| Output | Description |
|---|---|
| `results/fold_validation/example_binary_classification/fold0.xlsx` | Fold 0 prediction results |
| `results/fold_validation/example_binary_classification/fold1.xlsx` | Fold 1 prediction results |
| `results/fold_validation/example_binary_classification/fold2.xlsx` | Fold 2 prediction results |
| `results/fold_validation/example_binary_classification/fold3.xlsx` | Fold 3 prediction results |
| `results/fold_validation/example_binary_classification/fold4.xlsx` | Fold 4 prediction results |
| `results/figures/fold5_validation_summary.pdf` | Summary figure for fold validation |
| `results/figures/fold5_validation_with_external_models.pdf` | Validation summary with external ROC comparison |
| `results/figures/combined_roc_from_saved_model.pdf` | ROC figure generated from a saved model object |
| `models/saved_ml_model.joblib` | Saved machine-learning model wrapper |
| `results/shap/RF_feature_importance.pdf` | SHAP feature-importance plot for Random Forest |
| `results/shap/XGB_feature_importance.pdf` | SHAP feature-importance plot for XGBoost |
| `results/shap/LGB_feature_importance.pdf` | SHAP feature-importance plot for LightGBM |
| `results/shap/feature_importances.joblib` | Saved SHAP feature-importance arrays |

---

## Notes

1. This tutorial assumes that the input dataset has already been generated as a MassLinker `ExcelDataset`-like object.
2. The core input file is `data/processed_dataset.joblib`.
3. `fit_mode="single"` uses binary labels from index `3` of the loaded dataset object.
4. Other fitting modes use labels from index `4`, which are typically generated class IDs or multi-class labels.
5. `data_transform()` is required to convert MassLinker token tensors into conventional machine-learning feature matrices.
6. Fold-level prediction files are saved with `ML_tools.save_model_results()`.
7. `visualization_foldn_valid()` reads saved fold-level results and generates summary validation figures.
8. External ROC files should contain true labels in the first column and predicted positive-class probabilities in the second column.
9. `ML_tools.addition_roc()` can be used to compare MassLinker models with external models.
10. SHAP analysis in this tutorial focuses on RF, XGB, and LightGBM.
11. `plot_combined_roc()` requires a valid test set stored inside the `ML_models` object.
12. If a model is trained on all samples with no test set, use fold-level validation results for performance evaluation instead of treating training data as an independent test set.
13. This workflow is intended for supervised model validation and interpretation, not for MassLinker token generation.
