"""
Machine-learning validation and SHAP analysis example for MassLinker.

This script demonstrates how to evaluate conventional machine-learning models
on MassLinker tokenized metabolomics datasets. It supports N-fold validation,
fold-level result visualization, external ROC comparison, full-dataset model
training, model serialization, and SHAP-based feature-importance analysis.

The conventional models are managed through `ML_tools.ML_models`, which wraps
SVM, XGBoost, random forest, and LightGBM classifiers. MassLinker token tensors
are flattened into feature vectors before being passed to these models.
"""

import os
import joblib
import torch

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
    Run n-fold validation using traditional machine-learning models.

    Parameters
    ----------
    data_path : str
        Path to the processed MassLinker dataset joblib file.
    task_name : str
        Name of the validation task. Used as the output subdirectory name.
    save_dir : str
        Directory for saving fold-level prediction results.
    n_part : int
        Number of folds.
    fit_mode : str
        Training target mode. Use "single" for binary classification.
    """

    # Load the serialized MassLinker dataset and generate random fold indices.
    loaded_data, split_idx = load_data(
        data_path,
        n_part=n_part
    )

    for fold in tqdm(range(n_part)):
        # Build train/test indices for the current fold.
        test_idx, train_idx = ML_tools.fold_n(
            split_idx,
            fold
        )

        # Initialize conventional machine-learning models for the current fold.
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

        # Save fold-level predictions and true labels.
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
    save_dir="results/shap"
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
        Directory for saving SHAP plots.

    Returns
    -------
    ret : list
        Feature importance arrays from RF, XGB, and LightGBM.
    models : ML_tools.ML_models
        Trained or provided model wrapper.
    """

    # Ensure that the SHAP output directory exists.
    os.makedirs(save_dir, exist_ok=True)

    ret = []

    # Load all samples from the processed MassLinker dataset.
    loaded_data, split_idx = load_data(
        data_path,
        n_part=9
    )

    all_indices = [
        i
        for i in range(len(loaded_data))
    ]

    all_data = loaded_data[all_indices]

    # Train models on the full dataset if no pretrained model wrapper is provided.
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

    # Flatten MassLinker token tensors for conventional machine-learning models.
    X_all = data_transform(
        all_data[0]
    ).numpy()

    # Generate feature names corresponding to flattened MassLinker token features.
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

    # Plot and collect SHAP feature-importance results for each model.
    for analyzer, model_name in zip(analyzers, model_names):
        analyzer.plot_importance(
            model_name,
            save_dir=save_dir
        )

        ret.append(
            analyzer.get_feature_importance()
        )

    return ret, models


if __name__ == "__main__":

    # Define project paths.
    data_path = "data/processed_dataset.joblib"

    fold_result_dir = "results/fold_validation"
    visualization_dir = "results/fold_validation"
    model_dir = "models"
    shap_dir = "results/shap"

    os.makedirs(fold_result_dir, exist_ok=True)
    os.makedirs(visualization_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
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
        visualization_dir,
        ["SVM", "XGB", "RF", "LGB"],
        "fold5_validation_summary.pdf",
        title="Model validation summary"
    )

    # ------------------------------------------------------------
    # Example 3: Compare with external ROC results
    # ------------------------------------------------------------
    external_roc_paths = [
        "external_results/external_model_1_roc.csv",
        "external_results/external_model_2_roc.csv"
    ]

    external_rocs = ML_tools.addition_roc(
        paths=external_roc_paths,
        model_names=[
            "External_Model_1",
            "External_Model_2"
        ]
    )

    visualization_foldn_valid(
        fold_result_path,
        visualization_dir,
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
    # Example 5: SHAP feature-importance analysis
    # ------------------------------------------------------------
    feature_importances, models = shap_ana(
        data_path=data_path,
        fit_mode="single",
        models=models,
        max_display=20,
        save_dir=shap_dir
    )

    joblib.dump(
        feature_importances,
        os.path.join(shap_dir, "feature_importances.joblib")
    )
