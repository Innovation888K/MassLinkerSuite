"""
High-level analysis workflows for MassLinker.

This module provides user-facing analysis pipelines that connect MassLinker
tokenized datasets with downstream statistical analysis, visualization,
traditional machine-learning validation, and SHAP-based interpretability.

The main workflow `MassLinker_Anal` performs distance-based differential
analysis using reconstructed MassLinker token signals. It calculates
Jensen-Shannon and Wasserstein distances, identifies differential metabolic
tokens, generates p-value ranking plots, reconstructs top differential peaks,
performs KEGG enrichment analysis, and visualizes sample-level distance
patterns.

Additional helper workflows are provided for N-fold validation of conventional
machine-learning models and SHAP-based feature attribution analysis.
"""

import os.path
import joblib
from transformer_shap import plot_top_n_feature_shap_2d
from ML_tools import addition_roc
from data import ExcelDataset, load_data, data_transform, gen_feature_names
import ML_tools
import os
import pandas as pd
import copy
import torch
from tqdm import tqdm
from utils import visualization_foldn_valid
import pathway_mask
from utils import metabo_dis, get_diff, kegg_enrichment, plot_enrichment, plot_peak_comp, plot_2d, plot_p_value
import joblib
import numpy as np
from transformer import transformer_language
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy import stats
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from matplotlib.colors import LinearSegmentedColormap


def MassLinker_Anal(dataset_path, sele_sample=None, load_JS=None, load_was=None, use_JS=False, plot_peak_comp_dir=None,
                    plot_peak_topn=100, path_way_detail_file="pathway_compound_detail.csv", work_dir='./', save=True,
                    p_limit=0.05):
    """
    Run MassLinker distance-based differential analysis.

    This high-level workflow loads a processed MassLinker dataset, reconstructs
    chromatographic signals from RBF-derived tokens, calculates distance
    profiles relative to a reference sample, performs two-group statistical
    comparison, and generates downstream visualizations and KEGG enrichment
    results.

    By default, Wasserstein distance is used for differential analysis because
    it captures shifts in chromatographic signal distribution along the
    retention-time axis. Jensen-Shannon distance can also be used by setting
    `use_JS=True`.

    Parameters
    ----------
    dataset_path : str
        Path to a joblib-serialized MassLinker `ExcelDataset` object.
    sele_sample : list, optional
        Selected sample indices used for analysis. If None, all samples are used.
    load_JS : str, optional
        Path to precomputed Jensen-Shannon distance results.
    load_was : str, optional
        Path to precomputed Wasserstein distance results.
    use_JS : bool, optional
        Whether to use Jensen-Shannon distance for differential testing.
        If False, Wasserstein distance is used.
    plot_peak_comp_dir : str, optional
        Directory used to save reconstructed peak comparison plots.
        If None, peak comparison plots are skipped.
    plot_peak_topn : int, optional
        Number of top differential features used for peak comparison plots.
    path_way_detail_file : str, optional
        Path to the compound/pathway annotation table.
    work_dir : str, optional
        Output directory for intermediate results and figures.
    save : bool, optional
        Whether to save distance results, p-values, and enrichment results.
    p_limit : float, optional
        P-value threshold used to select features for 2D distance visualization.

    Returns
    -------
    tuple
        Jensen-Shannon distance results and Wasserstein distance results.
    """
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    # Load the processed MassLinker dataset.
    dataset = joblib.load(dataset_path)

    # Use all samples if no selected sample indices are provided.
    if sele_sample is None:
        sele_sample = [i for i in range(len(dataset))]

    # Extract binary group labels and MassLinker token tensors.
    group = [dataset.is_positive[i].numpy().item() for i in sele_sample]
    params = [dataset.samples[i] for i in sele_sample]

    # Load precomputed distance matrices if provided; otherwise calculate them.
    if load_JS is not None and load_was is not None:
        JS = joblib.load(load_JS)
        was = joblib.load(load_was)
    else:
        JS, was = metabo_dis(params)

    if save:
        joblib.dump(JS, os.path.join(work_dir, "JS.joblib"))
        joblib.dump(was, os.path.join(work_dir, "was.joblib"))

    # Convert distance results into sample-by-feature matrices.
    was_arr = np.array(was)[0]
    JS_arr = np.array(JS)[0]

    # Perform two-group differential analysis using JS or Wasserstein distance.
    if use_JS:
        p_values, diff = get_diff(JS_arr, group)
    else:
        p_values, diff = get_diff(was_arr, group)

    if save:
        joblib.dump(p_values, os.path.join(work_dir, "p_values.joblib"))

    # Load compound names and m/z values for feature annotation.
    met_name = pd.read_csv(path_way_detail_file)["compound_names"].tolist()
    mzs = pd.read_csv(path_way_detail_file)["mz"].tolist()

    # Plot top-ranked differential features by p-value.
    plot_p_value(p_values, met_name, mzs=mzs, n=20, save_path=os.path.join(work_dir, "p_value.pdf"))

    # Optionally reconstruct and plot chromatographic curves for top features.
    if plot_peak_comp_dir is not None:
        plot_peak_comp(p_values, group, params, met_name, mzs, save_path=plot_peak_comp_dir, top_n=plot_peak_topn)

    # Perform KEGG pathway enrichment analysis based on differential tokens.
    pathway_df_full = pd.read_csv(path_way_detail_file, index_col=0)
    kegg_enriched = kegg_enrichment(pathway_df_full, p_values, diff)

    if save:
        joblib.dump(kegg_enriched, os.path.join(work_dir, "kegg_enriched.joblib"))

    # Plot KEGG enrichment results.
    plot_df = kegg_enriched.copy()
    plot_enrichment(plot_df, save_name=os.path.join(work_dir, "KEGG_Enriched1.pdf"))

    # Select significant distance features for low-dimensional visualization.
    was1 = [[] for i in range(len(was[0]))]
    JS1 = [[] for i in range(len(was[0]))]

    for i in range(len(p_values)):
        if p_values[i] <= p_limit:
            for j in range(len(was[0])):
                was1[j].append(was[0][j][i])
                JS1[j].append(JS[0][j][i])

    # Visualize sample separation using selected JS and Wasserstein features.
    plot_2d([JS1], [was1], group, save_path=work_dir)

    return JS, was


def fold_n_valid(data_path, task_name, save_dir, n_part=5, fit_mode='single', compare_model_path=None,
                 compare_models=None):
    """
    Run N-fold validation for conventional MassLinker machine-learning models.

    This workflow loads a serialized MassLinker dataset, generates random
    partitions, trains SVM, XGBoost, random forest, and LightGBM models on each
    fold, saves fold-level predictions, and generates a summary performance
    plot.

    Parameters
    ----------
    data_path : str
        Path to a joblib-serialized MassLinker dataset.
    task_name : str
        Name used to organize output files and figure titles.
    save_dir : str
        Directory used to save model results and validation plots.
    n_part : int, optional
        Number of folds or partitions.
    fit_mode : str, optional
        Modeling mode. `single` uses binary group labels, while other modes
        use generated class labels.
    compare_model_path : list, optional
        Paths to external prediction-score files for additional ROC comparison.
    compare_models : list, optional
        Names of external models.
    """
    # Load the MassLinker dataset and generate fold indices.
    loaded_data, split_idx = load_data(data_path, n_part=n_part)

    addition_roc = None

    # Optionally load external ROC curves for comparison.
    if compare_model_path:
        addition_roc = ML_tools.addition_roc(paths=compare_model_path, model_names=compare_models)

    for fold in tqdm(range(n_part)):
        test_idx, train_idx = ML_tools.fold_n(split_idx, fold)

        # Build model wrapper using fold-specific training and test partitions.
        models = ML_tools.ML_models(loaded_data[train_idx], loaded_data[train_idx], loaded_data[test_idx],
                                    loaded_data[test_idx])

        models.fit_models(fit_mode=fit_mode)
        models.prediction()
        models.validation(pos_label=2)

        # Save fold-level predictions and the trained model wrapper.
        ML_tools.save_model_results(models, fold, task_name, save_dir=save_dir)
        joblib.dump(models, os.path.join(save_dir, f"models_fold{fold}.joblib"))

        # if fit_mode == 'single':
        #     models.plot_combined_roc(save_path=os.path.join(save_dir, task_name + f"fold{fold}_ROC.pdf"),
        #                              addition_ROCs=addition_roc, titles=task_name)

    # Summarize N-fold validation metrics across models.
    visualization_foldn_valid(os.path.join(save_dir, task_name), save_dir, ['SVM', 'XGB', 'RF', 'LGB'],
                              task_name + 'fold5_valid.pdf',
                              addition_ROCs=addition_roc, title=task_name)


def shap_ana(data_path, fit_mode='single', models=None, max_display=30, save_dir='./'):
    """
    Run SHAP analysis for conventional MassLinker machine-learning models.

    This workflow trains or reuses conventional machine-learning models and
    applies the universal SHAP analyzer to random forest, XGBoost, and LightGBM.
    It saves feature-importance plots and returns class-wise SHAP importance
    summaries.

    Parameters
    ----------
    data_path : str
        Path to a joblib-serialized MassLinker dataset.
    fit_mode : str, optional
        Modeling mode. `single` uses binary group labels, while other modes
        use generated class labels.
    models : ML_tools.ML_models, optional
        Pretrained model wrapper. If None, models are trained on the full dataset.
    max_display : int, optional
        Number of top features displayed in SHAP importance plots.
    save_dir : str, optional
        Directory used to save SHAP feature-importance figures.

    Returns
    -------
    tuple
        ret and models. `ret` contains SHAP feature-importance arrays, and
        `models` is the trained or provided model wrapper.
    """
    ret = []

    # Load the MassLinker dataset. The split index is generated for compatibility.
    loaded_data, split_idx = load_data(data_path, n_part=9)

    # Train conventional models on the full dataset if no model wrapper is provided.
    if models is None:
        models = ML_tools.ML_models(loaded_data[[i for i in range(len(loaded_data))]],
                                    loaded_data[[i for i in range(len(loaded_data))]], None, None)
        models.fit_models(fit_mode=fit_mode)

    # Build SHAP analyzers for random forest, XGBoost, and LightGBM.
    analyzers = [
        ML_tools.UniversalSHAPAnalyzer(
            model=models.RF,
            X_train=data_transform(loaded_data[[i for i in range(len(loaded_data))]][0]).numpy(),
            X_test=data_transform(loaded_data[[i for i in range(len(loaded_data))]][0]).numpy(),
            max_display=max_display,
            feature_names=gen_feature_names()
        ),
        ML_tools.UniversalSHAPAnalyzer(
            model=models.XGB,
            X_train=data_transform(loaded_data[[i for i in range(len(loaded_data))]][0]).numpy(),
            X_test=data_transform(loaded_data[[i for i in range(len(loaded_data))]][0]).numpy(),
            max_display=max_display,
            feature_names=gen_feature_names()
        ),
        ML_tools.UniversalSHAPAnalyzer(
            model=models.lgb,
            X_train=data_transform(loaded_data[[i for i in range(len(loaded_data))]][0]).numpy(),
            X_test=data_transform(loaded_data[[i for i in range(len(loaded_data))]][0]).numpy(),
            max_display=max_display,
            feature_names=gen_feature_names()
        ),
    ]

    for analyzer in range(len(analyzers)):
        # if fit_mode == 'single':
        #     analyzer.plot_waterfall()
        analyzers[analyzer].plot_importance(["RF", "XGB", "LGB"][analyzer], save_dir=save_dir)
        ret.append(analyzers[analyzer].get_feature_importance())

    return ret, models
