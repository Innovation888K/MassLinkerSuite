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
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    dataset = joblib.load(dataset_path)
    if sele_sample is None:
        sele_sample = [i for i in range(len(dataset))]
    group = [dataset.is_positive[i].numpy().item() for i in sele_sample]
    params = [dataset.samples[i] for i in sele_sample]
    if load_JS is not None and load_was is not None:
        JS = joblib.load(load_JS)
        was = joblib.load(load_was)
    else:
        JS, was = metabo_dis(params)
    if save:
        joblib.dump(JS, os.path.join(work_dir, "JS.joblib"))
        joblib.dump(was, os.path.join(work_dir, "was.joblib"))
    was_arr = np.array(was)[0]
    JS_arr = np.array(JS)[0]
    if use_JS:
        p_values, diff = get_diff(JS_arr, group)
    else:
        p_values, diff = get_diff(was_arr, group)
    if save:
        joblib.dump(p_values, os.path.join(work_dir, "p_values.joblib"))
    met_name = pd.read_csv(path_way_detail_file)["compound_names"].tolist()
    mzs = pd.read_csv(path_way_detail_file)["mz"].tolist()
    plot_p_value(p_values, met_name, mzs=mzs, n=20, save_path=os.path.join(work_dir, "p_value.pdf"))
    if plot_peak_comp_dir is not None:
        plot_peak_comp(p_values, group, params, met_name, mzs, save_path=plot_peak_comp_dir, top_n=plot_peak_topn)
    pathway_df_full = pd.read_csv(path_way_detail_file, index_col=0)
    kegg_enriched = kegg_enrichment(pathway_df_full, p_values, diff)
    if save:
        joblib.dump(kegg_enriched, os.path.join(work_dir, "kegg_enriched.joblib"))
    plot_df = kegg_enriched.copy()
    plot_enrichment(plot_df, save_name=os.path.join(work_dir, "KEGG_Enriched1.pdf"))
    was1 = [[] for i in range(len(was[0]))]
    JS1 = [[] for i in range(len(was[0]))]
    for i in range(len(p_values)):
        if p_values[i] <= p_limit:
            for j in range(len(was[0])):
                was1[j].append(was[0][j][i])
                JS1[j].append(JS[0][j][i])
    plot_2d([JS1], [was1], group, save_path=work_dir)
    return JS, was


def fold_n_valid(data_path, task_name, save_dir, n_part=5, fit_mode='single', compare_model_path=None,
                 compare_models=None):
    loaded_data, split_idx = load_data(data_path, n_part=n_part)
    addition_roc = None
    if compare_model_path:
        addition_roc = ML_tools.addition_roc(paths=compare_model_path, model_names=compare_models)
    for fold in tqdm(range(n_part)):
        test_idx, train_idx = ML_tools.fold_n(split_idx, fold)
        models = ML_tools.ML_models(loaded_data[train_idx], loaded_data[train_idx], loaded_data[test_idx],
                                    loaded_data[test_idx])
        models.fit_models(fit_mode=fit_mode)
        models.prediction()
        models.validation(pos_label=2)
        ML_tools.save_model_results(models, fold, task_name, save_dir=save_dir)
        joblib.dump(models, os.path.join(save_dir, f"models_fold{fold}.joblib"))
        # if fit_mode == 'single':
        #     models.plot_combined_roc(save_path=os.path.join(save_dir, task_name + f"fold{fold}_ROC.pdf"),
        #                              addition_ROCs=addition_roc, titles=task_name)
    visualization_foldn_valid(os.path.join(save_dir, task_name), save_dir, ['SVM', 'XGB', 'RF', 'LGB'],
                              task_name + 'fold5_valid.pdf',
                              addition_ROCs=addition_roc, title=task_name)


def shap_ana(data_path, fit_mode='single', models=None, max_display=30, save_dir='./'):
    ret = []
    loaded_data, split_idx = load_data(data_path, n_part=9)
    if models is None:
        models = ML_tools.ML_models(loaded_data[[i for i in range(len(loaded_data))]],
                                    loaded_data[[i for i in range(len(loaded_data))]], None, None)
        models.fit_models(fit_mode=fit_mode)
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
