import joblib
from data import ExcelDataset, load_data, data_transform, gen_feature_names
import ML_tools
import os
import pandas as pd
import copy
import torch
from tqdm import tqdm
from utils import visualization_foldn_valid

def fold_n_valid(data_path, task_name, save_dir, n_part=5,fit_mode='single'):
    loaded_data, split_idx = load_data(data_path, n_part=n_part)
    for fold in tqdm(range(n_part)):
        test_idx, train_idx = ML_tools.fold_n(split_idx, fold)
        models = ML_tools.ML_models(loaded_data[train_idx], loaded_data[train_idx], loaded_data[test_idx],
                                    loaded_data[test_idx])
        models.fit_models(fit_mode=fit_mode)
        models.prediction()
        models.validation()
        ML_tools.save_model_results(models, fold, task_name, save_dir=save_dir)
def shap_ana(data_path, fit_mode='single', models=None, max_display=30):
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
        analyzers[analyzer].plot_importance(["RF", "XGB", "LGB"][analyzer])
        ret.append(analyzers[analyzer].get_feature_importance())
    return ret, models

