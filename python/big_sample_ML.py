import joblib
from data import ExcelDataset, load_data, data_transform, gen_feature_names
import ML_tools
import os
import pandas as pd
import copy
import torch
from tqdm import tqdm
from utils import visualization_foldn_valid
import pathway_mask


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


if __name__ == '__main__':
    # for dataset in ["2521"]:
    for dataset in ['1705', "3444", "1122", "2521", "6039",'3838']:
        # loaded_data, split_idx = load_data(
        #     os.path.join(r'F:\质谱不鉴定数据\源数据\合并后joblib数据', dataset + ".joblib"))
        # test_idx, train_idx = ML_tools.fold_n(split_idx, 4)
        # models = ML_tools.ML_models(loaded_data[train_idx], loaded_data[train_idx], loaded_data[test_idx],
        #                             loaded_data[test_idx])
        # models.fit_models()
        # models.prediction()
        # models.validation()
        # joblib.dump(models, os.path.join(r'F:\MassLinker_workspace\saved_MLmodels', dataset + "_single.joblib"))
        # models = joblib.load(os.path.join(r'F:\MassLinker_workspace\saved_MLmodels', '2521enhance_single.joblib'))
        # models = joblib.load(os.path.join(r'F:\MassLinker_workspace\saved_MLmodels', dataset + "_single.joblib"))
        # models.plot_combined_roc(save_path=os.path.join(r'F:\MassLinker_workspace\visual', dataset + "ROC.pdf"),
        #                          addition_ROCs=ML_tools.addition_roc(
        #                              paths=[
        #                                  os.path.join(r'F:\MetImage\save_model', dataset, r'roc_data_.csv'),
        #                                  os.path.join(r"D:\git\MassLinker\pseudoMS-image-predictor\trained_models",
        #                                               dataset + "_predicted_224x224_fold5.csv")],
        #                              model_names=["MetImage", 'Deep_pseudoMSI']),
        #                          titles="MTBLS" + dataset)
        # ML_tools.save_model_results(models, 0, '1122_origin',save_dir=)
        # fold_n_valid(
        #     os.path.join(r'F:\质谱不鉴定数据\源数据\合并后joblib数据', dataset + ".joblib"),
        #     dataset + '_origin',
        #     r'F:\MassLinker_workspace\save_fold5_results')
        # visualization_foldn_valid(
        #     os.path.join(r'F:\MassLinker_workspace\save_fold5_results', dataset + '_origin'),
        #     r'F:\MassLinker_workspace\visual',
        #     ['SVM', 'XGB', 'RF', 'LGB'],
        #     dataset + 'fold5_valid.pdf',
        #     addition_ROCs=ML_tools.addition_roc(
        #         paths=[
        #             os.path.join(r'F:\MetImage\save_model', dataset, r'roc_data_.csv'),
        #             os.path.join(r"D:\git\MassLinker\pseudoMS-image-predictor\trained_models",
        #                          dataset + "_predicted_224x224_fold5.csv")],
        #         model_names=["MetImage", 'Deep_pseudoMSI']),
        #     title=dataset + "Model_indexes"
        # )
        #
        # loaded_data, split_idx = load_data("1122.joblib", n_part=9)
        # models = ML_tools.ML_models(loaded_data[[i for i in range(len(loaded_data))]],
        #                             loaded_data[[i for i in range(len(loaded_data))]], None, None)
        # models.fit_models(fit_mode='single')
        # joblib.dump(models, "models_single.joblib")
        #models = joblib.load("models_single.joblib")
        feature_importances, models = shap_ana(f"F:\质谱不鉴定数据\源数据\合并后joblib数据/{dataset}.joblib", max_display=20)
        joblib.dump(feature_importances,f"F:\MassLinker_workspace\ML_shap/{dataset}sample_feature_improtances.joblib")
