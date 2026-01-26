import copy
import os
import random
from data import ExcelDataset
import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
import joblib
from data import data_transform
import shap
import numpy as np

plt.rcParams.update({
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
    'text.color': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'font.sans-serif': 'Arial'
})


def addition_roc(paths: list, model_names: list = None, colors: list = None):
    """
    读取机器学习训练结果文件并返回addition_rocs变量
    Parameters:
    -----------
    paths : list
        文件路径列表，每个文件包含两列：第一列是实际标签，第二列是预测概率
    model_names : list, optional
        模型名称列表，如果为None则使用默认名称
    colors : list, optional
        颜色列表，如果为None则使用默认颜色
    Returns:
    --------
    list
        返回格式为 [(y_trues, y_scores, color, model_name), ...]
    """
    if model_names is None:
        model_names = [f'Model_{i + 1}' for i in range(len(paths))]
    if colors is None:
        default_colors = ['orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        colors = default_colors[:len(paths)]
    addition_rocs = []
    for path, model_name, color in zip(paths, model_names, colors):
        try:
            data = pd.read_csv(path)
            y_true = data.iloc[:, 0].values
            y_ture_2d = np.column_stack([1 - y_true, y_true])
            y_score = data.iloc[:, 1].values
            y_score_2d = np.column_stack([1 - y_score, y_score])
            addition_rocs.append((y_ture_2d, y_score_2d, color, model_name))
        except Exception as e:
            print(f"处理文件 {path} 时出错: {e}")
            continue
    return addition_rocs


def save_model_results(models, fold, name, save_dir="./"):
    results = copy.deepcopy(models.predict_result)
    results.append(list())
    for i in copy.deepcopy(models.y_test[models.idx]):
        results[-1].append(i.item())
    temp = pd.DataFrame(results).T
    if not os.path.exists(os.path.join(save_dir, name)):
        os.makedirs(os.path.join(save_dir, name))
    temp.to_excel(os.path.join(save_dir, name, "fold" + str(fold) + ".xlsx"))


def fold_n(split_idx, n):
    test_idx = list()
    for i in range(len(split_idx)):
        if not i == n:
            for j in split_idx[i]:
                test_idx.append(j)
    return split_idx[n], test_idx


class ML_models:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.class_num = None
        self.y_test = None
        self.x_test = None
        self.y_train = None
        self.x_train = None
        self.RF = None
        self.XGB = None
        self.num_round = None
        self.param = None
        self.scaler = None
        self.SVM = None
        self.lgb = None
        self.predict_result = list()
        self.load_data(x_train, y_train, x_test, y_test)
        self.init_SVM()
        self.init_XGB()
        self.init_RF()
        self.init_lgb()

    def load_data(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.class_num = len(set(x_train[4]))

    def init_SVM(self, kernel='rbf', C=1.0, gamma='scale'):
        self.scaler = StandardScaler()
        self.SVM = SVC(kernel=kernel, C=C, gamma=gamma)

    def init_XGB(self, max_depth=50, eta=0.001, num_round=200, n_worker=20):
        self.param = {'max_depth': max_depth, 'eta': eta, 'objective': 'multi:softmax', 'num_class': self.class_num}
        self.num_round = num_round
        self.XGB = xgb.XGBClassifier(nthread=n_worker)

    def init_RF(self, n_estimators=1000, max_depth=None):
        self.RF = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

    def init_lgb(self):
        self.lgb = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=self.class_num,
            metric='multi_logloss',
            n_estimators=200,
            learning_rate=0.0001,
            num_leaves=50,
            max_depth=-1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )

    def fit_models(self, fit_mode="single"):
        if fit_mode == 'single':
            self.idx = 3
        else:
            self.idx = 4
        print("Fitting RF")
        self.RF.fit(data_transform(self.x_train[0]), self.y_train[self.idx])
        print("Fitting SVm")
        self.SVM.fit(data_transform(self.x_train[0]), self.y_train[self.idx])
        print("Fitting XGB")
        self.XGB.fit(data_transform(self.x_train[0]), self.y_train[self.idx])
        print("Fitting LGB")
        self.lgb.fit(data_transform(self.x_train[0]), self.y_train[self.idx])

    def prediction(self):
        self.predict_result.append(self.SVM.predict(data_transform(self.x_test[0])))
        self.predict_result.append(self.XGB.predict(data_transform(self.x_test[0])))
        self.predict_result.append(self.RF.predict(data_transform(self.x_test[0])))
        self.predict_result.append(self.lgb.predict(data_transform(self.x_test[0])))

    def validation(self,pos_label=1):
        for y_pred in self.predict_result:
            accuracy = accuracy_score(self.y_test[self.idx], y_pred)
            precision = precision_score(self.y_test[self.idx], y_pred, pos_label=pos_label, average='weighted')
            recall = recall_score(self.y_test[self.idx], y_pred, pos_label=pos_label, average='weighted')
            f1 = f1_score(self.y_test[self.idx], y_pred, pos_label=pos_label, average='weighted')
            conf_matrix = confusion_matrix(self.y_test[self.idx], y_pred)
            print("model:")
            print(f"acc={accuracy}")
            print(f"pre={precision}")
            print(f"recall={recall}")
            print(f"f1={f1}")
            print(f"conf_matrix:\n{conf_matrix}")

    def predict_proba_all(self):
        self.predict_proba_result = {
            'XGB': self.XGB.predict_proba(data_transform(self.x_test[0])),
            'RF': self.RF.predict_proba(data_transform(self.x_test[0])),
            'LGB': self.lgb.predict_proba(data_transform(self.x_test[0]))[:, :2]
        }

    def plot_combined_roc(self, figsize=(12, 8), save_path=None, addition_ROCs=None,
                          titles='ROC Curves Comparison (Micro-Average)'):
        self.predict_proba_all()
        y_true = self.y_test[self.idx]
        classes = np.unique(y_true)
        n_classes = len(classes)
        y_true_bin = label_binarize(y_true, classes=classes)
        if n_classes == 2:
            y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
        plt.figure(figsize=figsize)
        model_names = ['XGB', 'RF', 'LGB']
        colors = ['blue', 'red', 'green']
        for model_name, color in zip(model_names, colors):
            if model_name not in self.predict_proba_result:
                continue
            y_score = self.predict_proba_result[model_name]
            fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
            roc_auc_micro = auc(fpr_micro, tpr_micro)
            plt.plot(fpr_micro, tpr_micro, color=color, linewidth=2,
                     label=f'{model_name} (AUC = {roc_auc_micro:.3f})')
        if addition_ROCs is not None:
            for model_roc in addition_ROCs:
                y_trues, y_scores, color, model_name = model_roc
                fpr_micro, tpr_micro, _ = roc_curve(y_trues[:, 1], y_scores[:, 1])
                roc_auc_micro = auc(fpr_micro, tpr_micro)
                plt.plot(fpr_micro, tpr_micro, color=color, linewidth=2,
                         label=f'{model_name} (AUC = {roc_auc_micro:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        plt.xlim([-0.1, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(titles, fontsize=14)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path, format='pdf')


def init_indexes():
    model_names = ['SVM', 'XGB', 'RF', 'LightGBM']
    all_accuracy = {f'{i}': [] for i in model_names}
    all_precision = {f'{i}': [] for i in model_names}
    all_recall = {f'{i}': [] for i in model_names}
    all_f1 = {f'{i}': [] for i in model_names}
    roc_data = {f'{i}': {'fpr': [], 'tpr': [], 'auc': []} for i in model_names}
    return all_accuracy, all_precision, all_recall, all_f1, roc_data


def cal_all_index(dic):
    all_accuracy, all_precision, all_recall, all_f1, roc_data = init_indexes()
    for fold_name, df_fold_result in dic.items():
        y_true = df_fold_result[4]
        for idx in range(4):
            model_idx = ['SVM', 'XGB', 'RF', 'LightGBM'][idx]
            model_name = f'{model_idx}'
            y_pred = df_fold_result[idx]
            all_accuracy[model_name].append(accuracy_score(y_true, y_pred))
            all_precision[model_name].append(
                precision_score(y_true, y_pred, pos_label=1, average='weighted', zero_division=0))
            all_recall[model_name].append(
                recall_score(y_true, y_pred, pos_label=1, average='weighted', zero_division=0))
            all_f1[model_name].append(f1_score(y_true, y_pred, pos_label=1, average='weighted', zero_division=0))
            y_pred_proba_for_auc = df_fold_result[idx]
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba_for_auc, pos_label=1)
            fold_auc = auc(fpr, tpr)
            roc_data[model_name]['fpr'].append(fpr)
            roc_data[model_name]['tpr'].append(tpr)
            roc_data[model_name]['auc'].append(fold_auc)
    metrics_data = {
        'Accuracy': all_accuracy,
        'Precision': all_precision,
        'Recall': all_recall,
        'F1-Score': all_f1
    }
    if all(roc_data[model]['auc'] for model in
           roc_data):
        metrics_data['AUC'] = {model: roc_data[model]['auc'] for model in roc_data}
    return metrics_data, roc_data


def box_plot(metrics_data):
    for metric_name, data_dict in metrics_data.items():
        plot_df = pd.DataFrame({
            'Model': [model for model in data_dict.keys() for _ in data_dict[model]],
            'Value': [val for model in data_dict.keys() for val in data_dict[model]]
        })

        plt.figure(figsize=(8, 6))
        sns.boxplot(x='Model', y='Value', data=plot_df)
        plt.title(f'{metric_name} Across 5 Folds for Different Models')
        plt.ylabel(metric_name)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()


def prepare_barplot_data(metrics_data):
    plot_data_list = []
    for metric_name, data_dict in metrics_data.items():
        for model_name, values in data_dict.items():
            for val in values:
                plot_data_list.append({
                    'Model': str(model_name),
                    'Metric': str(metric_name),
                    'Value': val
                })
    return pd.DataFrame(plot_data_list)


def bar_plot(metrics_df, id):
    if not os.path.exists('./barplots/' + id):
        os.makedirs('./barplots/' + id)
    for metric_name in metrics_df['Metric'].unique():
        plt.figure(figsize=(4, 6))
        subset_df = metrics_df[metrics_df['Metric'] == metric_name]
        ax = sns.barplot(
            x='Model',
            y='Value',
            data=subset_df,
            errorbar='sd',
            capsize=0.1,
            errwidth=2,
            palette='viridis'
        )
        for p in ax.patches:  # 遍历每个柱子
            height = p.get_height()
            text_value = f'{height:.2f}'
            ax.text(p.get_x() + p.get_width() / 2.,
                    height,
                    text_value,
                    ha='center',  # 水平居中
                    va='bottom',  # 垂直对齐方式，使其位于柱子上方
                    fontsize=9,  # 字体大小
                    color='black')
        plt.title(metric_name)
        plt.ylabel(f'average {metric_name}')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.ylim(bottom=0)
        plt.tight_layout()
        plt.savefig(fname="./barplots/" + id + "/" + subset_df.iloc[0, 1] + "_enhance.pdf", format="pdf")


def visualization(id):
    path = "./" + id
    dic = {}
    for file in os.listdir(path):
        dic[file] = pd.read_excel(os.path.join(path, file))
    metrics_data, roc_data = cal_all_index(dic)
    metrics_df = prepare_barplot_data(metrics_data)
    bar_plot(metrics_df, id)


def cross_dataset_data_prepare(loaded_data, part):
    y_test = loaded_data[part][3]
    x_test = loaded_data[part][0]
    y_train = torch.cat([loaded_data[i][3] for i in range(len(loaded_data)) if i != part])
    x_train = torch.cat([loaded_data[i][0] for i in range(len(loaded_data)) if i != part])
    return x_train, y_train, x_test, y_test


class UniversalSHAPAnalyzer:
    def __init__(self, model, X_train, X_test=None, feature_names=None, max_display=20, background_sample=100):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test if X_test is not None else X_train[:100]
        self.explainer = None
        self.shap_values = None
        self.feature_names = feature_names if feature_names is not None else [f'Feature_{i}' for i in
                                                                              range(len(X_train[0]))]
        self.max_display = max_display
        self.background_sample = background_sample

    def create_explainer(self):
        try:
            self.explainer = shap.TreeExplainer(self.model, self.X_train, feature_perturbation='interventional')
        except:
            try:
                self.explainer = shap.LinearExplainer(self.model, self.X_train)
            except:
                background = shap.sample(self.X_train, min(self.background_sample, len(self.X_train)))
                self.explainer = shap.KernelExplainer(self.model.predict, background)
        return self.explainer

    def calculate_shap_values(self):
        if self.explainer is None:
            self.create_explainer()
        self.shap_values = self.explainer.shap_values(self.X_train, check_additivity=False)
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[0]
        if len(self.shap_values.shape) == 2:
            n_samples, n_features = self.shap_values.shape
            binary_shap_3d = np.zeros((n_samples, n_features, 2))
            binary_shap_3d[:, :, 0] = -self.shap_values
            binary_shap_3d[:, :, 1] = self.shap_values
            self.shap_values = binary_shap_3d
        return self.shap_values

    def plot_importance(self, model_name, save_dir=""):
        if self.shap_values is None:
            self.calculate_shap_values()
        shap_sum = np.mean(abs(self.shap_values), axis=0)
        fet_imp = [shap_sum[:, 0]]
        for i in range(shap_sum.shape[1] - 1):
            fet_imp.append(fet_imp[-1] + shap_sum[:, i + 1])
        sorted_indices = np.argsort(fet_imp[-1])[::-1]
        fet_name_sele = [self.feature_names[i] for i in sorted_indices[:self.max_display]]
        fet_imp_classified = [[i[j] for j in sorted_indices[:self.max_display]] for i in fet_imp]
        fig, ax = plt.subplots(figsize=(10, 8))
        if len(fet_imp_classified) == 2:
            colors = 'bb'
        else:
            colors = plt.cm.Set3(np.linspace(0, 1, len(fet_imp_classified)))
        y_positions = np.arange(len(fet_name_sele))
        fet_name_reversed = fet_name_sele[::-1]
        left = np.zeros(len(fet_name_sele))
        for i, (imp_values, color) in enumerate(zip(fet_imp_classified, colors)):
            imp_values_reversed = imp_values[::-1]
            ax.barh(y_positions, imp_values_reversed, left=left,
                    color=color, label=f'Class {i + 1}', alpha=0.8)
            left += imp_values_reversed
        ax.set_yticks(y_positions)
        ax.set_yticklabels(fet_name_reversed)
        ax.set_xlabel('Feature Importance')
        ax.set_ylabel('Features')
        ax.set_title('Feature Importance by Class (Stacked Horizontal Bar Chart)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, model_name + "_feature_importance.pdf"), format='pdf')

    def get_feature_importance(self):
        if self.shap_values is None:
            self.calculate_shap_values()
        return np.abs(self.shap_values).mean(axis=0)

    def plot_waterfall(self, sample_idx=0, max_display=20):
        if self.shap_values is None:
            self.calculate_shap_values()
        sample_data = self.X_test[sample_idx]
        if isinstance(self.explainer.expected_value, (list, np.ndarray)):
            expected_value = self.explainer.expected_value[0]
        else:
            expected_value = self.explainer.expected_value
        explanation = shap.Explanation(
            values=self.shap_values[sample_idx],
            base_values=expected_value,
            data=sample_data,
            feature_names=self.feature_names
        )
        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(
            explanation,
            max_display=max_display,
            show=False
        )
        plt.title(f'SHAP Waterfall Plot - Sample {sample_idx}', fontsize=14, pad=20)
        plt.tight_layout()
        plt.show()
