"""
Machine-learning and SHAP utility functions for MassLinker.

This module provides helper functions and model wrappers for supervised
predictive modeling based on MassLinker metabolic tokens. It includes
traditional machine-learning classifiers, cross-validation result handling,
ROC visualization, performance metric summaries, and SHAP-based model
interpretability utilities.

MassLinker represents each sample as high-dimensional RBF-derived metabolic
features. These functions flatten or organize the token representation for
machine-learning models such as SVM, random forest, XGBoost, and LightGBM,
and provide tools for evaluating diagnostic performance and interpreting
feature contributions.
"""

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


# Global plotting configuration.
# These settings improve compatibility with vector-graphic outputs and
# keep text editable in PDF/SVG figures for publication-ready visualization.
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
    Load external ROC data and format it for combined ROC visualization.

    Each input CSV file is expected to contain true labels in the first column
    and prediction scores in the second column. The function converts binary
    labels and scores into two-column arrays so that they can be plotted
    together with MassLinker model ROC curves.

    Parameters
    ----------
    paths : list
        List of CSV file paths containing external ROC data.
    model_names : list, optional
        Names of external models. If None, default model names are generated.
    colors : list, optional
        Colors used for plotting external ROC curves.

    Returns
    -------
    list
        A list of tuples containing true-label arrays, score arrays, color,
        and model name.
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

            # The first column stores binary true labels.
            y_true = data.iloc[:, 0].values
            y_ture_2d = np.column_stack([1 - y_true, y_true])

            # The second column stores prediction scores for the positive class.
            y_score = data.iloc[:, 1].values
            y_score_2d = np.column_stack([1 - y_score, y_score])

            addition_rocs.append((y_ture_2d, y_score_2d, color, model_name))

        except Exception as e:
            print(f"Error processing file {path}: {e}")
            continue

    return addition_rocs


def save_model_results(models, fold, name, save_dir="./"):
    """
    Save prediction results and true labels for one cross-validation fold.

    The function collects predictions stored in an `ML_models` object, appends
    the true labels of the selected test samples, and writes the fold-level
    results to an Excel file.

    Parameters
    ----------
    models : ML_models
        Trained model wrapper containing prediction results and test labels.
    fold : int
        Fold index used in the output filename.
    name : str
        Subdirectory name used to organize fold-level outputs.
    save_dir : str, optional
        Root output directory.
    """
    results = copy.deepcopy(models.predict_result)
    results.append(list())

    # Append true labels for the current test fold.
    for i in copy.deepcopy(models.y_test[models.idx]):
        results[-1].append(i.item())

    temp = pd.DataFrame(results).T

    if not os.path.exists(os.path.join(save_dir, name)):
        os.makedirs(os.path.join(save_dir, name))

    temp.to_excel(os.path.join(save_dir, name, "fold" + str(fold) + ".xlsx"))


def fold_n(split_idx, n):
    """
    Generate train/test indices for the n-th fold.

    Parameters
    ----------
    split_idx : list
        List of fold-specific sample indices.
    n : int
        Index of the fold used as the test set.

    Returns
    -------
    tuple
        Test indices and training indices for the selected fold.
    """
    test_idx = list()

    for i in range(len(split_idx)):
        if not i == n:
            for j in split_idx[i]:
                test_idx.append(j)

    return split_idx[n], test_idx


class ML_models:
    """
    Wrapper for traditional machine-learning models used in MassLinker analysis.

    This class initializes and trains multiple supervised classifiers, including
    SVM, XGBoost, random forest, and LightGBM. The input MassLinker token tensors
    are flattened into feature vectors using `data_transform` before model
    fitting and prediction.

    Parameters
    ----------
    x_train : tuple or list
        Training data structure containing MassLinker features and metadata.
    y_train : tuple or list
        Training labels.
    x_test : tuple or list
        Test data structure containing MassLinker features and metadata.
    y_test : tuple or list
        Test labels.
    """

    def __init__(self, x_train, y_train, x_test, y_test):
        """
        Initialize the model wrapper and construct all supported classifiers.
        """
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
        """
        Store training and test data and infer the number of target classes.

        Parameters
        ----------
        x_train : tuple or list
            Training data structure.
        y_train : tuple or list
            Training labels.
        x_test : tuple or list
            Test data structure.
        y_test : tuple or list
            Test labels.
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        # The class number is inferred from the class-label field.
        self.class_num = len(set(x_train[4]))

    def init_SVM(self, kernel='rbf', C=1.0, gamma='scale'):
        """
        Initialize the support vector machine classifier.

        Parameters
        ----------
        kernel : str, optional
            SVM kernel type.
        C : float, optional
            Regularization parameter.
        gamma : str or float, optional
            Kernel coefficient.
        """
        self.scaler = StandardScaler()
        self.SVM = SVC(kernel=kernel, C=C, gamma=gamma)

    def init_XGB(self, max_depth=50, eta=0.001, num_round=200, n_worker=20):
        """
        Initialize the XGBoost classifier.

        Parameters
        ----------
        max_depth : int, optional
            Maximum tree depth.
        eta : float, optional
            Learning rate parameter.
        num_round : int, optional
            Number of boosting rounds reserved for compatibility.
        n_worker : int, optional
            Number of worker threads.
        """
        self.param = {'max_depth': max_depth, 'eta': eta, 'objective': 'multi:softmax', 'num_class': self.class_num}
        self.num_round = num_round
        self.XGB = xgb.XGBClassifier(nthread=n_worker)

    def init_RF(self, n_estimators=1000, max_depth=None):
        """
        Initialize the random forest classifier.

        Parameters
        ----------
        n_estimators : int, optional
            Number of trees in the forest.
        max_depth : int, optional
            Maximum depth of each tree.
        """
        self.RF = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

    def init_lgb(self):
        """
        Initialize the LightGBM multiclass classifier.

        LightGBM is used as a gradient-boosted decision-tree model for
        supervised predictive modeling on flattened MassLinker token features.
        """
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
        """
        Fit all initialized machine-learning models.

        Parameters
        ----------
        fit_mode : str, optional
            If `single`, binary group labels are used. Otherwise, multiclass
            class labels are used.
        """
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
        """
        Generate predictions for all trained machine-learning models.

        Predictions are stored in `self.predict_result` in the order:
        SVM, XGBoost, random forest, and LightGBM.
        """
        self.predict_result.append(self.SVM.predict(data_transform(self.x_test[0])))
        self.predict_result.append(self.XGB.predict(data_transform(self.x_test[0])))
        self.predict_result.append(self.RF.predict(data_transform(self.x_test[0])))
        self.predict_result.append(self.lgb.predict(data_transform(self.x_test[0])))

    def validation(self, pos_label=1):
        """
        Print classification metrics for all stored prediction results.

        Parameters
        ----------
        pos_label : int, optional
            Positive class label used by scikit-learn metric functions.
        """
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
        """
        Calculate class probabilities for models that support probability outputs.

        The probability outputs are used for ROC and AUC visualization.
        """
        self.predict_proba_result = {
            'XGB': self.XGB.predict_proba(data_transform(self.x_test[0])),
            'RF': self.RF.predict_proba(data_transform(self.x_test[0])),
            'LGB': self.lgb.predict_proba(data_transform(self.x_test[0]))[:, :2]
        }

    def plot_combined_roc(self, figsize=(12, 8), save_path=None, addition_ROCs=None,
                          titles='ROC Curves Comparison (Micro-Average)'):
        """
        Plot combined micro-average ROC curves for multiple models.

        The function plots ROC curves for XGBoost, random forest, and LightGBM.
        External ROC curves can also be added through `addition_ROCs`.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size.
        save_path : str, optional
            Output PDF path. If None, the figure is displayed interactively.
        addition_ROCs : list, optional
            Additional ROC data returned by `addition_roc`.
        titles : str, optional
            Figure title.
        """
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
    """
    Initialize containers for cross-validation performance metrics.

    Returns
    -------
    tuple
        Dictionaries for accuracy, precision, recall, F1-score, and ROC data.
    """
    model_names = ['SVM', 'XGB', 'RF', 'LightGBM']
    all_accuracy = {f'{i}': [] for i in model_names}
    all_precision = {f'{i}': [] for i in model_names}
    all_recall = {f'{i}': [] for i in model_names}
    all_f1 = {f'{i}': [] for i in model_names}
    roc_data = {f'{i}': {'fpr': [], 'tpr': [], 'auc': []} for i in model_names}

    return all_accuracy, all_precision, all_recall, all_f1, roc_data


def cal_all_index(dic):
    """
    Calculate performance metrics from fold-level prediction results.

    Parameters
    ----------
    dic : dict
        Dictionary containing fold-level prediction result data frames.

    Returns
    -------
    tuple
        metrics_data and roc_data. `metrics_data` stores fold-level metric
        values by model, and `roc_data` stores ROC curve coordinates and AUCs.
    """
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
    """
    Draw box plots for cross-validation performance metrics.

    Parameters
    ----------
    metrics_data : dict
        Metric dictionary returned by `cal_all_index`.
    """
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
    """
    Convert metric dictionaries into a long-format data frame for plotting.

    Parameters
    ----------
    metrics_data : dict
        Metric dictionary returned by `cal_all_index`.

    Returns
    -------
    pandas.DataFrame
        Long-format table with Model, Metric, and Value columns.
    """
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
    """
    Draw and save bar plots for each performance metric.

    Parameters
    ----------
    metrics_df : pandas.DataFrame
        Long-format metric table generated by `prepare_barplot_data`.
    id : str
        Output identifier used as a subdirectory name.
    """
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

        for p in ax.patches:
            height = p.get_height()
            text_value = f'{height:.2f}'
            ax.text(p.get_x() + p.get_width() / 2.,
                    height,
                    text_value,
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    color='black')

        plt.title(metric_name)
        plt.ylabel(f'average {metric_name}')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.ylim(bottom=0)
        plt.tight_layout()
        plt.savefig(fname="./barplots/" + id + "/" + subset_df.iloc[0, 1] + "_enhance.pdf", format="pdf")


def visualization(id):
    """
    Load fold-level prediction files and generate performance bar plots.

    Parameters
    ----------
    id : str
        Directory identifier containing fold-level Excel result files.
    """
    path = "./" + id
    dic = {}

    for file in os.listdir(path):
        dic[file] = pd.read_excel(os.path.join(path, file))

    metrics_data, roc_data = cal_all_index(dic)
    metrics_df = prepare_barplot_data(metrics_data)
    bar_plot(metrics_df, id)


def cross_dataset_data_prepare(loaded_data, part):
    """
    Prepare train/test tensors for cross-dataset or fold-based evaluation.

    One partition is used as the test set, while all remaining partitions are
    concatenated as the training set.

    Parameters
    ----------
    loaded_data : list
        List of partitioned dataset objects or tuples.
    part : int
        Index of the partition used as the test set.

    Returns
    -------
    tuple
        x_train, y_train, x_test, and y_test.
    """
    y_test = loaded_data[part][3]
    x_test = loaded_data[part][0]

    y_train = torch.cat([loaded_data[i][3] for i in range(len(loaded_data)) if i != part])
    x_train = torch.cat([loaded_data[i][0] for i in range(len(loaded_data)) if i != part])

    return x_train, y_train, x_test, y_test


class UniversalSHAPAnalyzer:
    """
    Unified SHAP analyzer for conventional machine-learning models.

    This class provides a model-agnostic interface for SHAP-based
    interpretability analysis. It first attempts to use a tree-based SHAP
    explainer, then a linear explainer, and finally falls back to a kernel
    explainer when necessary.

    The resulting SHAP values can be summarized to estimate feature-level
    contribution patterns for MassLinker-derived features.

    Parameters
    ----------
    model : object
        Trained machine-learning model.
    X_train : np.ndarray
        Training feature matrix used for SHAP estimation.
    X_test : np.ndarray, optional
        Test feature matrix used for sample-level explanation.
    feature_names : list, optional
        Feature names corresponding to columns in the feature matrix.
    max_display : int, optional
        Number of top features to display in importance plots.
    background_sample : int, optional
        Number of background samples used by the kernel explainer fallback.
    """

    def __init__(self, model, X_train, X_test=None, feature_names=None, max_display=20, background_sample=100):
        """
        Initialize a universal SHAP analyzer.
        """
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
        """
        Create a SHAP explainer compatible with the provided model.

        The function tries TreeExplainer first, then LinearExplainer, and
        finally KernelExplainer as a general fallback.

        Returns
        -------
        object
            Initialized SHAP explainer.
        """
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
        """
        Calculate SHAP values for the training feature matrix.

        For binary outputs represented as two-dimensional SHAP values, the
        function expands them into a three-dimensional class-wise structure
        for downstream plotting compatibility.

        Returns
        -------
        np.ndarray
            SHAP values for samples, features, and classes.
        """
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
        """
        Plot class-wise stacked feature importance based on SHAP values.

        Global feature importance is calculated as the mean absolute SHAP value
        across samples, following the SHAP-based interpretability strategy
        described for MassLinker classifiers.

        Parameters
        ----------
        model_name : str
            Model name used in the output filename.
        save_dir : str, optional
            Directory used to save the output PDF.
        """
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
        """
        Return global feature importance based on mean absolute SHAP values.

        Returns
        -------
        np.ndarray
            Feature importance array averaged across samples.
        """
        if self.shap_values is None:
            self.calculate_shap_values()

        return np.abs(self.shap_values).mean(axis=0)

    def plot_waterfall(self, sample_idx=0, max_display=20):
        """
        Plot a SHAP waterfall explanation for a selected sample.

        Parameters
        ----------
        sample_idx : int, optional
            Index of the sample to explain.
        max_display : int, optional
            Maximum number of features displayed in the waterfall plot.
        """
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
