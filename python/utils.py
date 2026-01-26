import numpy as np
import torch
from numpy import floating
from numpy._typing import _64Bit
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from typing import Callable
import os
import matplotlib.cm as cm
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
import seaborn as sns
import joblib
from scipy import stats
from statsmodels.stats.multitest import multipletests
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import adjustText
from sklearn.preprocessing import StandardScaler
import umap
import matplotlib.colors as mcolors

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


def calculate_js_divergence(
        f: Callable[[np.ndarray, torch.Tensor], np.ndarray],
        w1: torch.Tensor,
        w2: torch.Tensor,
        t_start: float = 0,
        t_end: float = 1800,
        num_points: int = 1800
):
    t = np.linspace(t_start, t_end, num_points)
    p_observed = f(t, w1)
    q_observed = f(t, w2)
    p_observed[p_observed < 0] = 0
    q_observed[q_observed < 0] = 0
    p_sum = np.sum(p_observed)
    q_sum = np.sum(q_observed)
    if p_sum > 0:
        p_dist = p_observed / p_sum
    else:
        p_dist = np.full(num_points, 1.0 / num_points)
    if q_sum > 0:
        q_dist = q_observed / q_sum
    else:
        q_dist = np.full(num_points, 1.0 / num_points)
    return jensenshannon(p_dist, q_dist, base=2)


def calculate_wasserstein_distance(
        f: Callable[[np.ndarray, torch.Tensor], np.ndarray],
        w1: torch.Tensor,
        w2: torch.Tensor,
        t_start: float = 0,
        t_end: float = 1800,
        num_points: int = 18000
) -> float:
    t_values = np.linspace(t_start, t_end, num_points)
    p_observed = f(t_values, w1)
    q_observed = f(t_values, w2)
    p_observed[p_observed < 0] = 0
    q_observed[q_observed < 0] = 0
    p_sum = np.sum(p_observed)
    q_sum = np.sum(q_observed)
    if p_sum > 0:
        p_weights = p_observed / p_sum
    else:
        p_weights = np.full(num_points, 1.0 / num_points)

    if q_sum > 0:
        q_weights = q_observed / q_sum
    else:
        q_weights = np.full(num_points, 1.0 / num_points)
    distance = wasserstein_distance(
        u_values=t_values,
        v_values=t_values,
        u_weights=p_weights,
        v_weights=q_weights
    )
    return distance


def RBF(t: np.ndarray, w: torch.Tensor) -> np.ndarray:
    t_tensor = torch.as_tensor(t, dtype=w.dtype, device=w.device).double()
    weights = w[0].double()
    centers = w[1].double()
    widths = w[2].double()
    diff = t_tensor.unsqueeze(1) - centers
    basis_activations = torch.exp(- (diff ** 2) / (2 * widths ** 2))
    result = basis_activations[:, 0].double() * weights[0].double()
    for i in range(1, 20):
        result += basis_activations[:, i].double() * weights[i].double()
    result = torch.maximum(result, torch.zeros_like(result)).numpy()
    return result


def metabo_dis(params):
    ret_JS = []
    ret_was = []
    center_sample = 0
    ret_JS.append([])
    ret_was.append([])
    for edge_sample in tqdm(range(len(params)), desc="Calculating Samples distance"):
        par_center = params[center_sample]
        par_edge = params[edge_sample]
        ret_JS[-1].append([])
        ret_was[-1].append([])
        for met in range(len(params[0])):
            ret_JS[-1][-1].append(calculate_js_divergence(RBF, par_center[met], par_edge[met]))
            ret_was[-1][-1].append(calculate_wasserstein_distance(RBF, par_center[met], par_edge[met]))
    return ret_JS, ret_was


def plots(JS, was, group):
    JS_temp = np.array(JS[0])
    was_temp = np.array(was[0])
    JS_temp[np.isinf(JS_temp)] = 1
    was_temp[np.isinf(was_temp)] = 1
    n_samples = 6
    pca = PCA(n_components=2)
    JS_pca = pca.fit_transform(JS_temp)
    was_pca = pca.fit_transform(was_temp)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=JS_pca[:, 0], y=JS_pca[:, 1], hue=group, palette='viridis', s=1000)
    plt.title('JS Divergence - PCA (2D)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Sample Index')
    plt.grid(True)
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=was_pca[:, 0], y=was_pca[:, 1], hue=group, palette='viridis', s=100)
    plt.title('Wasserstein Distance - PCA (2D)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Sample Index')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("pca.pdf")
    perplexity_value = min(5, n_samples - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42, init='pca', learning_rate='auto')
    JS_tsne = tsne.fit_transform(JS_temp)
    was_tsne = tsne.fit_transform(was_temp)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=JS_tsne[:, 0], y=JS_tsne[:, 1], hue=group, palette='viridis', s=100)
    plt.title('JS Divergence - t-SNE (2D)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(title='Sample Index')
    plt.grid(True)
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=was_tsne[:, 0], y=was_tsne[:, 1], hue=group, palette='viridis', s=100)
    plt.title('Wasserstein Distance - t-SNE (2D)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(title='Sample Index')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("tsne.pdf")


def kegg_enrichment(pathway_df_full, p_values, was_diff):
    significant_rows = pathway_df_full[np.array(p_values) < 0.05]
    significant_compounds_set = set(significant_rows['compound_names'].unique())
    universe_compounds = set(pathway_df_full['compound_names'].unique())
    pathways = pathway_df_full[['kegg_id', 'pathway_name']].drop_duplicates()
    N = len(universe_compounds)
    M = len(significant_compounds_set)
    enrichment_results = []
    desc = "执行通路富集分析"
    j = 0
    for index, row in tqdm(pathways.iterrows(), total=pathways.shape[0], desc=desc):
        pathway_id = row['kegg_id']
        pathway_name = row['pathway_name']
        compounds_in_pathway = set(pathway_df_full[pathway_df_full['kegg_id'] == pathway_id]['compound_names'])
        K = len(compounds_in_pathway)
        PPS = 0
        for idx in range(K):
            PPS += abs(was_diff[j]) / K
            j += 1
        hits_in_pathway_set = significant_compounds_set.intersection(compounds_in_pathway)
        k = len(hits_in_pathway_set)
        if k > 0:
            p_val = stats.hypergeom.sf(k - 1, N, M, K)
            enrichment_ratio = (k / K) / (M / N)
            hit_compounds_str = ", ".join(list(hits_in_pathway_set))
            enrichment_results.append({
                'Pathway_ID': pathway_id,
                'Pathway_Name': pathway_name,
                'P_Value': p_val,
                'Enrichment_Ratio': enrichment_ratio,
                'Hits_in_Pathway (k)': k,
                'Total_in_Pathway (K)': K,
                'Hit_Compounds': hit_compounds_str,
                "PPS": PPS * (-math.log10(p_val))
            })
    if not enrichment_results:
        print("警告: 没有在任何通路中找到显著代谢物。")
        return pd.DataFrame()
    results_df = pd.DataFrame(enrichment_results)
    reject, q_values, _, _ = multipletests(results_df['P_Value'], alpha=0.05, method='fdr_bh')
    results_df['Q_Value (FDR_BH)'] = q_values
    results_df = results_df.sort_values(by='P_Value', ascending=True)
    output_filename = 'metabolite_enrichment_analysis_results.xlsx'
    results_df.to_excel(output_filename, index=False)
    print(f"\n富集分析完成，结果已保存至: {output_filename}")
    return results_df


def plot_enrichment(plot_df, topn_p=3, topn_pps=3, save_dir=None, save_name="KEGG_Enriched.pdf",
                    low_p_color='#D15D73', high_p_color='#B3B3B3'):
    plot_df = plot_df.copy()
    plot_df.dropna(subset=['P_Value'], inplace=True)
    plot_df['neg_log10_p'] = -np.log10(plot_df['P_Value'].replace(0, np.nextafter(0, 1)))
    fig = plt.figure(figsize=(13, 11))
    ax = fig.add_subplot(111, projection='3d')
    cmap_name = f"custom_{low_p_color}_{high_p_color}"
    colors = [low_p_color, high_p_color]
    custom_cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors)
    norm = mcolors.Normalize(vmin=plot_df['P_Value'].min(), vmax=plot_df['P_Value'].max())
    x_axis = plot_df['Enrichment_Ratio']
    y_axis = plot_df['PPS']
    z_axis = plot_df['neg_log10_p']
    scatter = ax.scatter(x_axis, y_axis, z_axis, c=plot_df['P_Value'], cmap=custom_cmap, norm=norm,
                         s=60, alpha=0.8, edgecolors='k', linewidth=0.5)
    ax.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax.set_xlabel('Enrichment Ratio', fontsize=12, labelpad=15)
    ax.set_ylabel('PPS', fontsize=12, labelpad=15)
    ax.set_zlabel('-log10(p_value)', fontsize=12, labelpad=15)
    ax.set_title('KEGG Enrichment', fontsize=16, pad=20)

    ax.view_init(elev=2, azim=315)

    cbar = fig.colorbar(scatter, shrink=0.6, aspect=20)
    cbar.set_label('P-value', fontsize=12)
    annotation_color = 'black'
    if len(plot_df) > 0:
        num_to_plot_p = min(len(plot_df), topn_p)
        top_pathways_p = plot_df.nsmallest(num_to_plot_p, 'P_Value')
        for _, row in top_pathways_p.iterrows():
            ax.text(row['Enrichment_Ratio'], row['PPS'], row['neg_log10_p'],
                    f"  {row['Pathway_Name']}",
                    color=annotation_color, fontsize=9, ha='left', va='center')
    if len(plot_df) > 0:
        num_to_plot_pps = min(len(plot_df), topn_pps)
        top_pathways_pps = plot_df.nlargest(num_to_plot_pps, 'PPS')
        already_labeled_indices = top_pathways_p.index if 'top_pathways_p' in locals() else []

        for idx, row in top_pathways_pps.iterrows():
            if idx not in already_labeled_indices:
                ax.text(row['Enrichment_Ratio'], row['PPS'], row['neg_log10_p'],
                        f"  {row['Pathway_Name']}",
                        color=annotation_color, fontsize=9, ha='left', va='center')

    plt.tight_layout()

    if save_dir is None:
        save_path = save_name
    else:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, save_name)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def get_diff(dis_mat, group):
    if isinstance(group,torch.Tensor):
        group = group.cpu().numpy()
    group_0_data = dis_mat[np.array(group) == list(set(group))[0]]
    group_1_data = dis_mat[np.array(group) == list(set(group))[1]]
    num_features = dis_mat.shape[1]
    p_values = []
    for i in tqdm(range(num_features)):
        feature_data_g0 = group_0_data[:, i]
        feature_data_g1 = group_1_data[:, i]
        t_statistic, p_value = stats.ttest_ind(feature_data_g0, feature_data_g1, equal_var=False)
        p_values.append(p_value)
    was_diff = sum(group_0_data) / len(group_0_data) - (sum(group_1_data) / len(group_1_data))
    return p_values, was_diff


def plot_peak_comp(p_values, group, params, met_name, mzs, top_n=100, plot_num=180000, save_path="/", window_size=5):
    p_values_arr = np.array(p_values)
    smallest_indices = np.argsort(p_values_arr)[:top_n]
    RBF_points = []
    x = np.linspace(0, 1800, plot_num)
    filename = []
    p_s = []
    group = np.array(group)
    # kernel = np.ones(window_size) / window_size
    for idx in tqdm(smallest_indices, desc="Outputting peak compare plots"):
        rbf_data = [RBF(x, params[i][idx]) for i in range(len(params))]
        filename = met_name[idx]
        p_s = p_values[idx]
        # for j in tqdm(range(len(RBF_points)), desc="Outputting peak compare plots"):
        # rbf_data = RBF_points[j]
        plt.figure(figsize=(12, 6))
        unique_groups = np.unique(group)
        color_map = {unique_groups[0]: '#51AAD1', unique_groups[1]: '#D15D73'}
        for i in range(len(rbf_data)):
            plt.plot(x, rbf_data[i],
                     color=color_map[group[i]],
                     alpha=0.7,
                     label=f'Group {group[i]}' if i == 0 or group[i] != group[i - 1] else '')
        plt.xlabel('X')
        plt.ylabel('RBF Value')
        plt.title('RBF Curves for Different Samples')
        plt.legend(loc='upper right')
        plt.text(0.05, 0.95, f'Metabolite: {filename}\np-value: {str(p_s)}\nmz={mzs[idx]}',
                 transform=plt.gca().transAxes,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, filename + ".pdf"), format='pdf')
        plt.close()


def plot_2d(JS, was, group, save_path="/", color_map=None):
    if color_map is None:
        color_map = ["#51AAD1", '#D15D73']
    JS_temp = np.array(JS[0])
    was_temp = np.array(was[0])
    JS_temp[np.isinf(JS_temp)] = 1
    was_temp[np.isinf(was_temp)] = 1
    n_samples = 6

    pca = PCA(n_components=2)
    JS_pca = pca.fit_transform(JS_temp)
    was_pca = pca.fit_transform(was_temp)

    plt.figure(figsize=(11, 5))

    plt.subplot(1, 2, 1)
    sns.scatterplot(x=JS_pca[:, 0], y=JS_pca[:, 1], hue=group, palette=color_map, s=100)
    plt.title('JS Divergence - PCA (2D)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Group')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    sns.scatterplot(x=was_pca[:, 0], y=was_pca[:, 1], hue=group, palette=color_map, s=100)
    plt.title('Wasserstein Distance - PCA (2D)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Group')
    plt.grid(True)

    plt.tight_layout()
    if not os.path.exists(save_path): os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, "pca.pdf"))

    perplexity_value = min(5, n_samples - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42, init='pca', learning_rate='auto')
    JS_tsne = tsne.fit_transform(JS_temp)
    was_tsne = tsne.fit_transform(was_temp)

    plt.figure(figsize=(11, 5))

    plt.subplot(1, 2, 1)
    sns.scatterplot(x=JS_tsne[:, 0], y=JS_tsne[:, 1], hue=group, palette=color_map, s=100)
    plt.title('JS Divergence - t-SNE (2D)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(title='Group')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    sns.scatterplot(x=was_tsne[:, 0], y=was_tsne[:, 1], hue=group, palette=color_map, s=100)
    plt.title('Wasserstein Distance - t-SNE (2D)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(title='Group')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "tsne.pdf"))


def visualization_foldn_valid(data_path, save_dir, model_names, save_name, addition_ROCs=None,
                              title='Model Performance Comparison (N-Fold Cross Validation)'):
    records = []
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    for files in os.listdir(data_path):
        if files[-4:]=="xlsx":
            records.append(pd.read_excel(os.path.join(data_path, files),engine='openpyxl'))
    dataframes = []
    for i in range(records[0].shape[1] - 2):
        dataframes.append([])
        for j in records:
            accuracy = accuracy_score(j.iloc[:, i + 1], j.iloc[:, records[0].shape[1] - 1])
            precision = precision_score(j.iloc[:, i + 1], j.iloc[:, records[0].shape[1] - 1], average='weighted')
            recall = recall_score(j.iloc[:, i + 1], j.iloc[:, records[0].shape[1] - 1], average='weighted')
            f1 = f1_score(j.iloc[:, i + 1], j.iloc[:, records[0].shape[1] - 1], average='weighted')
            dataframes[-1].append({
                'acc': accuracy,
                'pre': precision,
                'rec': recall,
                'f1': f1
            })
    addition_df = []
    if addition_ROCs is not None:
        model_groups = {}
        for y_trues, y_scores, color, model_name in addition_ROCs:
            model_names.append(model_name)
            if model_name not in model_groups:
                model_groups[model_name] = []
            model_groups[model_name].append((y_trues, y_scores))
        for model_name, model_data in model_groups.items():
            for y_trues, y_scores in model_data:
                if y_trues.ndim == 2:
                    y_true_1d = y_trues[:, 1]
                else:
                    y_true_1d = y_trues
                if y_scores.ndim == 2:
                    y_pred = np.argmax(y_scores, axis=1)
                else:
                    y_pred = (y_scores > 0.5).astype(int)
                accuracy = accuracy_score(y_true_1d, y_pred)
                precision = precision_score(y_true_1d, y_pred, average='weighted')
                recall = recall_score(y_true_1d, y_pred,average='weighted')
                f1 = f1_score(y_true_1d, y_pred, average='weighted')
                addition_df.append({
                    'acc': accuracy,
                    'pre': precision,
                    'rec': recall,
                    'f1': f1
                })
    barplot_foldn_valid(dataframes, model_names, save_dir, save_name, addition_df=addition_df, title=title)


def barplot_foldn_valid(dataframes, model_names, save_dir, save_name,
                        addition_df=None, colors=None, title='Model Performance Comparison (N-Fold Cross Validation)'):
    if colors is None:
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    metrics = ['acc', 'pre', 'rec', 'f1']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    means = {metric: [] for metric in metrics}
    stds = {metric: [] for metric in metrics}
    for model_results in dataframes:
        for metric in metrics:
            values = [fold[metric] for fold in model_results]
            means[metric].append(np.mean(values))
            stds[metric].append(np.std(values))
    if addition_df is not None:
        additional_models = {}
        start_idx = len(dataframes)
        for i, metrics_dict in enumerate(addition_df):
            model_idx = start_idx + i
            if model_idx < len(model_names):
                model_name = model_names[model_idx]
                if model_name not in additional_models:
                    additional_models[model_name] = []
                additional_models[model_name].append(metrics_dict)
        for model_name, model_data in additional_models.items():
            for metric in metrics:
                values = [fold[metric] for fold in model_data]
                means[metric].append(np.mean(values))
                stds[metric].append(np.std(values) if len(values) > 1 else 0)
    n_models = len(model_names)
    n_metrics = len(metrics)
    x = np.arange(n_models)
    width = 0.2
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        offset = width * (i - n_metrics / 2 + 0.5)
        bars = ax.bar(x + offset, means[metric], width,
                      label=label,
                      color=colors[i % len(colors)],
                      yerr=stds[metric],
                      capsize=5,
                      alpha=0.8,
                      edgecolor='black',
                      linewidth=1)
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.,
                    height + stds[metric][j] + 0.01,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=16)
    ax.set_xlabel('Models', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=12, rotation=45 if n_models > 6 else 0)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0.5, 1.15)
    if dataframes:
        n_folds = len(dataframes[0])
        ax.text(0.02, 0.98, f'{n_folds}-Fold Cross Validation',
                transform=ax.transAxes, fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, save_name), dpi=300, bbox_inches='tight')


def plot_MetTD(e_p_values, e_scores, e_names, permutation_time, save_path="MetTD_result.pdf", top_n=5,
               fig_size=(12, 8)):
    adjusted_p_values = np.where(e_p_values == 0, 1 / permutation_time, e_p_values)
    neg_log_p = -np.log10(adjusted_p_values)
    fig, ax = plt.subplots(figsize=fig_size)
    abs_scores = np.abs(e_scores)
    log_scores = np.log(abs_scores + 1e-10)
    sizes = (log_scores - log_scores.min()) / (log_scores.max() - log_scores.min()) * 19 + 1
    scatter = ax.scatter(e_scores, neg_log_p,
                         c=-adjusted_p_values,
                         s=sizes,
                         cmap='plasma_r',
                         edgecolors='none',
                         alpha=0.7)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('P-value', fontsize=12)
    ax.set_ylabel('-log₁₀(p-value)', fontsize=12)
    ax.set_xlabel('Score', fontsize=12)
    ax.set_title('Score vs -log₁₀(p-value) Scatter Plot', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    if top_n > 0 and len(e_names) > 0:
        top_indices = np.lexsort((-np.array(e_scores), adjusted_p_values))[:top_n]
        texts = []
        for idx in top_indices:
            text = ax.annotate(e_names[idx],
                               (e_scores[idx], neg_log_p[idx] - 0.1),
                               xytext=(5, 5),
                               textcoords='offset points',
                               fontsize=9,
                               bbox=dict(boxstyle='round,pad=0.3',
                                         facecolor='white',
                                         alpha=0.8,
                                         edgecolor='gray'),
                               alpha=0.9)
            texts.append(text)
        # adjustText.adjust_text(texts, force_text=0.001)
    ax.text(0.85, 0.05, f'Total points: {len(e_scores)}\nTop {top_n} labeled',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', dpi=300)


def low_dim_plots(plot1, groups, figsize=None, random_state=42, save_path="2D_plot.pdf", dim=2, group_names=None,colors=None):
    if isinstance(plot1, list):
        X = np.array(plot1)
    else:
        X = np.array(plot1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    if len(groups) > 0 and (isinstance(groups[0], (list, np.ndarray, pd.Series)) or hasattr(groups[0], 'numpy')):
        if hasattr(groups[0], '__len__') and len(groups[0]) == len(X):
            processed_groups = groups
        else:
            processed_groups = [groups]
    else:
        processed_groups = [groups]
    n_rows = len(processed_groups)
    final_groups = []
    for g in processed_groups:
        if hasattr(g, 'numpy'):
            g = g.numpy()
        elif hasattr(g, 'values'):
            g = g.values
        else:
            g = np.array(g)
        final_groups.append(g)
    print("Starting Dimensionality Reduction...")
    results = {}
    pca = PCA(n_components=dim, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    results['pca'] = X_pca
    pca_for_tsne = PCA(n_components=min(500, len(X_scaled[0])), random_state=random_state)
    X_encoded_tsne = pca_for_tsne.fit_transform(X_scaled)
    tsne = TSNE(n_components=min(dim, 3), random_state=random_state, perplexity=50, n_iter=2000)
    X_tsne = tsne.fit_transform(X_encoded_tsne)
    results['tsne'] = X_tsne
    umap_reducer = umap.UMAP(n_components=dim, random_state=random_state, n_neighbors=15, min_dist=0.01)
    X_umap = umap_reducer.fit_transform(X_scaled)
    results['umap'] = X_umap
    print("Coordinates calculated. Plotting...")
    if figsize is None:
        figsize = (18, 5 * n_rows)
    fig, axes = plt.subplots(n_rows, 3, figsize=figsize, squeeze=False)
    if n_rows == 1:
        fig.suptitle('Dimensionality Reduction Visualization', fontsize=16, y=1.02)
    col_titles = ['PCA', 't-SNE', 'UMAP']
    coords_list = [X_pca, X_tsne, X_umap]
    for row_idx in range(n_rows):
        current_group = final_groups[row_idx]
        unique_labels = np.unique(current_group)
        if colors is None:
            if len(unique_labels) > 12:
                colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
            else:
                colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        row_label = f"Group {row_idx + 1}"
        if group_names and row_idx < len(group_names):
            row_label = group_names[row_idx]
        for col_idx in range(3):
            ax = axes[row_idx, col_idx]
            X_coords = coords_list[col_idx]
            for i, label in enumerate(unique_labels):
                mask = np.where(current_group == label)[0]
                ax.scatter(X_coords[mask, 0], X_coords[mask, 1],
                           color=colors[i % len(colors)],
                           label=f'{label}',
                           alpha=0.7, s=30, edgecolors='w', linewidth=0.5)
            ax.grid(True, alpha=0.3)
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=14, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(row_label, fontsize=14, fontweight='bold', labelpad=10)
            if len(unique_labels) < 20:
                ax.legend(fontsize='small', markerscale=0.8, loc='upper right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
        print(f"Saved plot to {save_path}")
    return results


def plot_p_value(p_values, met_names, mzs=None, n=10, figsize=(12, 8),
                 color_palette='viridis', add_significance_lines=True, save_path="p_value_rank.pdf"):
    if mzs is None:
        df = pd.DataFrame({
            'p_value': p_values,
            'name': met_names
        })
        df_sorted = df.sort_values('p_value').head(n)
        df_sorted['-log10_p'] = -np.log10(df_sorted['p_value'])
        plt.figure(figsize=figsize)
        colors = '#C2DDF2'
        bars = plt.barh(range(len(df_sorted)), df_sorted['-log10_p'],
                        color=colors, edgecolor='black', linewidth=0.5)
        for i, (bar, p_val, log_p) in enumerate(zip(bars, df_sorted['p_value'], df_sorted['-log10_p'])):
            plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                     f'{p_val:.2e}', va='center', fontsize=9)
        if add_significance_lines:
            plt.axvline(x=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p=0.05')
            plt.axvline(x=-np.log10(0.01), color='orange', linestyle='--', alpha=0.7, label='p=0.01')
            plt.axvline(x=-np.log10(0.001), color='green', linestyle='--', alpha=0.7, label='p=0.001')
            plt.legend()
        plt.yticks(range(len(df_sorted)), df_sorted['name'])
        plt.xlabel('-log10(p-value)', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.title(f'Top {n} Most Significant Results\n(Lower p-values = Higher bars)',
                  fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, format='pdf')
        return df_sorted
    else:
        df = pd.DataFrame({
            'p_value': p_values,
            'name': met_names,
            'mz': mzs
        })
        df_clean = df.dropna(subset=['mz', 'p_value'])
        df_sorted = df_clean.sort_values('p_value')
        df_unique_mz = df_sorted.loc[df_sorted.groupby('mz')['p_value'].idxmin()]
        df_unique_mz = df_unique_mz.sort_values('p_value').head(n)
        df_unique_mz['-log10_p'] = -np.log10(df_unique_mz['p_value'])
        plt.figure(figsize=figsize)
        colors =  '#C2DDF2'
        bars = plt.barh(range(len(df_unique_mz)), df_unique_mz['-log10_p'],
                        color=colors, edgecolor='black', linewidth=0.5)
        for i, (bar, p_val, mz_val) in enumerate(zip(bars, df_unique_mz['p_value'], df_unique_mz['mz'])):
            plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                     f'{p_val:.2e}', va='center', fontsize=9)
        if add_significance_lines:
            plt.axvline(x=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p=0.05')
            plt.axvline(x=-np.log10(0.01), color='orange', linestyle='--', alpha=0.7, label='p=0.01')
            plt.axvline(x=-np.log10(0.001), color='green', linestyle='--', alpha=0.7, label='p=0.001')
            plt.legend()
        mz_labels = [f'{mz:.4f}' for mz in df_unique_mz['mz']]
        plt.yticks(range(len(df_unique_mz)), mz_labels)
        plt.xlabel('-log10(p-value)', fontsize=12)
        plt.ylabel('m/z Values', fontsize=12)
        plt.title(f'Top {len(df_unique_mz)} Most Significant m/z Values\n(Lower p-values = Higher bars)',
                  fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, format='pdf')
        return df_unique_mz
