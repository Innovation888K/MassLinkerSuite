"""
Utility functions for MassLinker downstream analysis.

This module provides functions for reconstructing chromatographic signals from
MassLinker radial basis function (RBF) tokens, calculating distribution-level
distances, performing two-group differential analysis, conducting KEGG pathway
enrichment analysis, and generating visualization outputs.

MassLinker represents each compound-associated m/z window as a fixed-length
metabolic token. With the default configuration, each token contains three
groups of RBF parameters: peak height, peak position, and peak width.
"""

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


# Global plotting configuration.
# These settings improve compatibility with vector-graphics outputs and
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


def calculate_js_divergence(
        f: Callable[[np.ndarray, torch.Tensor], np.ndarray],
        w1: torch.Tensor,
        w2: torch.Tensor,
        t_start: float = 0,
        t_end: float = 1800,
        num_points: int = 1800
):
    """
    Calculate the Jensen-Shannon distance between two reconstructed chromatographic signals.

    The two signals are reconstructed from MassLinker RBF parameters on a shared
    retention-time grid and then normalized into probability distributions.
    Negative values caused by numerical artifacts are clipped to zero before
    distance calculation.

    Parameters
    ----------
    f : Callable[[np.ndarray, torch.Tensor], np.ndarray]
        Signal reconstruction function. In MassLinker, this is typically `RBF`.
    w1 : torch.Tensor
        RBF parameter tensor for the first chromatographic signal.
    w2 : torch.Tensor
        RBF parameter tensor for the second chromatographic signal.
    t_start : float, optional
        Start of the normalized retention-time range.
    t_end : float, optional
        End of the normalized retention-time range.
    num_points : int, optional
        Number of grid points used for signal reconstruction.

    Returns
    -------
    float
        Jensen-Shannon distance between the two normalized chromatographic
        distributions.
    """
    t = np.linspace(t_start, t_end, num_points)

    # Reconstruct both chromatographic signals on the same retention-time grid.
    p_observed = f(t, w1)
    q_observed = f(t, w2)

    # Remove negative values caused by numerical artifacts after reconstruction.
    p_observed[p_observed < 0] = 0
    q_observed[q_observed < 0] = 0

    # Normalize reconstructed intensities into probability distributions.
    p_sum = np.sum(p_observed)
    q_sum = np.sum(q_observed)

    if p_sum > 0:
        p_dist = p_observed / p_sum
    else:
        # Use a uniform distribution when no valid signal is reconstructed.
        p_dist = np.full(num_points, 1.0 / num_points)

    if q_sum > 0:
        q_dist = q_observed / q_sum
    else:
        # Use a uniform distribution when no valid signal is reconstructed.
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
    """
    Calculate the first-order Wasserstein distance between two reconstructed signals.

    MassLinker reconstructs chromatographic curves from RBF parameters and
    normalizes them as one-dimensional distributions along the retention-time
    axis. The Wasserstein distance measures the displacement between these two
    distributions and captures changes in peak position, shape, and intensity
    distribution.

    Parameters
    ----------
    f : Callable[[np.ndarray, torch.Tensor], np.ndarray]
        Signal reconstruction function. In MassLinker, this is typically `RBF`.
    w1 : torch.Tensor
        RBF parameter tensor for the first chromatographic signal.
    w2 : torch.Tensor
        RBF parameter tensor for the second chromatographic signal.
    t_start : float, optional
        Start of the normalized retention-time range.
    t_end : float, optional
        End of the normalized retention-time range.
    num_points : int, optional
        Number of grid points used for signal reconstruction.

    Returns
    -------
    float
        Wasserstein distance between the two normalized chromatographic
        distributions.
    """
    t_values = np.linspace(t_start, t_end, num_points)

    # Reconstruct both chromatographic signals on the same retention-time grid.
    p_observed = f(t_values, w1)
    q_observed = f(t_values, w2)

    # Remove negative values caused by numerical artifacts after reconstruction.
    p_observed[p_observed < 0] = 0
    q_observed[q_observed < 0] = 0

    # Normalize reconstructed intensities into probability weights.
    p_sum = np.sum(p_observed)
    q_sum = np.sum(q_observed)

    if p_sum > 0:
        p_weights = p_observed / p_sum
    else:
        # Use a uniform distribution when no valid signal is reconstructed.
        p_weights = np.full(num_points, 1.0 / num_points)

    if q_sum > 0:
        q_weights = q_observed / q_sum
    else:
        # Use a uniform distribution when no valid signal is reconstructed.
        q_weights = np.full(num_points, 1.0 / num_points)

    distance = wasserstein_distance(
        u_values=t_values,
        v_values=t_values,
        u_weights=p_weights,
        v_weights=q_weights
    )
    return distance


def RBF(t: np.ndarray, w: torch.Tensor) -> np.ndarray:
    """
    Reconstruct a chromatographic signal from MassLinker RBF parameters.

    Each MassLinker token is composed of three RBF parameter groups:
    peak height, peak position, and peak width. With the default configuration,
    20 RBF components are used to represent the chromatographic signal within
    each compound-associated m/z window.

    Parameters
    ----------
    t : np.ndarray
        Retention-time grid used for signal reconstruction.
    w : torch.Tensor
        RBF parameter tensor. The expected structure is:
        w[0] for peak heights, w[1] for peak centers, and w[2] for peak widths.

    Returns
    -------
    np.ndarray
        Reconstructed non-negative chromatographic intensity values.
    """
    t_tensor = torch.as_tensor(t, dtype=w.dtype, device=w.device).double()

    # MassLinker token structure:
    # w[0] = RBF heights, w[1] = RBF centers, w[2] = RBF widths.
    weights = w[0].double()
    centers = w[1].double()
    widths = w[2].double()

    # Calculate RBF activations for all retention-time points and components.
    diff = t_tensor.unsqueeze(1) - centers
    basis_activations = torch.exp(- (diff ** 2) / (2 * widths ** 2))

    # Sum weighted RBF components.
    result = basis_activations[:, 0].double() * weights[0].double()
    for i in range(1, 20):
        result += basis_activations[:, i].double() * weights[i].double()

    # Enforce non-negative reconstructed chromatographic intensity.
    result = torch.maximum(result, torch.zeros_like(result)).numpy()
    return result


def metabo_dis(params):
    """
    Calculate JS and Wasserstein distances between each sample and a reference sample.

    The first sample is used as the default reference sample. For each sample
    and each MassLinker metabolic token, the corresponding chromatographic
    signal is reconstructed and compared against the reference signal.

    Parameters
    ----------
    params : list
        List of sample-level MassLinker token tensors.

    Returns
    -------
    tuple
        ret_JS and ret_was. Each object stores sample-by-token distance profiles
        relative to the reference sample.
    """
    ret_JS = []
    ret_was = []

    # Use the first sample as the default reference sample.
    center_sample = 0

    ret_JS.append([])
    ret_was.append([])

    for edge_sample in tqdm(range(len(params)), desc="Calculating sample distances"):
        par_center = params[center_sample]
        par_edge = params[edge_sample]

        ret_JS[-1].append([])
        ret_was[-1].append([])

        for met in range(len(params[0])):
            ret_JS[-1][-1].append(calculate_js_divergence(RBF, par_center[met], par_edge[met]))
            ret_was[-1][-1].append(calculate_wasserstein_distance(RBF, par_center[met], par_edge[met]))

    return ret_JS, ret_was


def plots(JS, was, group):
    """
    Generate PCA and t-SNE plots for JS and Wasserstein distance matrices.

    This function visualizes sample-level distance profiles using dimensionality
    reduction. It is kept for compatibility with earlier analysis scripts.
    The `plot_2d` function provides more explicit output-path control.

    Parameters
    ----------
    JS : list
        Jensen-Shannon distance results.
    was : list
        Wasserstein distance results.
    group : list or array-like
        Sample group labels used for coloring points.
    """
    JS_temp = np.array(JS[0])
    was_temp = np.array(was[0])

    # Replace infinite values before dimensionality reduction.
    JS_temp[np.isinf(JS_temp)] = 1
    was_temp[np.isinf(was_temp)] = 1

    n_samples = 6

    # PCA visualization.
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

    # t-SNE visualization.
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
    """
    Perform KEGG pathway enrichment analysis based on significant MassLinker tokens.

    Significant compounds are selected using p < 0.05. For each KEGG pathway,
    a hypergeometric test is used to evaluate whether significant compounds are
    over-represented. A pathway perturbation score-like value, PPS, is also
    calculated from Wasserstein-distance differences.

    Parameters
    ----------
    pathway_df_full : pandas.DataFrame
        KEGG annotation table containing compound, pathway, and KEGG identifiers.
        Required columns include `compound_names`, `kegg_id`, and `pathway_name`.
    p_values : list or np.ndarray
        Token-level p-values from two-group differential analysis.
    was_diff : list or np.ndarray
        Group-level Wasserstein distance differences for MassLinker tokens.

    Returns
    -------
    pandas.DataFrame
        KEGG enrichment result table.
    """
    significant_rows = pathway_df_full[np.array(p_values) < 0.05]
    significant_compounds_set = set(significant_rows['compound_names'].unique())
    universe_compounds = set(pathway_df_full['compound_names'].unique())
    pathways = pathway_df_full[['kegg_id', 'pathway_name']].drop_duplicates()

    N = len(universe_compounds)
    M = len(significant_compounds_set)

    enrichment_results = []
    desc = "Performing KEGG pathway enrichment analysis"
    j = 0

    for index, row in tqdm(pathways.iterrows(), total=pathways.shape[0], desc=desc):
        pathway_id = row['kegg_id']
        pathway_name = row['pathway_name']
        compounds_in_pathway = set(pathway_df_full[pathway_df_full['kegg_id'] == pathway_id]['compound_names'])

        K = len(compounds_in_pathway)

        # Calculate pathway-level perturbation score using Wasserstein differences.
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
        print("Warning: no significant metabolites were found in any KEGG pathway.")
        return pd.DataFrame()

    results_df = pd.DataFrame(enrichment_results)

    # Apply Benjamini-Hochberg FDR correction.
    reject, q_values, _, _ = multipletests(results_df['P_Value'], alpha=0.05, method='fdr_bh')
    results_df['Q_Value (FDR_BH)'] = q_values

    results_df = results_df.sort_values(by='P_Value', ascending=True)

    output_filename = 'metabolite_enrichment_analysis_results.xlsx'
    results_df.to_excel(output_filename, index=False)

    print(f"\nEnrichment analysis completed. Results saved to: {output_filename}")

    return results_df


def plot_enrichment(plot_df, topn_p=3, topn_pps=3, save_dir=None, save_name="KEGG_Enriched.pdf",
                    low_p_color='#D15D73', high_p_color='#B3B3B3'):
    """
    Plot KEGG enrichment results as a 3D scatter plot.

    The plot uses enrichment ratio, PPS, and -log10(p-value) as the three axes.
    Pathways with the smallest p-values and largest PPS values are annotated.

    Parameters
    ----------
    plot_df : pandas.DataFrame
        KEGG enrichment result table generated by `kegg_enrichment`.
    topn_p : int, optional
        Number of pathways with the smallest p-values to label.
    topn_pps : int, optional
        Number of pathways with the largest PPS values to label.
    save_dir : str, optional
        Output directory. If None, `save_name` is treated as the output path.
    save_name : str, optional
        Output figure filename.
    low_p_color : str, optional
        Color assigned to lower p-value points.
    high_p_color : str, optional
        Color assigned to higher p-value points.
    """
    plot_df = plot_df.copy()
    plot_df.dropna(subset=['P_Value'], inplace=True)
    plot_df['neg_log10_p'] = -np.log10(plot_df['P_Value'].replace(0, np.nextafter(0, 1)))

    fig = plt.figure(figsize=(13, 11))
    ax = fig.add_subplot(111, projection='3d')

    # Build a custom color map where lower p-values are emphasized.
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

    # Annotate pathways with the smallest p-values.
    if len(plot_df) > 0:
        num_to_plot_p = min(len(plot_df), topn_p)
        top_pathways_p = plot_df.nsmallest(num_to_plot_p, 'P_Value')
        for _, row in top_pathways_p.iterrows():
            ax.text(row['Enrichment_Ratio'], row['PPS'], row['neg_log10_p'],
                    f"  {row['Pathway_Name']}",
                    color=annotation_color, fontsize=9, ha='left', va='center')

    # Annotate pathways with the largest PPS values, avoiding duplicate labels.
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
    """
    Perform two-group statistical comparison for distance profiles.

    For each MassLinker token, this function compares the Wasserstein distance
    profiles between two groups using Welch's two-sample t-test. The group-level
    mean difference is also returned for downstream enrichment and visualization.

    Parameters
    ----------
    dis_mat : np.ndarray
        Sample-by-feature distance matrix.
    group : list, np.ndarray, or torch.Tensor
        Binary group labels for the selected samples.

    Returns
    -------
    tuple
        p_values and was_diff. `p_values` contains token-level t-test p-values,
        and `was_diff` contains group-level mean distance differences.
    """
    if isinstance(group, torch.Tensor):
        group = group.cpu().numpy()

    group_0_data = dis_mat[np.array(group) == list(set(group))[0]]
    group_1_data = dis_mat[np.array(group) == list(set(group))[1]]

    num_features = dis_mat.shape[1]
    p_values = []

    for i in tqdm(range(num_features)):
        feature_data_g0 = group_0_data[:, i]
        feature_data_g1 = group_1_data[:, i]

        # Welch's t-test is used because equal variance is not assumed.
        t_statistic, p_value = stats.ttest_ind(feature_data_g0, feature_data_g1, equal_var=False)
        p_values.append(p_value)

    was_diff = sum(group_0_data) / len(group_0_data) - (sum(group_1_data) / len(group_1_data))

    return p_values, was_diff


def plot_peak_comp(p_values, group, params, met_name, mzs, top_n=100, plot_num=180000, save_path="/", window_size=5):
    """
    Plot reconstructed RBF chromatographic curves for top differential features.

    Features are ranked by p-value, and the top features are reconstructed from
    MassLinker RBF parameters for all selected samples. Curves are colored by
    group to visualize group-associated differences in chromatographic signal
    morphology.

    Parameters
    ----------
    p_values : list or np.ndarray
        Token-level p-values from differential analysis.
    group : list or np.ndarray
        Sample group labels.
    params : list
        Sample-level MassLinker token tensors.
    met_name : list
        Metabolite or compound names corresponding to MassLinker tokens.
    mzs : list
        m/z values corresponding to MassLinker tokens.
    top_n : int, optional
        Number of top-ranked features to plot.
    plot_num : int, optional
        Number of retention-time grid points used for reconstruction.
    save_path : str, optional
        Directory used to save output PDF files.
    window_size : int, optional
        Reserved parameter for compatibility with previous versions.
    """
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
    """
    Generate PCA and t-SNE visualizations from JS and Wasserstein distance profiles.

    This function is used to visualize sample-level separation based on
    significant MassLinker distance features. JS and Wasserstein distance
    profiles are reduced to two dimensions independently.

    Parameters
    ----------
    JS : list
        Jensen-Shannon distance results.
    was : list
        Wasserstein distance results.
    group : list or array-like
        Sample group labels used for coloring points.
    save_path : str, optional
        Directory used to save PCA and t-SNE figures.
    color_map : list, optional
        Colors used for sample groups.
    """
    if color_map is None:
        color_map = ["#51AAD1", '#D15D73']

    JS_temp = np.array(JS[0])
    was_temp = np.array(was[0])

    # Replace infinite values before dimensionality reduction.
    JS_temp[np.isinf(JS_temp)] = 1
    was_temp[np.isinf(was_temp)] = 1

    n_samples = 6

    # PCA visualization.
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

    # t-SNE visualization.
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
    """
    Summarize and visualize model performance from N-fold cross-validation results.

    This function reads prediction result files, calculates common classification
    metrics, optionally incorporates additional ROC-derived model outputs, and
    delegates grouped bar plotting to `barplot_foldn_valid`.

    Parameters
    ----------
    data_path : str
        Directory containing cross-validation prediction files in xlsx format.
    save_dir : str
        Directory used to save the output figure.
    model_names : list
        Names of models to be displayed in the final plot.
    save_name : str
        Output figure filename.
    addition_ROCs : list, optional
        Additional model outputs represented as tuples of true labels, scores,
        color information, and model name.
    title : str, optional
        Figure title.
    """
    records = []
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

    for files in os.listdir(data_path):
        if files[-4:] == "xlsx":
            records.append(pd.read_excel(os.path.join(data_path, files), engine='openpyxl'))

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
                recall = recall_score(y_true_1d, y_pred, average='weighted')
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
    """
    Draw grouped bar plots for cross-validation performance metrics.

    The function visualizes accuracy, precision, recall, and F1-score across
    multiple models. Mean values and standard deviations across folds are shown.

    Parameters
    ----------
    dataframes : list
        Per-model, per-fold metric dictionaries.
    model_names : list
        Model names displayed on the x-axis.
    save_dir : str
        Directory used to save the output figure.
    save_name : str
        Output figure filename.
    addition_df : list, optional
        Additional metric dictionaries from externally provided model outputs.
    colors : list, optional
        Bar colors for different metrics.
    title : str, optional
        Figure title.
    """
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
    """
    Plot MetTD-style feature scores against permutation-derived p-values.

    The plot summarizes feature-level importance or perturbation scores and
    highlights top-ranked features according to adjusted p-values and scores.

    Parameters
    ----------
    e_p_values : np.ndarray
        Feature-level p-values.
    e_scores : np.ndarray
        Feature-level scores.
    e_names : list
        Feature names used for annotation.
    permutation_time : int
        Number of permutations used to estimate p-values.
    save_path : str, optional
        Output PDF path.
    top_n : int, optional
        Number of top features to annotate.
    fig_size : tuple, optional
        Figure size.
    """
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


def low_dim_plots(plot1, groups, figsize=None, random_state=42, save_path="2D_plot.pdf", dim=2, group_names=None, colors=None):
    """
    Generate PCA, t-SNE, and UMAP visualizations for high-dimensional features.

    This function standardizes the input feature matrix, computes low-dimensional
    embeddings using three common dimensionality-reduction methods, and plots
    the results with user-provided group labels.

    Parameters
    ----------
    plot1 : array-like
        Input feature matrix.
    groups : list or array-like
        Group labels. Multiple group-label arrays can also be provided.
    figsize : tuple, optional
        Figure size.
    random_state : int, optional
        Random seed for reproducible dimensionality reduction.
    save_path : str, optional
        Output PDF path.
    dim : int, optional
        Number of dimensions to compute.
    group_names : list, optional
        Names for multiple group-label rows.
    colors : list, optional
        Colors used for group labels.

    Returns
    -------
    dict
        Low-dimensional coordinates from PCA, t-SNE, and UMAP.
    """
    if isinstance(plot1, list):
        X = np.array(plot1)
    else:
        X = np.array(plot1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Normalize group input into a list of group arrays.
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

    # PCA embedding.
    pca = PCA(n_components=dim, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    results['pca'] = X_pca

    # t-SNE embedding, preceded by PCA compression for high-dimensional data.
    pca_for_tsne = PCA(n_components=min(500, len(X_scaled[0])), random_state=random_state)
    X_encoded_tsne = pca_for_tsne.fit_transform(X_scaled)
    tsne = TSNE(n_components=min(dim, 3), random_state=random_state, perplexity=50, n_iter=2000)
    X_tsne = tsne.fit_transform(X_encoded_tsne)
    results['tsne'] = X_tsne

    # UMAP embedding.
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
    """
    Plot the top-ranked significant MassLinker features by p-value.

    If m/z values are provided, features are grouped by m/z and the most
    significant p-value for each unique m/z value is used. Otherwise, feature
    names are plotted directly.

    Parameters
    ----------
    p_values : list or np.ndarray
        Feature-level p-values.
    met_names : list
        Metabolite or feature names.
    mzs : list, optional
        m/z values corresponding to features.
    n : int, optional
        Number of top-ranked features to display.
    figsize : tuple, optional
        Figure size.
    color_palette : str, optional
        Reserved plotting parameter for compatibility.
    add_significance_lines : bool, optional
        Whether to add p-value threshold reference lines.
    save_path : str, optional
        Output PDF path.

    Returns
    -------
    pandas.DataFrame
        Data frame containing the plotted top-ranked features.
    """
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

        # Keep the most significant result for each unique m/z value.
        df_unique_mz = df_sorted.loc[df_sorted.groupby('mz')['p_value'].idxmin()]
        df_unique_mz = df_unique_mz.sort_values('p_value').head(n)
        df_unique_mz['-log10_p'] = -np.log10(df_unique_mz['p_value'])

        plt.figure(figsize=figsize)
        colors = '#C2DDF2'

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
