# Utility functions
This document provides an overview of the utility functions implemented in `utils.py`.

The utility module contains helper functions for MassLinker token comparison, RBF curve reconstruction, metabolite-level distance analysis, statistical testing, KEGG enrichment analysis, dimensionality-reduction visualization, model-performance visualization, and feature-importance plotting.

These functions are mainly designed for downstream analysis after MassLinker metabolic tokens have been generated and converted into Python datasets.

---

## Overview

The `utils.py` module provides tools for the following tasks:

1. Reconstructing chromatographic signals from RBF token parameters.
2. Measuring distances between metabolite tokens across samples.
3. Performing metabolite-level statistical comparison.
4. Visualizing sample separation using PCA, t-SNE, and UMAP.
5. Plotting RBF peak curves for selected metabolites.
6. Performing KEGG pathway enrichment analysis.
7. Visualizing enrichment results.
8. Summarizing cross-validation model performance.
9. Visualizing feature scores and p-values.

---

## Main function groups

| Function group | Functions | Purpose |
|---|---|---|
| RBF reconstruction | `RBF` | Reconstruct a chromatographic signal from RBF parameters |
| Distribution distance | `calculate_js_divergence`, `calculate_wasserstein_distance` | Compare two reconstructed RBF curves |
| Metabolite distance analysis | `metabo_dis` | Compute JS divergence and Wasserstein distance between samples |
| Distance visualization | `plots`, `plot_2d` | Visualize sample-level distance matrices using PCA and t-SNE |
| Statistical comparison | `get_diff` | Compare metabolite-level distance features between two groups |
| Peak comparison | `plot_peak_comp` | Plot reconstructed RBF curves for top differential metabolites |
| KEGG enrichment | `kegg_enrichment` | Perform pathway enrichment based on significant metabolites |
| Enrichment visualization | `plot_enrichment` | Generate 3D KEGG enrichment plots |
| Model evaluation plots | `visualization_foldn_valid`, `barplot_foldn_valid` | Summarize model performance across validation folds |
| Token visualization | `low_dim_plots` | Visualize flattened token features using PCA, t-SNE, and UMAP |
| Feature significance plots | `plot_MetTD`, `plot_p_value` | Visualize feature scores, p-values, and top-ranked metabolites |

---

## RBF signal reconstruction

### `RBF`

The `RBF()` function reconstructs a chromatographic-like signal from one MassLinker token.

Each token contains three groups of RBF parameters:

| Row | Parameter | Description |
|---|---|---|
| `0` | weights | Amplitude-related RBF parameters |
| `1` | centers | Retention-time center parameters |
| `2` | widths / sigmas | RBF width parameters |

The function evaluates the RBF mixture over a given retention-time axis and returns the reconstructed signal intensity.

Conceptually, it converts a compact MassLinker token:

```text
3 × 20
```

into a reconstructed chromatographic curve.

This function is used internally by the distance and peak-comparison utilities.

---

## Distribution distance calculation

### `calculate_js_divergence`

This function calculates the Jensen-Shannon divergence between two reconstructed RBF curves.

The workflow is:

1. Reconstruct two curves from two RBF parameter sets.
2. Remove negative signal values.
3. Normalize both curves as probability distributions.
4. Compute Jensen-Shannon divergence.

The Jensen-Shannon divergence is useful for comparing the shape difference between two metabolite-level chromatographic profiles.

---

### `calculate_wasserstein_distance`

This function calculates the Wasserstein distance between two reconstructed RBF curves.

The workflow is:

1. Reconstruct two curves over the retention-time axis.
2. Remove negative signal values.
3. Normalize both curves as distributions.
4. Compute the Wasserstein distance along retention time.

The Wasserstein distance is useful for measuring retention-time shifts or shape displacement between two reconstructed metabolite signals.

---

## Metabolite-level distance analysis

### `metabo_dis`

The `metabo_dis()` function computes metabolite-level distances between samples.

It compares each sample against a reference sample and calculates two distance metrics for every metabolite token:

1. Jensen-Shannon divergence.
2. Wasserstein distance.

The returned results are:

| Output | Description |
|---|---|
| `ret_JS` | JS divergence values for metabolite tokens |
| `ret_was` | Wasserstein distance values for metabolite tokens |

This function can be used to quantify how much each sample differs from a reference sample at the metabolite-token level.

---

## Distance-based visualization

### `plots`

The `plots()` function performs PCA and t-SNE visualization based on precomputed JS divergence and Wasserstein distance matrices.

It generates two PDF files:

| Output file | Description |
|---|---|
| `pca.pdf` | PCA visualization of JS and Wasserstein distance features |
| `tsne.pdf` | t-SNE visualization of JS and Wasserstein distance features |

This function provides a quick overview of whether samples separate in the distance-feature space.

---

### `plot_2d`

The `plot_2d()` function is a more configurable version for two-dimensional visualization of distance matrices.

It supports:

- PCA visualization;
- t-SNE visualization;
- custom save directory;
- custom color map;
- group-based coloring.

Generated files include:

```text
pca.pdf
tsne.pdf
```

---

## Statistical comparison

### `get_diff`

The `get_diff()` function compares distance-derived metabolite features between two groups.

For each feature, it performs a Welch's t-test between two groups and returns:

| Output | Description |
|---|---|
| `p_values` | p-values for all metabolite-level features |
| `was_diff` | Difference in mean feature values between the two groups |

This function is mainly used to identify differential metabolite tokens between two biological or clinical groups.

---

## Peak comparison plots

### `plot_peak_comp`

The `plot_peak_comp()` function visualizes reconstructed RBF curves for top-ranked differential metabolites.

It selects metabolites with the smallest p-values and plots their reconstructed RBF curves across samples.

Each output plot includes:

- reconstructed RBF curves;
- group-specific coloring;
- metabolite name;
- p-value;
- m/z value.

This function is useful for visually inspecting whether selected metabolite tokens show group-specific curve differences.

---

## KEGG pathway enrichment

### `kegg_enrichment`

The `kegg_enrichment()` function performs KEGG pathway enrichment analysis based on significant metabolite-level features.

The function identifies significant compounds according to p-values and evaluates whether they are enriched in KEGG pathways using a hypergeometric test.

The output table includes:

| Column | Description |
|---|---|
| `Pathway_ID` | KEGG pathway identifier |
| `Pathway_Name` | KEGG pathway name |
| `P_Value` | Enrichment p-value |
| `Enrichment_Ratio` | Relative enrichment strength |
| `Hits_in_Pathway (k)` | Number of significant compounds in the pathway |
| `Total_in_Pathway (K)` | Number of compounds in the pathway |
| `Hit_Compounds` | Significant compounds found in the pathway |
| `PPS` | Pathway-level score weighted by distance difference |
| `Q_Value (FDR_BH)` | FDR-adjusted q-value |

The function saves the enrichment result as:

```text
metabolite_enrichment_analysis_results.xlsx
```

---

## KEGG enrichment visualization

### `plot_enrichment`

The `plot_enrichment()` function generates a three-dimensional enrichment plot.

The plot visualizes enriched pathways using:

| Axis / Visual element | Meaning |
|---|---|
| x-axis | Enrichment ratio |
| y-axis | PPS score |
| z-axis | Negative log10 p-value |
| color | Enrichment p-value |
| text labels | Top pathways ranked by p-value and PPS |

This function is used to summarize enriched KEGG pathways in a publication-style figure.

---

## Dimensionality-reduction visualization

### `low_dim_plots`

The `low_dim_plots()` function visualizes high-dimensional sample features using three dimensionality-reduction methods:

1. PCA.
2. t-SNE.
3. UMAP.

Before dimensionality reduction, the input feature matrix is standardized using `StandardScaler`.

The function supports:

- one or multiple grouping vectors;
- two-dimensional or three-dimensional embeddings;
- custom colors;
- custom group names;
- PDF output.

The returned object contains the calculated coordinates:

| Key | Description |
|---|---|
| `pca` | PCA coordinates |
| `tsne` | t-SNE coordinates |
| `umap` | UMAP coordinates |

This function is commonly used to visualize flattened MassLinker token features or model-derived embeddings.

---

## Model performance visualization

### `visualization_foldn_valid`

The `visualization_foldn_valid()` function summarizes model predictions from multiple validation folds.

It reads `.xlsx` result files from a directory and calculates common classification metrics:

| Metric | Description |
|---|---|
| Accuracy | Overall prediction accuracy |
| Precision | Weighted precision |
| Recall | Weighted recall |
| F1-score | Weighted F1-score |

The function can also incorporate additional ROC-related prediction results through the `addition_ROCs` argument.

Internally, it calls `barplot_foldn_valid()` to generate a summary bar plot.

---

### `barplot_foldn_valid`

The `barplot_foldn_valid()` function generates a grouped bar plot for cross-validation performance.

It visualizes:

- mean accuracy;
- mean precision;
- mean recall;
- mean F1-score;
- standard deviation across folds.

This function is useful for comparing multiple models under the same validation setting.

---

## Feature-score and p-value visualization

### `plot_MetTD`

The `plot_MetTD()` function visualizes feature scores against statistical significance.

It creates a scatter plot using:

| Visual element | Meaning |
|---|---|
| x-axis | Feature score |
| y-axis | Negative log10 p-value |
| color | Adjusted p-value |
| point size | Log-scaled absolute score |
| labels | Top-ranked features |

This function is suitable for visualizing permutation-based or score-based metabolite discovery results.

---

### `plot_p_value`

The `plot_p_value()` function visualizes the top-ranked significant features according to p-values.

It can work in two modes:

1. Feature-name mode.
2. m/z-aware mode.

If m/z values are provided, the function keeps the most significant feature for each unique m/z value and then plots the top-ranked m/z values.

The output is a horizontal bar plot of:

```text
-log10(p-value)
```

This function is useful for reporting the most statistically significant metabolites or m/z features.

---

## Dependencies

The `utils.py` module uses packages for numerical computing, statistics, machine learning, and visualization.

| Package | Purpose |
|---|---|
| `numpy` | Numerical operations |
| `torch` | Tensor-based RBF parameters |
| `scipy` | Distance metrics and statistical tests |
| `pandas` | Reading and writing tabular data |
| `matplotlib` | Figure generation |
| `seaborn` | Statistical plotting |
| `scikit-learn` | PCA, t-SNE, metrics, and scaling |
| `umap-learn` | UMAP dimensionality reduction |
| `statsmodels` | Multiple-testing correction |
| `tqdm` | Progress bars |
| `joblib` | Loading and saving processed Python objects |
| `adjustText` | Optional text adjustment for plot annotations |

A typical installation command is:

```bash
pip install numpy torch scipy pandas matplotlib seaborn scikit-learn umap-learn statsmodels tqdm joblib adjustText openpyxl
```

---

## Notes

1. Most functions in `utils.py` are designed for downstream analysis after MassLinker token generation.
2. RBF-related functions assume that each token follows the MassLinker parameter format
3. Distance-based functions reconstruct RBF curves before calculating distribution distances.
4. KEGG enrichment functions require pathway and compound annotation tables.
5. Visualization functions save figures as PDF files by default.
6. Some functions are optimized for binary-group comparison.
7. Several functions are intended as analysis utilities rather than general-purpose library APIs.
8. Detailed case studies and usage examples are provided in separate tutorials.
