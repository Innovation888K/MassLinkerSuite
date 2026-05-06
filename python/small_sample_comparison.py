from utils import (
    metabo_dis,
    get_diff,
    kegg_enrichment,
    plot_enrichment,
    plot_peak_comp,
    plot_2d,
    plot_p_value
)

import os
import joblib
import numpy as np
import pandas as pd


# Define project directories
data_dir = "data"
metadata_dir = "metadata"
cache_dir = "cache"
results_dir = "results"

peak_plot_dir = os.path.join(results_dir, "differential_peaks")
distance_plot_dir = os.path.join(results_dir, "distance_visualization")

os.makedirs(cache_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(peak_plot_dir, exist_ok=True)
os.makedirs(distance_plot_dir, exist_ok=True)


# Define input and output files
dataset_path = os.path.join(data_dir, "processed_dataset.joblib")
annotation_path = os.path.join(metadata_dir, "pathway_compound_detail.csv")

JS_cache_path = os.path.join(cache_dir, "JS_distance.joblib")
Wasserstein_cache_path = os.path.join(cache_dir, "Wasserstein_distance.joblib")


# Load a processed MassLinker dataset.
# The loaded object should be an ExcelDataset-like object.
dataset = joblib.load(dataset_path)


# Select samples for analysis.
# Here, all samples in the dataset are selected.
sele_sample = [
    i
    for i in range(len(dataset))
]

# For testing or debugging, selected sample indices can be used instead.
# sele_sample = [0, 1]


# Extract group labels.
# In this example, dataset.is_positive is used as the binary group indicator.
group = [
    dataset.is_positive[i].numpy().item()
    for i in sele_sample
]


# Extract MassLinker token parameters.
params = [
    dataset.samples[i]
    for i in sele_sample
]


# Calculate or load JS divergence and Wasserstein distance.
# This step can be time-consuming, so cached results are reused if available.
if os.path.exists(JS_cache_path) and os.path.exists(Wasserstein_cache_path):
    JS = joblib.load(JS_cache_path)
    was = joblib.load(Wasserstein_cache_path)
else:
    JS, was = metabo_dis(params)
    joblib.dump(JS, filename=JS_cache_path)
    joblib.dump(was, filename=Wasserstein_cache_path)


# Convert distance results to arrays.
was_arr = np.array(was)[0]
JS_arr = np.array(JS)[0]


# Perform two-group statistical comparison.
p_values, was_diff = get_diff(
    was_arr,
    group
)


# Load compound, m/z, and pathway annotation information.
compound_df = pd.read_csv(annotation_path)

met_name = compound_df["compound_names"].tolist()
mzs = compound_df["mz"].tolist()


# Plot top differential m/z features ranked by p-value.
plot_p_value(
    p_values,
    met_name,
    mzs=mzs,
    n=20,
    save_path=os.path.join(results_dir, "p_value_rank.pdf")
)


# Plot reconstructed RBF curves for top differential peaks.
plot_peak_comp(
    p_values,
    group,
    params,
    met_name,
    mzs,
    save_path=peak_plot_dir,
    top_n=2000
)


# Perform KEGG enrichment analysis.
pathway_df_full = pd.read_csv(
    annotation_path,
    index_col=0
)

kegg_enriched = kegg_enrichment(
    pathway_df_full,
    p_values,
    was_diff
)


# Plot KEGG enrichment result.
plot_enrichment(
    kegg_enriched.copy(),
    save_dir=results_dir,
    save_name="KEGG_enrichment.pdf"
)


# Select significant distance features for 2D visualization.
was_selected = [
    []
    for i in range(len(was[0]))
]

JS_selected = [
    []
    for i in range(len(JS[0]))
]

for i in range(len(p_values)):
    if p_values[i] <= 0.01:
        for j in range(len(was[0])):
            was_selected[j].append(was[0][j][i])
            JS_selected[j].append(JS[0][j][i])


# Visualize samples using significant JS and Wasserstein distance features.
plot_2d(
    [JS_selected],
    [was_selected],
    group,
    save_path=distance_plot_dir
)
