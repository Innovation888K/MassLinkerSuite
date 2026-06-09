import argparse
import os
import joblib
import numpy as np
import pandas as pd

from utils import (
    metabo_dis,
    get_diff,
    kegg_enrichment,
    plot_enrichment,
    plot_peak_comp,
    plot_2d,
    plot_p_value,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run MassLinker downstream analysis, including token-distance calculation, "
            "two-group comparison, peak visualization, KEGG enrichment, and 2D visualization."
        )
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./data/processed_dataset.joblib",
        help="Path to the processed MassLinker ExcelDataset joblib file.",
    )

    parser.add_argument(
        "--annotation_path",
        type=str,
        default="./metadata/pathway_compound_detail.csv",
        help=(
            "Path to the compound/pathway annotation CSV file. "
            "The file should contain compound_names and mz columns."
        ),
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./cache",
        help="Directory used to save or load cached distance results.",
    )

    parser.add_argument(
        "--results_dir",
        type=str,
        default="./results",
        help="Directory used to save analysis results and figures.",
    )

    parser.add_argument(
        "--js_cache_name",
        type=str,
        default="JS_distance.joblib",
        help="Filename for cached Jensen-Shannon distance results.",
    )

    parser.add_argument(
        "--wasserstein_cache_name",
        type=str,
        default="Wasserstein_distance.joblib",
        help="Filename for cached Wasserstein distance results.",
    )

    parser.add_argument(
        "--p_value_cutoff",
        type=float,
        default=0.01,
        help="P-value cutoff used to select significant distance features for 2D visualization.",
    )

    parser.add_argument(
        "--top_p_value_n",
        type=int,
        default=20,
        help="Number of top differential features shown in the p-value ranking plot.",
    )

    parser.add_argument(
        "--top_peak_n",
        type=int,
        default=2000,
        help="Number of top differential features used for reconstructed peak comparison plots.",
    )

    parser.add_argument(
        "--sample_indices",
        type=str,
        default="all",
        help=(
            "Comma-separated sample indices used for analysis, e.g. '0,1,2'. "
            "Use 'all' to select all samples."
        ),
    )

    parser.add_argument(
        "--force_recompute_distance",
        action="store_true",
        help="If set, recompute JS and Wasserstein distances even if cache files exist.",
    )

    return parser.parse_args()


def parse_sample_indices(sample_indices, dataset_length):
    if sample_indices == "all":
        return list(range(dataset_length))

    indices = [
        int(x.strip())
        for x in sample_indices.split(",")
        if x.strip() != ""
    ]

    for idx in indices:
        if idx < 0 or idx >= dataset_length:
            raise ValueError(
                f"Sample index {idx} is out of range. "
                f"Dataset contains {dataset_length} samples."
            )

    return indices


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Define output directories
    # ------------------------------------------------------------------
    peak_plot_dir = os.path.join(args.results_dir, "differential_peaks")
    distance_plot_dir = os.path.join(args.results_dir, "distance_visualization")

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(peak_plot_dir, exist_ok=True)
    os.makedirs(distance_plot_dir, exist_ok=True)

    js_cache_path = os.path.join(args.cache_dir, args.js_cache_name)
    wasserstein_cache_path = os.path.join(args.cache_dir, args.wasserstein_cache_name)

    print("============================================================")
    print("MassLinker downstream analysis")
    print("============================================================")
    print(f"Dataset path:        {args.dataset_path}")
    print(f"Annotation path:     {args.annotation_path}")
    print(f"Cache directory:     {args.cache_dir}")
    print(f"Results directory:   {args.results_dir}")
    print(f"P-value cutoff:      {args.p_value_cutoff}")
    print(f"Top p-value n:       {args.top_p_value_n}")
    print(f"Top peak n:          {args.top_peak_n}")
    print("============================================================")

    # ------------------------------------------------------------------
    # Load processed MassLinker dataset
    # ------------------------------------------------------------------
    print("Loading processed MassLinker dataset...")
    dataset = joblib.load(args.dataset_path)

    sele_sample = parse_sample_indices(
        sample_indices=args.sample_indices,
        dataset_length=len(dataset),
    )

    print(f"Selected samples: {len(sele_sample)}")

    # ------------------------------------------------------------------
    # Extract group labels and MassLinker token parameters
    # ------------------------------------------------------------------
    # dataset.is_positive is used as the binary group indicator.
    group = [
        dataset.is_positive[i].numpy().item()
        for i in sele_sample
    ]

    params = [
        dataset.samples[i]
        for i in sele_sample
    ]

    unique_groups = sorted(set(group))
    print(f"Detected groups: {unique_groups}")

    if len(unique_groups) != 2:
        raise ValueError(
            "This quick-start pipeline expects a two-group comparison based on "
            "dataset.is_positive. Please make sure the target file contains "
            "a binary is_positive column."
        )

    # ------------------------------------------------------------------
    # Calculate or load JS divergence and Wasserstein distance
    # ------------------------------------------------------------------
    cache_exists = (
        os.path.exists(js_cache_path)
        and os.path.exists(wasserstein_cache_path)
    )

    if cache_exists and not args.force_recompute_distance:
        print("Loading cached JS and Wasserstein distances...")
        JS = joblib.load(js_cache_path)
        was = joblib.load(wasserstein_cache_path)
    else:
        print("Calculating JS and Wasserstein distances...")
        JS, was = metabo_dis(params)

        joblib.dump(JS, filename=js_cache_path)
        joblib.dump(was, filename=wasserstein_cache_path)

        print(f"JS distance cache saved to: {js_cache_path}")
        print(f"Wasserstein distance cache saved to: {wasserstein_cache_path}")

    # ------------------------------------------------------------------
    # Convert distance results to arrays
    # ------------------------------------------------------------------
    was_arr = np.array(was)[0]
    JS_arr = np.array(JS)[0]

    print(f"Wasserstein distance array shape: {was_arr.shape}")
    print(f"JS distance array shape: {JS_arr.shape}")

    # ------------------------------------------------------------------
    # Perform two-group statistical comparison
    # ------------------------------------------------------------------
    print("Performing two-group statistical comparison...")
    p_values, was_diff = get_diff(
        was_arr,
        group,
    )

    p_value_output = os.path.join(args.results_dir, "p_values.csv")
    pd.DataFrame(
        {
            "p_value": p_values,
            "wasserstein_difference": was_diff,
        }
    ).to_csv(p_value_output, index=False)

    print(f"P-values saved to: {p_value_output}")

    # ------------------------------------------------------------------
    # Load compound, m/z, and pathway annotation information
    # ------------------------------------------------------------------
    print("Loading annotation file...")
    compound_df = pd.read_csv(args.annotation_path)

    required_columns = {"compound_names", "mz"}
    missing_columns = required_columns - set(compound_df.columns)

    if missing_columns:
        raise ValueError(
            f"Annotation file is missing required columns: {missing_columns}"
        )

    met_name = compound_df["compound_names"].tolist()
    mzs = compound_df["mz"].tolist()

    # ------------------------------------------------------------------
    # Plot top differential m/z features ranked by p-value
    # ------------------------------------------------------------------
    print("Plotting p-value ranking...")
    plot_p_value(
        p_values,
        met_name,
        mzs=mzs,
        n=args.top_p_value_n,
        save_path=os.path.join(args.results_dir, "p_value_rank.pdf"),
    )

    # ------------------------------------------------------------------
    # Plot reconstructed RBF curves for top differential peaks
    # ------------------------------------------------------------------
    print("Plotting reconstructed RBF curves for top differential peaks...")
    plot_peak_comp(
        p_values,
        group,
        params,
        met_name,
        mzs,
        save_path=peak_plot_dir,
        top_n=args.top_peak_n,
    )

    # ------------------------------------------------------------------
    # Perform KEGG enrichment analysis
    # ------------------------------------------------------------------
    print("Performing KEGG enrichment analysis...")
    pathway_df_full = pd.read_csv(
        args.annotation_path,
        index_col=0,
    )

    kegg_enriched = kegg_enrichment(
        pathway_df_full,
        p_values,
        was_diff,
    )

    enrichment_output = os.path.join(args.results_dir, "KEGG_enrichment.csv")
    kegg_enriched.to_csv(enrichment_output, index=False)

    print(f"KEGG enrichment table saved to: {enrichment_output}")

    # ------------------------------------------------------------------
    # Plot KEGG enrichment result
    # ------------------------------------------------------------------
    print("Plotting KEGG enrichment result...")
    plot_enrichment(
        kegg_enriched.copy(),
        save_dir=args.results_dir,
        save_name="KEGG_enrichment.pdf",
    )

    # ------------------------------------------------------------------
    # Select significant distance features for 2D visualization
    # ------------------------------------------------------------------
    print("Selecting significant features for 2D visualization...")

    was_selected = [
        []
        for _ in range(len(was[0]))
    ]

    JS_selected = [
        []
        for _ in range(len(JS[0]))
    ]

    for i in range(len(p_values)):
        if p_values[i] <= args.p_value_cutoff:
            for j in range(len(was[0])):
                was_selected[j].append(was[0][j][i])
                JS_selected[j].append(JS[0][j][i])

    selected_feature_count = sum(
        1
        for p in p_values
        if p <= args.p_value_cutoff
    )

    print(
        f"Number of selected features with p <= {args.p_value_cutoff}: "
        f"{selected_feature_count}"
    )

    if selected_feature_count == 0:
        print(
            "No significant features were selected for 2D visualization. "
            "Skipping plot_2d."
        )
    else:
        print("Plotting 2D distance visualization...")
        plot_2d(
            [JS_selected],
            [was_selected],
            group,
            save_path=distance_plot_dir,
        )

    print("============================================================")
    print("MassLinker downstream analysis completed.")
    print(f"Results saved in: {args.results_dir}")
    print("============================================================")


if __name__ == "__main__":
    main()
