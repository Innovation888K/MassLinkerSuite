# Transformer SHAP analysis for MassLinker token classifiers

This example demonstrates how to perform SHAP-based interpretation for a trained Transformer classifier using MassLinker token tensors.

The goal of this workflow is to identify important MassLinker token positions, visualize class-specific SHAP contributions, and summarize which metabolite-related sequence steps contribute most strongly to Transformer predictions.

This example is intended as an advanced downstream interpretation workflow. It is not required for the basic MassLinker pipeline.

---

## Overview

The `transformer_shap.py` script provides utilities for interpreting a trained Transformer classifier.

The workflow includes:

1. Loading a trained Transformer model checkpoint.
2. Loading a MassLinker dataset object.
3. Preparing token tensors for Transformer input.
4. Running SHAP `DeepExplainer`.
5. Saving SHAP results for later reuse.
6. Visualizing top important token positions.
7. Generating class-specific SHAP heatmaps.
8. Plotting aggregated token importance across classes.

This workflow is useful when you want to understand which MassLinker tokens, metabolites, or pathway-associated features contribute most strongly to Transformer-based classification.

---

## Recommended project structure

A recommended project structure is:

```text
transformer_shap_project/
├── data/
│   └── dataset_for_shap.joblib
├── models/
│   └── best_model.pth
├── metadata/
│   └── pathway_compound_detail.csv
├── results/
│   └── transformer_shap/
│       ├── shap_values.joblib
│       ├── shap_bar_top_tokens.pdf
│       ├── shap_aggregated_top_tokens.pdf
│       ├── shap_heatmap_class_0.pdf
│       ├── shap_heatmap_class_1.pdf
│       └── ...
└── scripts/
    └── run_transformer_shap_analysis.py
```

The role of each file or directory is summarized below.

| Path | Description |
|---|---|
| `data/dataset_for_shap.joblib` | MassLinker dataset used for SHAP analysis |
| `models/best_model.pth` | Trained Transformer model checkpoint |
| `metadata/pathway_compound_detail.csv` | Compound or token annotation table |
| `results/transformer_shap/` | Output directory for SHAP values and figures |
| `scripts/run_transformer_shap_analysis.py` | Example script for running SHAP analysis |

---

## Required input

This example requires three main inputs.

| Input | Description |
|---|---|
| Trained Transformer checkpoint | A `.pth` file saved after Transformer classifier training |
| MassLinker dataset object | A `.joblib` file containing an `ExcelDataset`-like object |
| Feature annotation table | A CSV file containing token or compound names |

A typical setup is:

```text
models/best_model.pth
data/dataset_for_shap.joblib
metadata/pathway_compound_detail.csv
```

The dataset object should contain:

| Attribute | Description |
|---|---|
| `dataset.samples` | MassLinker token tensors |
| `dataset.is_positive` | Binary class labels, if available |
| `dataset.classes` | Multi-class labels, if available |
| `dataset.name` | Sample names, if available |

---

## Input tensor format

The Transformer classifier expects MassLinker token tensors in the shape:

```text
n_samples × n_tokens × 60
```

The value `60` usually corresponds to flattened RBF parameters:

```text
3 parameter groups × 20 RBF components
```

The three parameter groups are:

| Group | Meaning |
|---|---|
| Height | RBF weights or peak heights |
| Position | RBF centers or retention-time positions |
| Width | RBF widths or sigmas |

If the dataset stores samples in another compatible shape, they should be reshaped before being passed to the Transformer.

A common transformation is:

```python
x_tensor = dataset.samples
x_tensor = torch.reshape(
    x_tensor,
    (x_tensor.shape[0], x_tensor.shape[1], 60)
)
```

---

## Important implementation note

The Transformer model architecture used during SHAP analysis must match the architecture used during training.

The following parameters should be consistent:

| Parameter | Meaning |
|---|---|
| `seq_len` | Number of MassLinker tokens |
| `class_count` | Number of output classes |
| `embed_dim` | Transformer embedding dimension |
| `n_heads` | Number of attention heads |
| `depth` | Number of Transformer encoder layers |

For example, if the classifier was trained with:

```python
model = transformer_language(
    seq_len=seq_len,
    class_count=class_count,
    embed_dim=512,
    n_heads=8,
    depth=8
)
```

then the same configuration should be used when loading the checkpoint for SHAP analysis.

---

## Main functions in `transformer_shap.py`

The `transformer_shap.py` script contains several helper functions for SHAP calculation and visualization.

| Function | Purpose |
|---|---|
| `run_shap_analysis()` | Run SHAP `DeepExplainer` on selected test samples |
| `plot_top_n_feature_shap_2d()` | Plot class-specific SHAP heatmaps for top token positions |
| `plot_time_step_bar_chart()` | Plot top token positions using grouped or stacked bar charts |
| `plot_aggregated_time_importance()` | Plot aggregated SHAP importance across selected token positions |

---

## SHAP result format

Depending on the SHAP version and model output format, SHAP values may be returned as either:

1. A list of class-specific arrays;
2. A class-aware NumPy array.

For visualization, the recommended plotting format is:

```text
(n_test × seq_len) × 60 × n_classes
```

This format treats each MassLinker token position as one visualization unit, with 60 RBF-derived features and class-specific SHAP values.

The example script below includes a helper function to convert common SHAP outputs into this plotting format.

---

## Example script

For a reusable public example, avoid hard-coding the sequence length. Instead, infer it from the dataset.

A robust implementation should follow this pattern:

```python
def run_shap_analysis(
    model_path,
    data_path,
    device,
    n_background=50,
    n_test=3,
    embed_dim=512,
    n_heads=8,
    depth=8
):
    dataset = joblib.load(data_path)

    x_raw = dataset.samples

    if isinstance(x_raw, torch.Tensor):
        x_tensor = x_raw.float()
    else:
        x_tensor = torch.tensor(
            x_raw,
            dtype=torch.float32
        )

    x_tensor = x_tensor.reshape(
        x_tensor.shape[0],
        x_tensor.shape[1],
        60
    )

    seq_len = x_tensor.shape[1]

    total_idx = np.arange(
        len(x_tensor)
    )

    bg_idx = np.random.choice(
        total_idx,
        n_background,
        replace=False
    )

    test_idx = np.random.choice(
        total_idx,
        n_test,
        replace=False
    )

    background = x_tensor[
        bg_idx
    ].to(device)

    test_samples = x_tensor[
        test_idx
    ]

    if hasattr(dataset, "classes"):
        num_classes = len(
            set(dataset.classes)
        )
    else:
        y = getattr(
            dataset,
            "is_positive",
            None
        )

        if y is not None:
            num_classes = len(
                np.unique(y)
            )
        else:
            num_classes = 2

    model = transformer_language(
        seq_len=seq_len,
        class_count=num_classes,
        embed_dim=embed_dim,
        n_heads=n_heads,
        depth=depth
    )

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint
    )

    model.to(device)
    model.eval()

    explainer = shap.DeepExplainer(
        model,
        background
    )

    all_shap_values = []

    for i in tqdm(
        range(len(test_samples)),
        desc=f"Calculating SHAP for {n_test} test samples"
    ):
        single_test_sample = test_samples[
            i
        ].unsqueeze(0).to(device)

        shap_values_for_sample = explainer.shap_values(
            single_test_sample
        )

        all_shap_values.append(
            shap_values_for_sample
        )

    return all_shap_values, test_samples.cpu().numpy(), num_classes
```

This avoids writing dataset-specific sequence lengths into the public code.

---

## Running SHAP analysis

After preparing the project files, run:

```bash
python scripts/run_transformer_shap_analysis.py
```

The script will:

1. Load the trained Transformer checkpoint.
2. Load the MassLinker dataset.
3. Randomly select background samples for SHAP.
4. Randomly select test samples for interpretation.
5. Calculate SHAP values.
6. Save SHAP values to a `.joblib` file.
7. Generate class-specific SHAP heatmaps.
8. Generate top-token importance plots.

---

## Output files

The workflow may generate the following outputs.

| Output | Description |
|---|---|
| `results/transformer_shap/shap_values.joblib` | Saved SHAP values, test samples, and metadata |
| `results/transformer_shap/shap_heatmap_class_0.pdf` | SHAP heatmap for class 0 |
| `results/transformer_shap/shap_heatmap_class_1.pdf` | SHAP heatmap for class 1 |
| `results/transformer_shap/shap_bar_top_tokens.pdf` | Bar plot of top important token positions |
| `results/transformer_shap/shap_aggregated_top_tokens.pdf` | Aggregated SHAP importance plot |

For multi-class models, additional heatmaps will be generated:

```text
shap_heatmap_class_2.pdf
shap_heatmap_class_3.pdf
...
```

---

## Explanation of `run_shap_analysis()`

The function:

```python
run_shap_analysis(...)
```

performs the actual SHAP calculation.

Its main steps are:

```text
load dataset
    ↓
reshape MassLinker tokens
    ↓
select background samples
    ↓
select test samples
    ↓
load Transformer checkpoint
    ↓
create SHAP DeepExplainer
    ↓
calculate SHAP values
    ↓
return SHAP values and test samples
```

The background samples are used by SHAP to approximate the reference distribution.

The test samples are the samples being explained.

---

## Explanation of `plot_top_n_feature_shap_2d()`

The function:

```python
plot_top_n_feature_shap_2d(...)
```

generates class-specific SHAP heatmaps.

It selects the most important MassLinker token positions based on SHAP magnitude and visualizes the 60 RBF-derived parameters for each selected token.

The 60 features are divided into three groups:

| Feature group | Number of parameters |
|---|---:|
| Height parameters | 20 |
| Position parameters | 20 |
| Width parameters | 20 |

The heatmap shows whether each parameter contributes positively or negatively to the selected class.

A typical call is:

```python
plot_top_n_feature_shap_2d(
    shap_plot_values,
    n_test=20,
    top_n_time_steps=15,
    target_class_index=0,
    feature_labels=feature_names,
    save_path="results/transformer_shap/shap_heatmap_class_0.pdf",
    mode="trans"
)
```

---

## Explanation of `plot_time_step_bar_chart()`

The function:

```python
plot_time_step_bar_chart(...)
```

summarizes the most important token positions across classes.

It calculates the total absolute SHAP contribution for each token position and plots the top-ranked positions.

The plot can be either:

| Mode | Description |
|---|---|
| Stacked bar chart | Shows class contributions stacked together |
| Grouped bar chart | Shows class contributions side by side |

Example:

```python
plot_time_step_bar_chart(
    shap_plot_values,
    n_test=20,
    top_n_time_steps=50,
    feature_labels=feature_names,
    stacked=True,
    save_path="results/transformer_shap/shap_bar_top_tokens.pdf"
)
```

---

## Explanation of `plot_aggregated_time_importance()`

The function:

```python
plot_aggregated_time_importance(...)
```

generates a stacked area-style visualization for important token positions.

It is useful for showing how class-specific SHAP importance is distributed across the top-ranked MassLinker tokens.

Example:

```python
plot_aggregated_time_importance(
    shap_plot_values,
    n_test=20,
    top_n_time_steps=50,
    feature_labels=feature_names,
    save_path="results/transformer_shap/shap_aggregated_top_tokens.pdf"
)
```

---

## Feature labels

The plotting functions can use metabolite or compound names as token labels.

These names are loaded from:

```text
metadata/pathway_compound_detail.csv
```

The expected column is:

```text
compound_names
```

Example:

```python
feature_names = pd.read_csv(
    "metadata/pathway_compound_detail.csv"
)["compound_names"].tolist()
```

The order of `compound_names` should match the token order in the MassLinker dataset.

---

## Choosing `n_background` and `n_test`

SHAP analysis for Transformer models can be computationally expensive.

The following settings are recommended for initial testing:

| Parameter | Suggested value | Description |
|---|---:|---|
| `n_background` | `5` to `10` | Number of background samples for SHAP |
| `n_test` | `2` to `5` | Number of test samples to explain |
| `top_n_time_steps` | `10` to `20` | Number of token positions shown in heatmaps |

For final analysis, these values can be increased if GPU memory allows.

Example:

```python
n_background = 10
n_test = 20
top_n_time_steps = 15
```

---

## GPU memory considerations

Transformer SHAP analysis is memory-intensive.

If CUDA memory is limited, reduce:

1. `n_background`;
2. `n_test`;
3. Transformer model size;
4. Number of classes analyzed;
5. Number of top token positions plotted.

A safe first test is:

```python
n_background = 5
n_test = 2
```

After confirming that the workflow runs successfully, increase the values gradually.

---

## Notes

1. This example is intended for interpreting trained Transformer classifiers.
2. The Transformer architecture used for SHAP must match the architecture used during training.
3. The input tensor should be reshaped to `n_samples × n_tokens × 60`.
4. The token dimension `n_tokens` should be inferred from the dataset instead of hard-coded.
5. The feature annotation table should follow the same token order as `dataset.samples`.
6. SHAP `DeepExplainer` can be slow and memory-intensive for Transformer models.
7. Use a small `n_background` and `n_test` for the first run.
8. The saved `shap_values.joblib` file allows plots to be regenerated without recalculating SHAP values.
9. The heatmap separates the 60 token features into height, position, and width parameter groups.
10. This example is an advanced interpretation workflow and is not required for basic MassLinker analysis.
