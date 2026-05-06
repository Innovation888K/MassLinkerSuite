# before python
Before you start the tutorial for the Python stage, please make sure that you have completed the following file structure.
analysis_project/
├── data/
│   └── processed_dataset.joblib
├── metadata/
│   └── pathway_compound_detail.csv
├── cache/
│   ├── JS_distance.joblib
│   └── Wasserstein_distance.joblib
├── results/
│   ├── p_value_rank.pdf
│   ├── KEGG_enrichment.pdf
│   ├── metabolite_enrichment_analysis_results.xlsx
│   ├── differential_peaks/
│   │   ├── feature_001.pdf
│   │   ├── feature_002.pdf
│   │   └── ...
│   └── distance_visualization/
│       ├── pca.pdf
│       └── tsne.pdf
└── scripts/
    └── small_sample_statistical_comparison.py
