# Generating MassLinker metabolic tokens

This tutorial describes how to convert raw LC–MS files into MassLinker metabolic tokens.

The script `MSI_Image_converter.R` extracts chromatographic intensity traces for pathway-associated metabolites and fits each trace using radial basis functions. The resulting RBF parameters are concatenated into fixed-length metabolic tokens.

## Required files

Before running the script, please make sure the following files are available:

| File | Description |
|---|---|
| `compounds_detail_res.Rda` | Compound exact mass table with m/z matching windows |
| `kegg_pathways.Rda` | KEGG pathway-to-compound mapping |
| `kegg_pathways_id.Rda` | KEGG pathway identifiers |
| `RBF_fit.R` | RBF fitting function |

## Input data

Place raw LC–MS files in one folder:

```text
raw_data/
├── sample_01.mzXML
├── sample_02.mzXML
└── sample_03.mzXML
