# Generating MassLinker metabolic tokens

This tutorial describes how to convert raw LC–MS files into MassLinker metabolic tokens.

The script `MSI_Image_converter.R` extracts chromatographic intensity traces for pathway-associated metabolites and fits each trace using radial basis functions. 

The resulting RBF parameters are concatenated into fixed-length metabolic tokens.

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
```
## Usage
```
Rscript MSI_Image_converter.R ./raw_data ./MassLinker_tokens positive path_to_root
```

## Ion mode
For positive ion mode, the script shifts the compound mass window by the proton mass and matches [M+H]+.

For negative ion mode, the script subtracts the proton mass and matches [M-H]-.

Users working with other adducts should modify the mass shift or prepare an adduct-specific compound mass table.

## Output

For each LC–MS sample, MassLinker generates a sample-specific token file:

```text
MassLinker_tokens/
└── sample_01/
    └── sample_01_MassLinker_tokens.csv
```
