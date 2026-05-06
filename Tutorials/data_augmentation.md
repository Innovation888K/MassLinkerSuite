<h1 align="left">
  LC–MS data augmentation
</h1>

This tutorial describes how to generate augmented LC–MS files for MassLinker Suite.

The script `data_enhance.R` creates augmented versions of raw LC–MS files by applying small random perturbations to m/z values, peak intensities, and retention times. The augmented files can be used to increase training data diversity and improve the robustness of downstream MassLinker tokenization and machine-learning models.


## Overview

MassLinker uses chromatographic signal structures extracted from raw LC–MS data. In some applications, especially when training deep-learning models, data augmentation can help reduce overfitting and improve model generalization.

The augmentation module performs three types of perturbation:

1. **m/z shift**  
   Each peak is shifted by a small random m/z offset, approximately following a ppm-scale mass error.

2. **Intensity perturbation**  
   Peak intensities are randomly scaled to simulate experimental variation.

3. **Retention-time shift**  
   Scan-level retention times are shifted slightly to simulate run-to-run chromatographic variation.


## Required R packages

Please install the required packages before running the script.

```r
install.packages("BiocManager")
BiocManager::install("mzR")
BiocManager::install("MSnbase")
BiocManager::install("ProtGenerics")
```
Then load the required packages:
```r
library(mzR)
library(MSnbase)
library(ProtGenerics)
```

## Input data
Place raw LC–MS files in one folder.

Example:
```Input
raw_data/
├── sample_01.mzXML
├── sample_02.mzXML
├── sample_03.mzXML
└── sample_04.mzXML
```
Supported formats depend on mzR, commonly including:
```text
mzXML
mzML
CDF
```
## Usage  
Run the script from the command line:
```bash
Rscript R/data_enhance.R ./raw_data ./augmented_data 5
```

## Arguments:
|Argument|	Description|
|---|---|
|work_dir|	Directory containing raw LC–MS files|
|output|	Directory for saving augmented LC–MS files|
|enhance_num| Number of augmented files generated for each input file|

## Output
For each input file, the script generates multiple augmented files.

Example:

```text
augmented_data/
├── 1-sample_01.mzXML
├── 2-sample_01.mzXML
├── 3-sample_01.mzXML
├── 1-sample_02.mzXML
├── 2-sample_02.mzXML
└── 3-sample_02.mzXML
```
The output files preserve the original file format and can be used as input for MassLinker token generation.

---
<h1 align="left">
  Augmentation strategy
</h1>

## m/z perturbation
For each peak, a small random m/z shift is applied:
```r
mz_shift <- rnorm(nrow(x), mean = 0, sd = 5) *
  ifelse(x$mz < 400, 400, x$mz) / 1000000
```
This simulates ppm-scale mass measurement error.

# Intensity perturbation
Peak intensities are randomly scaled:
```r
int_shift <- rnorm(nrow(x), mean = 0, sd = 0.06)
x$intensity <- x$intensity * (1 - int_shift)
```
This simulates moderate experimental intensity variation.

## Retention-time shift
A small scan-level retention-time shift is applied:
```r
rt_diff <- rnorm(enhance_num, mean = 0, sd = 10)
```
This simulates chromatographic retention-time variation across LC–MS runs.


## Notes
The augmentation module is mainly intended for model training and robustness evaluation.
For final biological interpretation, it is recommended to report results on the original non-augmented samples.
The default augmentation settings are moderate and designed to mimic common LC–MS technical variation.
Users may adjust the m/z, intensity, and retention-time perturbation parameters according to their instrument platform and experimental design.
If downstream analysis is sensitive to exact retention time, please verify that the retention-time shift range is appropriate for the dataset.









