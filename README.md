<h1 align="left">
  MassLinker
  <img src="./assets/logo.png" align="right" width="400" />
</h1>

<br />
<br />
<br />
<br />
<br />
<br />

---

## About

**MassLinker Suite** is a cross-language computational framework for tokenizing raw liquid chromatography–mass spectrometry LC–MS metabolomics data into interpretable and machine-readable metabolic representations.

 <img src="./assets/readme1.png" align="mid" width="800" />
 
The framework implements **MassLinker encoding**, which fits MS1 chromatographic signals at specific m/z values using radial basis functions and extracts peak height, peak width, and peak position as chemically meaningful descriptors. These descriptors are concatenated into fixed-length metabolic tokens that preserve chromatographic signal intensity, peak morphology, and retention behavior.

Unlike conventional metabolomics workflows that compress raw LC–MS signals into metabolite abundance tables at an early stage, MassLinker retains signal-level structure before downstream interpretation. This enables richer representations for machine learning, Transformer-based classification, visualization, and SHAP-based model interpretation.

MassLinker Suite integrates R and Python modules into a standardized workflow. The R-based modules support feature extraction, RBF fitting, parameterization, and feature reorganization, while the Python-based modules support dataset construction, model training, statistical analysis, visualization, and interpretability analysis.

Overall, MassLinker Suite provides an end-to-end bridge between raw LC–MS measurements, interpretable metabolic tokens, and clinically relevant computational inference.
