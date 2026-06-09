1. Clone the repositor
``` bash
git clone https://github.com/Innovation888K/MassLinkerSuite
cd MassLinkerSuite
```
2. Install Python dependencies
Create and activate a conda environment:
```bash
conda create -n masslinker python=3.10 -y
conda activate masslinker
```
Install MassLinker Suite and its Python dependencies:
```bash
pip install -e .
```
For GPU users, we recommend installing PyTorch according to the local CUDA version before installing MassLinker Suite. For example, for CUDA 12.4:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e .
```
Alternatively, the Python dependencies can be installed from requirements.txt:
```bash
pip install -r requirements.txt
```
3. Install R dependencies
3. Install R dependencies
MassLinker uses R for KEGG metadata retrieval, mzML data loading, MS1 signal extraction, and RBF-based MassLinker token encoding.
The required R packages are:
pracma
mzR
KEGGREST
tcltk
Install the required R packages with:
```R
install.packages("pracma")

if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager")
}

BiocManager::install(c("mzR", "KEGGREST"))
```
The tcltk package is usually included with standard R installations. To check whether it is available, run:
```R
capabilities("tcltk")
library(tcltk)
```
4. Prepare mzML files
Raw vendor LC–MS files should be converted to mzML format before MassLinker encoding. We recommend using ProteoWizard MSConvert.

Example command:
```bash
msconvert input.raw --mzML --filter "msLevel 1" -o ./mzML/
```
The expected input for MassLinker encoding is a directory containing .mzML files.
