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
