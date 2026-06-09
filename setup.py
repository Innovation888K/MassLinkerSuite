from setuptools import setup, find_packages
from pathlib import Path


# Read README file as the long description shown on PyPI/GitHub-compatible tools.
this_directory = Path(__file__).parent
readme_file = this_directory / "README.md"

if readme_file.exists():
    long_description = readme_file.read_text(encoding="utf-8")
else:
    long_description = ""


setup(
    name="masslinker-suite",
    version="0.1.0",
    author="MassLinker Development Team",
    author_email="tfxu@zju.edu.cn",
    description=(
        "A cross-language computational framework for tokenizing raw LC-MS "
        "metabolomics data into interpretable MassLinker metabolic tokens."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YOUR_USERNAME/MassLinker",
    license="Academic and non-commercial use only",

    # Automatically discover Python packages in the repository.
    # If your scripts are not organized as packages, see the note below.
    packages=find_packages(),

    # Include non-code files if MANIFEST.in is provided.
    include_package_data=True,

    # Python version requirement.
    python_requires=">=3.9,<3.12",

    # Core dependencies inferred from your imports and pip list.
    install_requires=[
        # Basic scientific computing
        "numpy>=1.23",
        "pandas>=1.5",
        "scipy>=1.9",
        "scikit-learn>=1.2",
        "statsmodels>=0.13",

        # Plotting and visualization
        "matplotlib>=3.6",
        "seaborn>=0.12",
        "plotly>=5.0",
        "adjustText>=1.0",

        # Progress bars and serialization
        "tqdm>=4.64",
        "joblib>=1.2",

        # Machine learning models
        "xgboost>=1.7",
        "lightgbm>=3.3",
        "shap>=0.42",

        # Deep learning
        # Note: PyTorch installation may depend on CUDA/CPU environment.
        # For most CPU users, this can be installed automatically.
        # GPU users are recommended to install torch manually first.
        "torch>=2.0",
        "torchvision>=0.15",
        "timm>=0.9",

        # Dimensionality reduction and manifold learning
        "umap-learn>=0.5",

        # Network/pathway analysis
        "igraph>=0.10",
        "networkx>=3.0",
        "goatools>=1.3",
        "gseapy>=1.0",

        # Excel and table IO
        "openpyxl>=3.1",
        "XlsxWriter>=3.0",

        # Optional MS-related Python utilities used in some workflows
        "pymzml>=2.5",
        "pyteomics>=4.6",
    ],

    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=23.0",
            "flake8>=6.0",
            "isort>=5.0",
        ],
        "gpu": [
            # GPU-specific PyTorch should usually be installed from the official PyTorch index.
            # This placeholder is intentionally not pinned to a CUDA wheel.
            "torch>=2.0",
            "torchvision>=0.15",
        ],
        "docs": [
            "mkdocs>=1.5",
            "mkdocs-material>=9.0",
        ],
    },

    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],

    keywords=[
        "LC-MS",
        "metabolomics",
        "mass spectrometry",
        "machine learning",
        "transformer",
        "RBF",
        "SHAP",
        "KEGG",
    ],
)
