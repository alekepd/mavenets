# mavenets

Tools for training neural networks on MAVE datasets and running Markov Chain Monte Carlo simulations.

## Citation

Durumeric, A. E. P.; McCarty, S.; Smith, J.; Köhler, J.; Elez, K.; Raich, L.; Suriana, P. A.; Sztain, T. Machine Learning-Driven Simulations of the SARS-CoV-2 Fitness Landscape from Deep Mutational Scanning Experiments. <i>J. Chem. Inf. Model.</i> <b>2026</b>, 66 (10), 5721–5735. [link](https://doi.org/10.1021/acs.jcim.6c00332)



## Installation

Installation is supported through a combination of `conda` and `pip`. Here is an example setup that will 
install a compatible environment into `./env` and installs the package using `cuda 12.4`. See pytorch
and pyg websites for more detailed installation options. For a basic (non-T5), the following should work.

```bash
conda create --prefix ./env python==3.9 pandas pytorch torchvision torchaudio pytorch-cuda=12.4 torch-scatter einops mdtraj numpy -c pytorch -c nvidia -c conda-forge
conda activate ./env
conda install pyg -c pyg
pip install triton
mkdir -p ./src
git clone https://github.com/SztainLab/mavenets.git ./src
pip install -e ./src
```

To install a T5 compatible environment, use the following.

```bash
conda create --prefix ./env python==3.9 transformers pandas pytorch torchvision torchaudio pytorch-cuda=12.4 torch-scatter einops mdtraj numpy -c huggingface -c pytorch -c nvidia -c conda-forge
conda activate ./env
conda install pyg -c pyg
pip uninstall tokenizers
pip install transformers
pip install triton
pip install sentencepiece
mkdir -p ./src
git clone https://github.com/SztainLab/mavenets.git ./src
pip install -e ./src
```

## Usage

Example scripts are provided in `./src/mavenets/example`. They can be invoked in the shell for
easy usage. For example:
```python
from mavenets.example import run_mlp
run_mlp.scan()
```
will launch a sample hyperparameter scan over possible multilayer perceptron architectures.
