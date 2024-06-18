# VECTOR GRIMOIRE: Codebook-based Stroke Generation under Raster Image Supervision

Welcome to the anonymous repository of Vector Grimoire! 🧙

# Installation

```bash
$ conda create --name SVG python=3.10
$ conda activate SVG
```
Clone and Install [diffsvg](https://github.com/BachiLi/diffvg) as explained in the repo:
```bash
git clone git@github.com:BachiLi/diffvg.git
git submodule update --init --recursive
conda install -y pytorch torchvision -c pytorch
conda install -y numpy
conda install -y scikit-image
conda install -y -c anaconda cmake
conda install -y -c conda-forge ffmpeg
pip install svgwrite
pip install svgpathtools
pip install cssutils
pip install numba
pip install torch-tools
pip install visdom
python setup.py install
```
Install our requirements: 
```bash
$ conda env update -n SVG --file requirements.yaml
```

# Datasets
In our paper we used fonts and icons (FIGR-8). 
For your convenience, and to simplify the reviewing process, instead of providing a script to download the raw data and have the reviewerds performing the pre-processing pipeline,
we have uploaded the already pre-processed data on an anonymous Hugging Face repository. 
Reviewers can download the data for icons at [this link](https://huggingface.co/datasets/anon-submission-data/xyz). We can also provide a ready-to-go folder for Fonts upon request.

This archive contains two zipped files: 
- simplified.zip: a folder with the pre-processsed icons from FIGR-8 required to train and test our Stage-1
- tokenized.zip: a folder with the data already tokenized with our VSQ module after training on Stage-1, and required to train and test Stage-2.
- stage_1.ckpt: checkpoint file you need for the decoder in Stage-2, and which we used to generate tokenized.zip

The whole codebase is written in pytorch lightning and supports wandb logging for both losses, images and codebook distribution.

# Training VQ-VAE module

This is a minimal version of our repository, to train the VSQ module, please adjust the path in `config/code_release/icons_stage1.yaml` (check for TODOs)
then simply run:
```
python run.py --config config/code_release/icons_stage1.yaml
```

# Training of autoregressive model
to train the autoregressive model, please adjust the path in `config/code_release/icons_stage2.yaml`  (check for TODOs)
then simply run:

```
python run_stage2.py --config config/code_release/icons_stage2.yaml
```

# Post-processing

To apply the post-processing algorithms described in the appendix, consider using `svg_fixing.py`

# Additional parts 

We plan to release clear and refactored pre-processing and post-processing pipelines upon acceptance, however, we are available to provide instructions and code if needed. 
