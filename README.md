# DSAI 490 : AE & VAE on Medical MNIST

## Overview
This project implements Autoencoders (AE) and Variational Autoencoders (VAE)
for representation learning on the Medical MNIST dataset. A separate AE and VAE
is trained for each of the 6 anatomical regions.

## Dataset
- **Source:** https://www.kaggle.com/datasets/andrewmvd/medical-mnist
- **Regions:** AbdomenCT, BreastMRI, ChestCT, CXR, Hand, HeadCT
- **Image size:** 64×64 grayscale

## Project Structure
```
ae_vae_ds490/
├── data/
│   └── raw/
│       └── medical_mnist/
│           ├── AbdomenCT/
│           ├── BreastMRI/
│           ├── ChestCT/
│           ├── CXR/
│           ├── Hand/
│           └── HeadCT/
├── models/               # saved model weights (gitignored)
├── histories/            # training loss JSON files (gitignored)
├── plots/                # generated visualizations (gitignored)
├── src/
│   ├── __init__.py
│   ├── data_processing.py   # tf.data pipeline
│   ├── model.py             # AE and VAE architectures
│   ├── train.py             # training functions
│   └── visualize.py         # all plot generation
├── main.py               # entry point
├── requirements.txt
└── README.md
```

## Setup

### 1. Create environment
```bash
conda create -n ds490 python=3.11
conda activate ds490
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Place dataset
Put the 6 region folders inside `data/raw/medical_mnist/`

### 4. Run
```bash
python main.py
```

Set `RETRAIN = False` in `main.py` after the first run to skip retraining.

## Outputs
Each region produces 8 plots inside `plots/<region>/`:
- `ae_reconstructions.png`
- `vae_reconstructions.png`
- `ae_latent_space.png`
- `vae_latent_space.png`
- `vae_generated_grid.png`
- `ae_denoising.png`
- `vae_denoising.png`
- `loss_curves.png`
