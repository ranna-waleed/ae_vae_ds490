"""
DSAI 490 – Assignment 1
Autoencoder (AE) and Variational Autoencoder (VAE)
Dataset: Medical MNIST  (one model per anatomical region)
"""

import os
from data_loader import get_all_regions, get_sample_images
from train import train_all, load_trained_models
from visualize import (
    plot_reconstructions,
    plot_latent_space,
    plot_generated_samples,
    plot_denoising,
    plot_loss,
)
DATA_DIR = "dataset/medical_mnist" 

RETRAIN = True 


def main():
    os.makedirs("plots", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("histories", exist_ok=True)

    if RETRAIN:
        ae_models, vae_models = train_all(DATA_DIR)
    else:
        ae_models, vae_models = load_trained_models(DATA_DIR)

    all_regions = get_all_regions(DATA_DIR)

    for region_name, dataset in all_regions.items():
        print(f"\n=== Visualizing: {region_name} ===")

        region_path = os.path.join(DATA_DIR, region_name)
        samples = get_sample_images(region_path, num_samples=10)

        ae = ae_models[region_name]
        vae = vae_models[region_name]

        # 1. Reconstruction comparison
        plot_reconstructions(ae, samples, region_name, model_type="AE")
        plot_reconstructions(vae, samples, region_name, model_type="VAE")

        # 2. Latent space scatter
        plot_latent_space(ae, dataset, region_name, model_type="AE")
        plot_latent_space(vae, dataset, region_name, model_type="VAE")

        # 3. VAE generates new samples from a 2D latent grid
        plot_generated_samples(vae, region_name, grid_size=8)

        # 4. Denoising demo
        plot_denoising(ae, samples, region_name, model_type="AE")
        plot_denoising(vae, samples, region_name, model_type="VAE")

        # 5. Loss curves
        plot_loss(region_name)

    print("\nDone! All plots saved in the 'plots/' folder.")


if __name__ == "__main__":
    main()