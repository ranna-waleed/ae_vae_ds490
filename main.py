"""
main.py
-------
Entry point for DSAI 490 Assignment 1.
Trains AE and VAE models for each Medical MNIST region
and generates all required visualizations.
"""

# Standard library
import os
import sys

# Add src/ to path so modules can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.data_processing import get_all_regions, get_sample_images
from src.train import train_all, load_trained_models
from src.visualize import (
    plot_reconstructions,
    plot_latent_space,
    plot_generated_samples,
    plot_denoising,
    plot_loss,
)

DATA_DIR: str = "data/raw/medical_mnist"

RETRAIN: bool = False


def main() -> None:
    """
    Main pipeline: train models then generate all visualizations.
    """
    # Create output directories
    for directory in ["plots", "models", "histories"]:
        os.makedirs(directory, exist_ok=True)

    # Train or load models
    if RETRAIN:
        ae_models, vae_models = train_all(DATA_DIR)
    else:
        ae_models, vae_models = load_trained_models(DATA_DIR)

    # Generate all visualizations per region
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

        # 2. Latent space scatter (2D, no PCA needed)
        plot_latent_space(ae, dataset, region_name, model_type="AE")
        plot_latent_space(vae, dataset, region_name, model_type="VAE")

        # 3. VAE: generate new samples from latent grid
        plot_generated_samples(vae, region_name, grid_size=8)

        # 4. Denoising demo
        plot_denoising(ae, samples, region_name, model_type="AE")
        plot_denoising(vae, samples, region_name, model_type="VAE")

        # 5. Loss curves
        plot_loss(region_name)

    print("\nDone! All plots saved in the 'plots/' folder.")


if __name__ == "__main__":
    main()