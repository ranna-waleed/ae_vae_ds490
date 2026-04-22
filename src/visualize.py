"""
visualize.py
All visualization functions for AE and VAE experiments.
Saves plots to the plots/ directory organized by region.
"""

# Standard library
import os
import json
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

PLOTS_DIR: str = "plots"


def _ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def plot_reconstructions(
    model: tf.keras.Model,
    sample_images: List[np.ndarray],
    region_name: str,
    model_type: str = "AE"
) -> None:
    """
    Plot original images alongside their reconstructions.

    Args:
        model: Trained AE or VAE model with a callable forward pass.
        sample_images: List of numpy arrays (H, W, 1).
        region_name: Used for plot title and save path.
        model_type: Either "AE" or "VAE" for labeling.
    """
    n = min(8, len(sample_images))
    fig, axes = plt.subplots(2, n, figsize=(n * 2, 4))
    fig.suptitle(f"{model_type} Reconstructions — {region_name}", fontsize=13)

    for i in range(n):
        img = np.expand_dims(sample_images[i], axis=0)
        reconstructed = model(img).numpy()[0]

        axes[0, i].imshow(sample_images[i].squeeze(), cmap="gray")
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("Original", fontsize=9)

        axes[1, i].imshow(reconstructed.squeeze(), cmap="gray")
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("Recon.", fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(
        PLOTS_DIR, region_name, f"{model_type.lower()}_reconstructions.png"
    )
    _ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=120)
    plt.show()
    print(f"Saved -> {save_path}")


def plot_latent_space(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    region_name: str,
    model_type: str = "AE",
    max_batches: int = 10
) -> None:
    """
    Scatter plot of 2D latent representations.

    Works directly since LATENT_DIM=2 (no PCA needed).

    Args:
        model: Model with an encode() method.
        dataset: tf.data.Dataset yielding (image, image) batches.
        region_name: Used for title and save path.
        model_type: Either "AE" or "VAE".
        max_batches: Number of batches to encode.
    """
    all_z = []

    for batch_x, _ in dataset.take(max_batches):
        z = model.encode(batch_x).numpy()
        all_z.append(z)

    all_z = np.concatenate(all_z, axis=0)

    plt.figure(figsize=(6, 6))
    plt.scatter(all_z[:, 0], all_z[:, 1], alpha=0.4, s=10, color="steelblue")
    plt.title(f"{model_type} Latent Space — {region_name}")
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.tight_layout()

    save_path = os.path.join(
        PLOTS_DIR, region_name, f"{model_type.lower()}_latent_space.png"
    )
    _ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=120)
    plt.show()
    print(f"Saved -> {save_path}")


def plot_generated_samples(
    vae_model: tf.keras.Model,
    region_name: str,
    grid_size: int = 8
) -> None:
    """
    Generate images by decoding a regular grid in 2D latent space.

    Args:
        vae_model: Trained VAE model with a decode() method.
        region_name: Used for title and save path.
        grid_size: Number of points per axis in the grid.
    """
    lin = np.linspace(-3, 3, grid_size)
    fig, axes = plt.subplots(
        grid_size, grid_size, figsize=(grid_size * 1.5, grid_size * 1.5)
    )
    fig.suptitle(f"VAE Generated Samples — {region_name}", fontsize=13)

    for i, yi in enumerate(lin):
        for j, xi in enumerate(lin):
            z = np.array([[xi, yi]], dtype=np.float32)
            img = vae_model.decode(z).numpy()[0]
            axes[i, j].imshow(img.squeeze(), cmap="gray")
            axes[i, j].axis("off")

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, region_name, "vae_generated_grid.png")
    _ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=120)
    plt.show()
    print(f"Saved -> {save_path}")


def plot_denoising(
    model: tf.keras.Model,
    sample_images: List[np.ndarray],
    region_name: str,
    model_type: str = "AE"
) -> None:
    """
    Show clean -> noisy -> denoised comparison.

    Args:
        model: Trained model used to denoise images.
        sample_images: List of clean numpy arrays.
        region_name: Used for title and save path.
        model_type: Either "AE" or "VAE".
    """
    n = min(6, len(sample_images))
    fig, axes = plt.subplots(3, n, figsize=(n * 2, 6))
    fig.suptitle(f"{model_type} Denoising — {region_name}", fontsize=13)
    row_labels = ["Clean", "Noisy", "Denoised"]

    for i in range(n):
        clean = sample_images[i]
        noise = np.random.normal(0, 0.1, clean.shape).astype(np.float32)
        noisy = np.clip(clean + noise, 0.0, 1.0)

        inp = np.expand_dims(noisy, axis=0)
        denoised = model(inp).numpy()[0]

        for row, img in enumerate([clean, noisy, denoised]):
            axes[row, i].imshow(img.squeeze(), cmap="gray")
            axes[row, i].axis("off")
            if i == 0:
                axes[row, i].set_title(row_labels[row], fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(
        PLOTS_DIR, region_name, f"{model_type.lower()}_denoising.png"
    )
    _ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=120)
    plt.show()
    print(f"Saved -> {save_path}")


def plot_loss(region_name: str) -> None:
    """
    Plot training loss curves from saved JSON history files.

    Args:
        region_name: Used to locate history files and save the plot.
    """
    hist_dir = os.path.join("histories", region_name)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Training Losses — {region_name}", fontsize=13)

    # AE loss
    ae_path = os.path.join(hist_dir, "ae_history.json")
    if os.path.exists(ae_path):
        with open(ae_path) as f:
            ae_hist = json.load(f)
        axes[0].plot(ae_hist["loss"], label="AE loss", color="steelblue")
        axes[0].set_title("Autoencoder Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()

    # VAE losses
    vae_path = os.path.join(hist_dir, "vae_history.json")
    if os.path.exists(vae_path):
        with open(vae_path) as f:
            vae_hist = json.load(f)
        axes[1].plot(vae_hist.get("total_loss", []), label="Total", color="tomato")
        axes[1].plot(
            vae_hist.get("recon_loss", []), label="Recon", color="orange", linestyle="--"
        )
        axes[1].plot(
            vae_hist.get("kl_loss", []), label="KL", color="green", linestyle=":"
        )
        axes[1].set_title("VAE Losses")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].legend()

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, region_name, "loss_curves.png")
    _ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=120)
    plt.show()
    print(f"Saved -> {save_path}")