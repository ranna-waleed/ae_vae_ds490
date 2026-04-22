import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

PLOTS_DIR = "plots"


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


#  Reconstruction comparison

def plot_reconstructions(model, sample_images, region_name, model_type="AE"):
    """Show original vs reconstructed images side by side."""
    n = min(8, len(sample_images))
    fig, axes = plt.subplots(2, n, figsize=(n * 2, 4))
    fig.suptitle(f"{model_type} Reconstructions — {region_name}", fontsize=13)

    for i in range(n):
        img = np.expand_dims(sample_images[i], axis=0)  # (1, 64, 64, 1)
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
    save_path = os.path.join(PLOTS_DIR, region_name, f"{model_type.lower()}_reconstructions.png")
    _ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=120)
    plt.show()
    print(f"Saved -> {save_path}")


#  Latent space (2D scatter)

def plot_latent_space(model, dataset, region_name, model_type="AE", max_batches=10):
    """
    Encode images and scatter their latent vectors in 2D.
    Works because LATENT_DIM = 2.
    """
    all_z = []

    for batch_x, _ in dataset.take(max_batches):
        z = model.encode(batch_x).numpy()
        all_z.append(z)

    all_z = np.concatenate(all_z, axis=0)  # (N, 2)

    plt.figure(figsize=(6, 6))
    plt.scatter(all_z[:, 0], all_z[:, 1], alpha=0.4, s=10, color="steelblue")
    plt.title(f"{model_type} Latent Space — {region_name}")
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, region_name, f"{model_type.lower()}_latent_space.png")
    _ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=120)
    plt.show()
    print(f"Saved -> {save_path}")


#  VAE sample generation (grid from latent grid)

def plot_generated_samples(vae_model, region_name, grid_size=8):
    """
    Sample points from a regular 2D grid in latent space
    and decode them to images.
    """
    # build a grid over [-3, 3] x [-3, 3]
    lin = np.linspace(-3, 3, grid_size)
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 1.5, grid_size * 1.5))
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


#  Denoising demo

def plot_denoising(model, sample_images, region_name, model_type="AE"):
    """Show: clean -> noisy -> denoised."""
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
    save_path = os.path.join(PLOTS_DIR, region_name, f"{model_type.lower()}_denoising.png")
    _ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=120)
    plt.show()
    print(f"Saved -> {save_path}")


#  Training loss curves

def plot_loss(region_name):
    """Plot loss curves from saved history JSON files."""
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
        axes[1].plot(vae_hist.get("recon_loss", []), label="Recon", color="orange", linestyle="--")
        axes[1].plot(vae_hist.get("kl_loss", []), label="KL", color="green", linestyle=":")
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