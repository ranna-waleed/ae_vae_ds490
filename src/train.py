"""
train.py

Training pipeline for AE and VAE models.
Trains one AE and one VAE per anatomical region and saves weights.
"""

# Standard library
import os
import json
from typing import Dict, Tuple
import tensorflow as tf

# Local
from .models import Autoencoder, VAE, LATENT_DIM
from .data_processing import get_all_regions

# Training configuration
EPOCHS: int = 15
MODELS_DIR: str = "models"
HISTORY_DIR: str = "histories"


def train_ae_for_region(
    region_name: str,
    dataset: tf.data.Dataset
) -> Tuple[Autoencoder, dict]:
    """
    Train an Autoencoder on a single anatomical region.

    Args:
        region_name: Name of the region (used for saving).
        dataset: tf.data.Dataset of (image, image) pairs.

    Returns:
        Tuple of (trained Autoencoder model, loss history dict).
    """
    print(f"\n>>> Training AE  |  region: {region_name}")

    model = Autoencoder(latent_dim=LATENT_DIM)
    model.compile(optimizer="adam", loss="binary_crossentropy")

    history = model.fit(dataset, epochs=EPOCHS, verbose=1)

    # Build model before saving weights
    dummy = tf.zeros((1, 64, 64, 1))
    model(dummy)

    # Save weights
    save_path = os.path.join(MODELS_DIR, region_name, "ae")
    os.makedirs(save_path, exist_ok=True)
    model.save_weights(os.path.join(save_path, "weights.weights.h5"))

    # Save loss history for later plotting
    hist_path = os.path.join(HISTORY_DIR, region_name)
    os.makedirs(hist_path, exist_ok=True)
    with open(os.path.join(hist_path, "ae_history.json"), "w") as f:
        json.dump(history.history, f)

    print(f"    AE saved -> {save_path}")
    return model, history.history


def train_vae_for_region(
    region_name: str,
    dataset: tf.data.Dataset
) -> Tuple[VAE, dict]:
    """
    Train a Variational Autoencoder on a single anatomical region.

    Args:
        region_name: Name of the region (used for saving).
        dataset: tf.data.Dataset of (image, image) pairs.

    Returns:
        Tuple of (trained VAE model, loss history dict).
    """
    print(f"\n>>> Training VAE |  region: {region_name}")

    model = VAE(latent_dim=LATENT_DIM)
    model.compile(optimizer="adam")

    history = model.fit(dataset, epochs=EPOCHS, verbose=1)

    # Build model before saving weights
    dummy = tf.zeros((1, 64, 64, 1))
    model(dummy)

    # Save weights
    save_path = os.path.join(MODELS_DIR, region_name, "vae")
    os.makedirs(save_path, exist_ok=True)
    model.save_weights(os.path.join(save_path, "weights.weights.h5"))

    # Save loss history
    hist_path = os.path.join(HISTORY_DIR, region_name)
    os.makedirs(hist_path, exist_ok=True)
    with open(os.path.join(hist_path, "vae_history.json"), "w") as f:
        json.dump(
            {k: [float(v) for v in vals] for k, vals in history.history.items()}, f
        )

    print(f"    VAE saved -> {save_path}")
    return model, history.history


def train_all(
    data_dir: str
) -> Tuple[Dict[str, Autoencoder], Dict[str, VAE]]:
    """
    Train one AE and one VAE for every anatomical region.

    Args:
        data_dir: Root directory of the Medical MNIST dataset.

    Returns:
        Tuple of (ae_models dict, vae_models dict),
        each mapping region_name -> trained model.
    """
    print("Loading datasets...")
    all_regions = get_all_regions(data_dir)

    ae_models: Dict[str, Autoencoder] = {}
    vae_models: Dict[str, VAE] = {}

    for region_name, dataset in all_regions.items():
        ae_models[region_name], _ = train_ae_for_region(region_name, dataset)
        vae_models[region_name], _ = train_vae_for_region(region_name, dataset)

    print("\nAll regions done!")
    return ae_models, vae_models


def load_trained_models(
    data_dir: str
) -> Tuple[Dict[str, Autoencoder], Dict[str, VAE]]:
    """
    Load previously saved model weights without retraining.

    Args:
        data_dir: Root directory of the Medical MNIST dataset.

    Returns:
        Tuple of (ae_models dict, vae_models dict).
    """
    all_regions = get_all_regions(data_dir)
    ae_models: Dict[str, Autoencoder] = {}
    vae_models: Dict[str, VAE] = {}

    dummy = tf.zeros((1, 64, 64, 1))

    for region_name in all_regions:
        ae = Autoencoder(latent_dim=LATENT_DIM)
        vae = VAE(latent_dim=LATENT_DIM)

        # Build models with a dummy forward pass
        ae(dummy)
        vae(dummy)

        ae_path = os.path.join(MODELS_DIR, region_name, "ae", "weights.weights.h5")
        vae_path = os.path.join(MODELS_DIR, region_name, "vae", "weights.weights.h5")

        if os.path.exists(ae_path):
            ae.load_weights(ae_path)
            print(f"Loaded AE weights for {region_name}")

        if os.path.exists(vae_path):
            vae.load_weights(vae_path)
            print(f"Loaded VAE weights for {region_name}")

        ae_models[region_name] = ae
        vae_models[region_name] = vae

    return ae_models, vae_models