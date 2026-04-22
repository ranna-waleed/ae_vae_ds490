import os
import json
import numpy as np
import tensorflow as tf

from models import Autoencoder, VAE, LATENT_DIM
from data_loader import get_all_regions

EPOCHS = 15
SAVE_DIR = "saved_models"
HISTORY_DIR = "histories"


def train_ae_for_region(region_name, dataset):
    print(f"\n>>> Training AE  |  region: {region_name}")

    model = Autoencoder(latent_dim=LATENT_DIM)
    model.compile(optimizer="adam", loss="binary_crossentropy")

    history = model.fit(dataset, epochs=EPOCHS, verbose=1)

    # save weights
    save_path = os.path.join(SAVE_DIR, region_name, "ae")
    os.makedirs(save_path, exist_ok=True)
    model.save_weights(os.path.join(save_path, "weights.weights.h5"))

    # save loss history
    hist_path = os.path.join(HISTORY_DIR, region_name)
    os.makedirs(hist_path, exist_ok=True)
    with open(os.path.join(hist_path, "ae_history.json"), "w") as f:
        json.dump(history.history, f)

    print(f"    AE saved  ->  {save_path}")
    return model, history


def train_vae_for_region(region_name, dataset):
    print(f"\n>>> Training VAE |  region: {region_name}")

    model = VAE(latent_dim=LATENT_DIM)
    model.compile(optimizer="adam")

    history = model.fit(dataset, epochs=EPOCHS, verbose=1)
    dummy = tf.zeros((1, 64, 64, 1))
    model(dummy)
    save_path = os.path.join(SAVE_DIR, region_name, "vae")
    os.makedirs(save_path, exist_ok=True)
    model.save_weights(os.path.join(save_path, "weights.weights.h5"))

    hist_path = os.path.join(HISTORY_DIR, region_name)
    os.makedirs(hist_path, exist_ok=True)
    with open(os.path.join(hist_path, "vae_history.json"), "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f)

    print(f"    VAE saved ->  {save_path}")
    return model, history


def train_all(data_dir):
    """Train one AE and one VAE per anatomical region."""
    print("Loading datasets...")
    all_regions = get_all_regions(data_dir)

    ae_models = {}
    vae_models = {}

    for region_name, dataset in all_regions.items():
        ae_model, ae_hist = train_ae_for_region(region_name, dataset)
        vae_model, vae_hist = train_vae_for_region(region_name, dataset)

        ae_models[region_name] = ae_model
        vae_models[region_name] = vae_model

    print("\nAll regions done!")
    return ae_models, vae_models


def load_trained_models(data_dir):
    """
    Load already-trained weights so you don't retrain every run.
    Call this instead of train_all() if models are already saved.
    """
    all_regions = get_all_regions(data_dir)
    ae_models = {}
    vae_models = {}

    for region_name in all_regions:
        ae = Autoencoder(latent_dim=LATENT_DIM)
        vae = VAE(latent_dim=LATENT_DIM)

        # build the model by passing a dummy batch
        dummy = tf.zeros((1, 64, 64, 1))
        ae(dummy)
        vae(dummy)

        ae_path = os.path.join(SAVE_DIR, region_name, "ae", "weights.weights.h5")
        vae_path = os.path.join(SAVE_DIR, region_name, "vae", "weights.weights.h5")

        if os.path.exists(ae_path):
            ae.load_weights(ae_path)
            print(f"Loaded AE weights for {region_name}")

        if os.path.exists(vae_path ):
            vae.load_weights(vae_path)
            print(f"Loaded VAE weights for {region_name}")

        ae_models[region_name] = ae
        vae_models[region_name] = vae

    return ae_models, vae_models