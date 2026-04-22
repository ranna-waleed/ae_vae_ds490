"""
data_processing.py
Handles all data loading and preprocessing for the Medical MNIST dataset.
Uses tf.data for efficient data pipelines.
"""

# Standard library
import os
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf

# Constants
IMG_SIZE: int = 64
BATCH_SIZE: int = 32
AUTOTUNE = tf.data.AUTOTUNE


def load_image(file_path: tf.Tensor) -> tf.Tensor:
    """
    Load and preprocess a single image from disk.

    Args:
        file_path: Path tensor pointing to the image file.

    Returns:
        Normalized float32 tensor of shape (64, 64, 1).
    """
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=1)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    return img


def add_noise(img: tf.Tensor) -> tf.Tensor:
    """
    Add Gaussian noise to an image for denoising experiments.

    Args:
        img: Input image tensor.

    Returns:
        Noisy image tensor clipped to [0, 1].
    """
    noise = tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=0.1)
    return tf.clip_by_value(img + noise, 0.0, 1.0)


def get_region_dataset(
    region_path: str,
    batch_size: int = BATCH_SIZE,
    for_denoising: bool = False
) -> tf.data.Dataset:
    """
    Build a tf.data.Dataset for a single anatomical region.

    Args:
        region_path: Path to the folder containing .jpeg images.
        batch_size: Number of images per batch.
        for_denoising: If True, returns (noisy, clean) pairs.
                       Otherwise returns (clean, clean) pairs.

    Returns:
        Batched and prefetched tf.data.Dataset.
    """
    pattern = os.path.join(region_path, "*.jpeg")
    image_paths = tf.data.Dataset.list_files(pattern, shuffle=True)
    images = image_paths.map(load_image, num_parallel_calls=AUTOTUNE)

    if for_denoising:
        dataset = images.map(
            lambda img: (add_noise(img), img),
            num_parallel_calls=AUTOTUNE
        )
    else:
        dataset = images.map(
            lambda img: (img, img),
            num_parallel_calls=AUTOTUNE
        )

    return dataset.batch(batch_size).prefetch(AUTOTUNE)


def get_all_regions(
    data_dir: str,
    batch_size: int = BATCH_SIZE
) -> Dict[str, tf.data.Dataset]:
    """
    Scan the dataset directory and return one dataset per region.

    Args:
        data_dir: Root directory containing one subfolder per region.
        batch_size: Number of images per batch.

    Returns:
        Dictionary mapping region name -> tf.data.Dataset.
    """
    regions: Dict[str, tf.data.Dataset] = {}

    for folder_name in sorted(os.listdir(data_dir)):
        full_path = os.path.join(data_dir, folder_name)
        if os.path.isdir(full_path):
            regions[folder_name] = get_region_dataset(full_path, batch_size)
            print(f"  Loaded region: {folder_name}")

    return regions


def get_sample_images(
    region_path: str,
    num_samples: int = 10
) -> List[np.ndarray]:
    """
    Retrieve a small number of images as numpy arrays for visualization.

    Args:
        region_path: Path to the region folder.
        num_samples: How many images to load.

    Returns:
        List of numpy arrays, each of shape (64, 64, 1).
    """
    pattern = os.path.join(region_path, "*.jpeg")
    image_paths = tf.data.Dataset.list_files(pattern, shuffle=True)
    images = image_paths.take(num_samples).map(
        load_image, num_parallel_calls=AUTOTUNE
    )
    return list(images.as_numpy_iterator())