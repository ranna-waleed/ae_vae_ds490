import tensorflow as tf
import os

IMG_SIZE = 64
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE


def load_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=1)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    return img


def add_noise(img):
    noise = tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=0.1)
    noisy_img = tf.clip_by_value(img + noise, 0.0, 1.0)
    return noisy_img


def get_region_dataset(region_path, batch_size=BATCH_SIZE, for_denoising=False):
    """
    Loads all images from a single region folder.
    If for_denoising=True, returns (noisy_img, clean_img) pairs.
    Otherwise returns (img, img) pairs for standard AE/VAE training.
    """
    image_paths = tf.data.Dataset.list_files(os.path.join(region_path, "*.jpeg"), shuffle=True)

    images = image_paths.map(load_image, num_parallel_calls=AUTOTUNE)

    if for_denoising:
        dataset = images.map(lambda img: (add_noise(img), img), num_parallel_calls=AUTOTUNE)
    else:
        dataset = images.map(lambda img: (img, img), num_parallel_calls=AUTOTUNE)

    dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)
    return dataset


def get_all_regions(data_dir, batch_size=BATCH_SIZE):
    """
    Scans the dataset directory and returns a dict:
    { region_name: tf.data.Dataset }
    """
    regions = {}
    for folder_name in sorted(os.listdir(data_dir)):
        full_path = os.path.join(data_dir, folder_name)
        if os.path.isdir(full_path):
            regions[folder_name] = get_region_dataset(full_path, batch_size)
            print(f"  Loaded region: {folder_name}")
    return regions


def get_sample_images(region_path, num_samples=10):
    """Get a few images for visualization (as numpy arrays)."""
    image_paths = tf.data.Dataset.list_files(os.path.join(region_path, "*.jpeg"), shuffle=True)
    images = image_paths.take(num_samples).map(load_image, num_parallel_calls=AUTOTUNE)
    return list(images.as_numpy_iterator())