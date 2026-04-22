"""
model.py
Defines the Autoencoder (AE) and Variational Autoencoder (VAE) architectures
used for representation learning on the Medical MNIST dataset.
"""

# Third-party
import tensorflow as tf
from tensorflow.keras import layers, Model

# Latent dimension = 2 allows direct 2D scatter plots without PCA
LATENT_DIM: int = 2


#  Autoencoder (AE)

def build_ae_encoder(latent_dim: int = LATENT_DIM) -> Model:
    """
    Build the encoder part of the Autoencoder.

    Uses two Conv2D layers for downsampling, then a Dense
    bottleneck layer to produce the latent vector.

    Args:
        latent_dim: Size of the latent space.

    Returns:
        Keras Model: input (64,64,1) -> latent vector (latent_dim,).
    """
    inputs = layers.Input(shape=(64, 64, 1), name="ae_encoder_input")
    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    z = layers.Dense(latent_dim, name="latent_vector")(x)
    return Model(inputs, z, name="ae_encoder")


def build_ae_decoder(latent_dim: int = LATENT_DIM) -> Model:
    """
    Build the decoder part of the Autoencoder.

    Mirrors the encoder: Dense layer -> reshape -> two Conv2DTranspose
    layers for upsampling back to original resolution.

    Args:
        latent_dim: Size of the latent space.

    Returns:
        Keras Model: latent vector (latent_dim,) -> image (64,64,1).
    """
    inputs = layers.Input(shape=(latent_dim,), name="ae_decoder_input")
    x = layers.Dense(16 * 16 * 64, activation="relu")(inputs)
    x = layers.Reshape((16, 16, 64))(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
    outputs = layers.Conv2DTranspose(
        1, 3, padding="same", activation="sigmoid", name="reconstructed"
    )(x)
    return Model(inputs, outputs, name="ae_decoder")


class Autoencoder(Model):
    """
    Standard Autoencoder model.

    Combines an encoder and decoder. Trained to minimize
    binary cross-entropy reconstruction loss.

    Attributes:
        latent_dim: Dimensionality of the latent space.
        encoder: Keras Model mapping input -> latent vector.
        decoder: Keras Model mapping latent vector -> reconstruction.
    """

    def __init__(self, latent_dim: int = LATENT_DIM) -> None:
        """
        Initialize the Autoencoder.

        Args:
            latent_dim: Size of the latent space.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = build_ae_encoder(latent_dim)
        self.decoder = build_ae_decoder(latent_dim)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Forward pass: encode then decode.

        Args:
            x: Input image batch.

        Returns:
            Reconstructed image batch.
        """
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x: tf.Tensor) -> tf.Tensor:
        """Encode input to latent vector."""
        return self.encoder(x)

    def decode(self, z: tf.Tensor) -> tf.Tensor:
        """Decode latent vector to image."""
        return self.decoder(z)


#  Variational Autoencoder (VAE)

class SamplingLayer(layers.Layer):
    """
    Reparameterization trick layer for the VAE.

    Samples z = mu + epsilon * exp(0.5 * log_var)
    where epsilon ~ N(0, I). This allows gradients to flow
    through the sampling operation during backpropagation.
    """

    def call(self, inputs: tuple) -> tf.Tensor:
        """
        Sample from the latent distribution.

        Args:
            inputs: Tuple of (mu, log_var) tensors.

        Returns:
            Sampled latent vector z.
        """
        mu, log_var = inputs
        batch = tf.shape(mu)[0]
        dim = tf.shape(mu)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return mu + tf.exp(0.5 * log_var) * epsilon


def build_vae_encoder(latent_dim: int = LATENT_DIM) -> Model:
    """
    Build the probabilistic encoder for the VAE.

    Outputs mu and log_var (parameters of the latent distribution)
    plus a sampled z via the reparameterization trick.

    Args:
        latent_dim: Size of the latent space.

    Returns:
        Keras Model: input -> [mu, log_var, z].
    """
    inputs = layers.Input(shape=(64, 64, 1), name="vae_encoder_input")
    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)

    mu = layers.Dense(latent_dim, name="mu")(x)
    log_var = layers.Dense(latent_dim, name="log_var")(x)
    z = SamplingLayer(name="z")([mu, log_var])

    return Model(inputs, [mu, log_var, z], name="vae_encoder")


def build_vae_decoder(latent_dim: int = LATENT_DIM) -> Model:
    """
    Build the decoder for the VAE.

    Same architecture as the AE decoder.

    Args:
        latent_dim: Size of the latent space.

    Returns:
        Keras Model: latent vector -> image (64,64,1).
    """
    inputs = layers.Input(shape=(latent_dim,), name="vae_decoder_input")
    x = layers.Dense(16 * 16 * 64, activation="relu")(inputs)
    x = layers.Reshape((16, 16, 64))(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
    outputs = layers.Conv2DTranspose(
        1, 3, padding="same", activation="sigmoid", name="reconstructed"
    )(x)
    return Model(inputs, outputs, name="vae_decoder")


class VAE(Model):
    """
    Variational Autoencoder model.

    Extends the standard AE with a probabilistic latent space.
    The total loss combines reconstruction loss and KL divergence,
    which regularizes the latent space toward N(0, I).

    Attributes:
        latent_dim: Dimensionality of the latent space.
        encoder: Probabilistic encoder outputting [mu, log_var, z].
        decoder: Decoder mapping z -> reconstruction.
    """

    def __init__(self, latent_dim: int = LATENT_DIM) -> None:
        """
        Initialize the VAE.

        Args:
            latent_dim: Size of the latent space.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = build_vae_encoder(latent_dim)
        self.decoder = build_vae_decoder(latent_dim)

        # Separate metric trackers for monitoring each loss component
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.recon_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self) -> list:
        """Return list of metrics tracked during training."""
        return [
            self.total_loss_tracker,
            self.recon_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Forward pass: encode then decode using sampled z.

        Args:
            x: Input image batch.

        Returns:
            Reconstructed image batch.
        """
        _, _, z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x: tf.Tensor) -> tf.Tensor:
        """Encode input and return mu as the latent representation."""
        mu, _, _ = self.encoder(x)
        return mu

    def decode(self, z: tf.Tensor) -> tf.Tensor:
        """Decode latent vector to image."""
        return self.decoder(z)

    def train_step(self, data: tuple) -> dict:
        """
        Custom training step computing combined VAE loss.

        The total loss = reconstruction loss + KL divergence.
        KL loss regularizes the latent space toward N(0, I).

        Args:
            data: Tuple of (input, target) from the dataset.

        Returns:
            Dictionary of loss values for logging.
        """
        x, _ = data

        with tf.GradientTape() as tape:
            mu, log_var, z = self.encoder(x)
            reconstruction = self.decoder(z)

            # Reconstruction loss: sum over pixels, mean over batch
            recon_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(x, reconstruction),
                    axis=(1, 2),
                )
            )

            # KL divergence: measures distance from N(0, I)
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + log_var - tf.square(mu) - tf.exp(log_var),
                    axis=1
                )
            )

            total_loss = recon_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {m.name: m.result() for m in self.metrics}