import tensorflow as tf
from tensorflow.keras import layers, Model

# latent dim = 2 so we can plot the space directly without PCA
LATENT_DIM = 2


#  Autoencoder (AE)

def build_ae_encoder(latent_dim=LATENT_DIM):
    inputs = layers.Input(shape=(64, 64, 1), name="ae_encoder_input")
    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    z = layers.Dense(latent_dim, name="latent_vector")(x)
    return Model(inputs, z, name="ae_encoder")


def build_ae_decoder(latent_dim=LATENT_DIM):
    inputs = layers.Input(shape=(latent_dim,), name="ae_decoder_input")
    x = layers.Dense(16 * 16 * 64, activation="relu")(inputs)
    x = layers.Reshape((16, 16, 64))(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
    outputs = layers.Conv2DTranspose(1, 3, padding="same", activation="sigmoid", name="reconstructed")(x)
    return Model(inputs, outputs, name="ae_decoder")


class Autoencoder(Model):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = build_ae_encoder(latent_dim)
        self.decoder = build_ae_decoder(latent_dim)

    def call(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


#  Variational Autoencoder (VAE)

class SamplingLayer(layers.Layer):
    """Reparameterization trick: z = mu + epsilon * sigma"""

    def call(self, inputs):
        mu, log_var = inputs
        batch = tf.shape(mu)[0]
        dim = tf.shape(mu)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        # sigma = exp(0.5 * log_var)
        return mu + tf.exp(0.5 * log_var) * epsilon


def build_vae_encoder(latent_dim=LATENT_DIM):
    inputs = layers.Input(shape=(64, 64, 1), name="vae_encoder_input")
    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)

    mu = layers.Dense(latent_dim, name="mu")(x)
    log_var = layers.Dense(latent_dim, name="log_var")(x)
    z = SamplingLayer(name="z")([mu, log_var])

    return Model(inputs, [mu, log_var, z], name="vae_encoder")


def build_vae_decoder(latent_dim=LATENT_DIM):
    # Same architecture as AE decoder
    inputs = layers.Input(shape=(latent_dim,), name="vae_decoder_input")
    x = layers.Dense(16 * 16 * 64, activation="relu")(inputs)
    x = layers.Reshape((16, 16, 64))(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
    outputs = layers.Conv2DTranspose(1, 3, padding="same", activation="sigmoid", name="reconstructed")(x)
    return Model(inputs, outputs, name="vae_decoder")


class VAE(Model):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = build_vae_encoder(latent_dim)
        self.decoder = build_vae_decoder(latent_dim)

        # track losses separately for visualization
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.recon_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker]

    def call(self, x):
        mu, log_var, z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x):
        mu, log_var, z = self.encoder(x)
        return mu  # use mu as the latent representation

    def decode(self, z):
        return self.decoder(z)

    def train_step(self, data):
        # data is (x, x) from our dataset
        x, _ = data

        with tf.GradientTape() as tape:
            mu, log_var, z = self.encoder(x)
            reconstruction = self.decoder(z)

            # reconstruction loss (sum over pixels, mean over batch)
            recon_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(x, reconstruction),
                    axis=(1, 2),
                )
            )

            # KL divergence loss
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), axis=1)
            )

            total_loss = recon_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {m.name: m.result() for m in self.metrics}