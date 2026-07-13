"""
WGAN-GP Data Augmentation for Thermal Response Temperature Dataset
=================================================================
This script implements a Wasserstein GAN with Gradient Penalty (WGAN-GP)
to generate synthetic samples for augmenting the thermal response 
temperature dataset of ultra-high temperature ceramics (UHTCs).

Requirements:
    - pandas
    - numpy
    - tensorflow >= 2.8
    - openpyxl

Usage:
    python wgan_data_augmentation.py

Output:
    - generated_thermal_response_samples.csv
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
import os

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# ============================================================================
# 1. Data Preprocessing
# ============================================================================

def load_and_preprocess_data(filepath: str) -> tuple:
    """
    Load raw data and extract features with their min-max ranges.

    Parameters
    ----------
    filepath : str
        Path to the Excel file containing thermal response data.
        Expected columns: CP, TC, Hr, Hv, MP, YM, GS, CE, EM, SVP, T

    Returns
    -------
    scaled_features : np.ndarray
        Min-max scaled feature matrix.
    feature_ranges : list
        List of (min, max) tuples for each feature.
    feature_columns : list
        Ordered list of feature column names.
    """
    data = pd.read_excel(filepath)

    feature_columns = ["CP", "TC", "Hr", "Hv", "MP", "YM", "GS", "CE", "EM", "SVP", "T"]

    # Validate columns exist
    missing = [c for c in feature_columns if c not in data.columns]
    if missing:
        raise ValueError(f"Missing columns in input file: {missing}. "
                         f"Available columns: {list(data.columns)}")

    features = data[feature_columns].values.astype(np.float32)

    # Compute min-max ranges from data
    feature_ranges = []
    for i, col in enumerate(feature_columns):
        min_val = float(np.min(features[:, i]))
        max_val = float(np.max(features[:, i]))
        # Add small epsilon to avoid division by zero for constant features
        if max_val - min_val < 1e-12:
            max_val = min_val + 1.0
        feature_ranges.append((min_val, max_val))

    # Min-max scaling to [0, 1]
    scaled = np.zeros_like(features, dtype=np.float32)
    for i, (min_val, max_val) in enumerate(feature_ranges):
        scaled[:, i] = (features[:, i] - min_val) / (max_val - min_val)

    return scaled, feature_ranges, feature_columns


def inverse_scale(data: np.ndarray, ranges: list) -> np.ndarray:
    """Inverse transform scaled data back to original scale."""
    inverse_scaled = np.zeros_like(data, dtype=np.float32)
    for i, (min_val, max_val) in enumerate(ranges):
        inverse_scaled[:, i] = data[:, i] * (max_val - min_val) + min_val
    return inverse_scaled


# ============================================================================
# 2. WGAN-GP Model Architecture
# ============================================================================

def build_generator(latent_dim: int = 10, output_dim: int = 11) -> keras.Model:
    """
    Build the Generator network.
    Maps latent noise vectors to synthetic data samples.
    """
    model = keras.Sequential([
        keras.layers.Dense(128, input_shape=(latent_dim,), activation='relu',
                           kernel_initializer='he_normal'),
        keras.layers.Dense(256, activation='relu', kernel_initializer='he_normal'),
        keras.layers.Dense(output_dim, activation='sigmoid',
                           kernel_initializer='glorot_uniform')
    ], name="Generator")
    return model


def build_critic(input_dim: int = 11) -> keras.Model:
    """
    Build the Critic (Discriminator) network.
    Outputs a scalar score for Wasserstein distance estimation.
    """
    model = keras.Sequential([
        keras.layers.Dense(128, input_shape=(input_dim,), activation='relu',
                           kernel_initializer='he_normal'),
        keras.layers.Dense(256, activation='relu', kernel_initializer='he_normal'),
        keras.layers.Dense(1, kernel_initializer='glorot_uniform')
    ], name="Critic")
    return model


# ============================================================================
# 3. Training Utilities
# ============================================================================

def gradient_penalty(critic: keras.Model, real: tf.Tensor, fake: tf.Tensor) -> tf.Tensor:
    """
    Compute gradient penalty for WGAN-GP as described in:
    Gulrajani et al., "Improved Training of Wasserstein GANs", NeurIPS 2017.
    """
    batch_size = tf.shape(real)[0]
    alpha = tf.random.uniform([batch_size, 1], 0.0, 1.0)
    diff = fake - real
    interpolated = real + alpha * diff

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = critic(interpolated, training=True)

    grads = gp_tape.gradient(pred, [interpolated])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1]) + 1e-8)
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp


@tf.function
def train_step(real_samples: tf.Tensor, generator: keras.Model, critic: keras.Model,
               g_opt: keras.optimizers.Optimizer, c_opt: keras.optimizers.Optimizer,
               latent_dim: int, gp_weight: float = 10.0) -> tuple:
    """
    Single training step for WGAN-GP.

    Returns
    -------
    gen_loss : tf.Tensor
        Generator loss (negative mean critic score on fake samples).
    critic_loss : tf.Tensor
        Critic loss (Wasserstein distance + gradient penalty).
    """
    batch_size = tf.shape(real_samples)[0]
    noise = tf.random.normal([batch_size, latent_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_samples = generator(noise, training=True)

        real_output = critic(real_samples, training=True)
        fake_output = critic(generated_samples, training=True)

        # Wasserstein losses
        gen_loss = -tf.reduce_mean(fake_output)
        critic_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

        # Gradient penalty
        gp = gradient_penalty(critic, real_samples, generated_samples)
        critic_loss += gp_weight * gp

    # Apply gradients
    gen_grads = gen_tape.gradient(gen_loss, generator.trainable_variables)
    critic_grads = disc_tape.gradient(critic_loss, critic.trainable_variables)

    g_opt.apply_gradients(zip(gen_grads, generator.trainable_variables))
    c_opt.apply_gradients(zip(critic_grads, critic.trainable_variables))

    return gen_loss, critic_loss


# ============================================================================
# 4. Main Training Loop
# ============================================================================

def train_wgan(scaled_data: np.ndarray, latent_dim: int = 10,
               epochs: int = 10000, batch_size: int = 128,
               critic_steps: int = 5, lr: float = 1e-4) -> keras.Model:
    """
    Train WGAN-GP on scaled data.

    Parameters
    ----------
    scaled_data : np.ndarray
        Min-max scaled training data.
    latent_dim : int
        Dimensionality of latent noise vector.
    epochs : int
        Total training epochs.
    batch_size : int
        Batch size for training.
    critic_steps : int
        Number of critic updates per generator update.
    lr : float
        Learning rate for Adam optimizer.

    Returns
    -------
    generator : keras.Model
        Trained generator model.
    """
    n_features = scaled_data.shape[1]
    generator = build_generator(latent_dim=latent_dim, output_dim=n_features)
    critic = build_critic(input_dim=n_features)

    g_opt = keras.optimizers.Adam(learning_rate=lr, beta_1=0.0, beta_2=0.9)
    c_opt = keras.optimizers.Adam(learning_rate=lr, beta_1=0.0, beta_2=0.9)

    dataset_size = scaled_data.shape[0]

    print(f"Training WGAN-GP: epochs={epochs}, batch_size={batch_size}, "
          f"critic_steps={critic_steps}, lr={lr}")
    print(f"Dataset size: {dataset_size}, Features: {n_features}, Latent dim: {latent_dim}")

    for epoch in range(epochs):
        # Train critic multiple times
        for _ in range(critic_steps):
            idx = np.random.randint(0, dataset_size, batch_size)
            real_batch = tf.convert_to_tensor(scaled_data[idx], dtype=tf.float32)
            _, d_loss = train_step(real_batch, generator, critic, g_opt, c_opt, latent_dim)

        # Train generator once
        idx = np.random.randint(0, dataset_size, batch_size)
        real_batch = tf.convert_to_tensor(scaled_data[idx], dtype=tf.float32)
        g_loss, _ = train_step(real_batch, generator, critic, g_opt, c_opt, latent_dim)

        if epoch % 100 == 0:
            print(f"Epoch {epoch:5d} | G Loss: {g_loss.numpy():.4f} | D Loss: {d_loss.numpy():.4f}")

    return generator


def generate_samples(generator: keras.Model, num_samples: int,
                     feature_ranges: list, latent_dim: int = 10) -> np.ndarray:
    """Generate synthetic samples using trained generator."""
    noise = tf.random.normal([num_samples, latent_dim])
    generated = generator(noise, training=False).numpy()
    return inverse_scale(generated, feature_ranges)


# ============================================================================
# 5. Entry Point
# ============================================================================

if __name__ == "__main__":
    # Configuration
    INPUT_FILE = "thermal_response_data.xlsx"  # Replace with actual filename
    OUTPUT_FILE = "generated_thermal_response_samples.csv"
    NUM_GENERATED = 1000
    EPOCHS = 10000
    BATCH_SIZE = 128
    LATENT_DIM = 10

    # Load data
    print(f"Loading data from {INPUT_FILE}...")
    scaled_features, feature_ranges, feature_columns = load_and_preprocess_data(INPUT_FILE)
    print(f"Loaded {scaled_features.shape[0]} samples with {scaled_features.shape[1]} features.")

    # Train WGAN-GP
    generator = train_wgan(scaled_features, latent_dim=LATENT_DIM,
                           epochs=EPOCHS, batch_size=BATCH_SIZE)

    # Generate synthetic samples
    print(f"\nGenerating {NUM_GENERATED} synthetic samples...")
    new_samples = generate_samples(generator, NUM_GENERATED, feature_ranges, LATENT_DIM)

    # Save results
    new_samples_df = pd.DataFrame(new_samples, columns=feature_columns)
    new_samples_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nGenerated samples saved to: {OUTPUT_FILE}")
    print("\nSample statistics:")
    for i, col in enumerate(feature_columns):
        print(f"  {col:6s}: min={new_samples[:, i].min():.2f}, "
              f"max={new_samples[:, i].max():.2f}, mean={new_samples[:, i].mean():.2f}")
