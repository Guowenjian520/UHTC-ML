#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd  
import numpy as np  
import tensorflow as tf  
from tensorflow import keras  
import random  

# Set random seed  
seed = 0  
np.random.seed(seed)  
tf.random.set_seed(seed)  
random.seed(seed)  

# 1. Data preprocessing  
data = pd.read_excel('Thermal_Response_Temperature_data.xlsx')  
feature_columns = ["CP","TC","Hr","Hv","MP","YM","GS","CE","EM","SVP","T"]  
features = data[feature_columns].values  

# Custom scaling function  
def custom_scale(data, ranges):  
    scaled = np.zeros_like(data, dtype=np.float32)  
    for i, (min_val, max_val) in enumerate(ranges):  
        scaled[:, i] = (data[:, i] - min_val) / (max_val - min_val)  
    return scaled  

# Custom inverse scaling function  
def custom_inverse_scale(data, ranges):  
    inverse_scaled = np.zeros_like(data, dtype=np.float32)  
    for i, (min_val, max_val) in enumerate(ranges):  
        inverse_scaled[:, i] = data[:, i] * (max_val - min_val) + min_val  
    return inverse_scaled  

# Define the range for each feature (needs to be adjusted according to actual data)  
feature_ranges = [  
    (data['CP'].min(), data['CP'].max()),    # CP  
    (data['TC'].min(), data['TC'].max()),    # TC  
    (data['Hr'].min(), data['Hr'].max()),    # Hr   
    (data['Hv'].min(), data['Hv'].max()),    # Hv   
    (data['MP'].min(), data['MP'].max()),    # MP  
    (data['YM'].min(), data['YM'].max()),    # YM  
    (data['GS'].min(), data['GS'].max()),    # GS  
    (data['CE'].min(), data['CE'].max()),    # CE  
    (data['EM'].min(), data['EM'].max()),    # EM   
    (data['SVP'].min(), data['SVP'].max()),  # SVP  
    (data['T'].min(), data['T'].max())       # T  
]  

scaled_features = custom_scale(features, feature_ranges)  

# 2. Build GAN model  
# Modify input and output dimensions to 11 to match the number of features  
def build_generator():  
    model = keras.Sequential([  
        keras.layers.Dense(128, input_shape=(11,), activation='relu'),  
        keras.layers.Dense(256, activation='relu'),  
        keras.layers.Dense(11, activation='sigmoid')  
    ])  
    return model  

def build_critic():  
    model = keras.Sequential([  
        keras.layers.Dense(128, input_shape=(11,), activation='relu'),  
        keras.layers.Dense(256, activation='relu'),  
        keras.layers.Dense(1)  
    ])  
    return model  

generator = build_generator()  
critic = build_critic()  

# 3. Train GAN model  
def wasserstein_loss(y_true, y_pred):  
    return tf.reduce_mean(y_true * y_pred)  

def gradient_penalty(real, fake):  
    alpha = tf.random.uniform([real.shape[0], 1], 0., 1.)  
    diff = fake - real  
    interpolated = real + alpha * diff  
    with tf.GradientTape() as gp_tape:  
        gp_tape.watch(interpolated)  
        pred = critic(interpolated)  
    grads = gp_tape.gradient(pred, [interpolated])[0]  
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1]))  
    gp = tf.reduce_mean((norm - 1.0) ** 2)  
    return gp  

generator_optimizer = tf.keras.optimizers.Adam(1e-4)  
critic_optimizer = tf.keras.optimizers.Adam(1e-4)  

@tf.function  
def train_step(real_samples):  
    # Modify noise dimension to 11 to match the number of features  
    noise = tf.random.normal([real_samples.shape[0], 11])  
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:  
        generated_samples = generator(noise, training=True)  
        
        real_output = critic(real_samples, training=True)  
        fake_output = critic(generated_samples, training=True)  
        
        gen_loss = -tf.reduce_mean(fake_output)  
        critic_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)  
        
        gp = gradient_penalty(real_samples, generated_samples)  
        critic_loss += 9 * gp  
    
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)  
    gradients_of_critic = disc_tape.gradient(critic_loss, critic.trainable_variables)  
    
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))  
    critic_optimizer.apply_gradients(zip(gradients_of_critic, critic.trainable_variables))  
    
    return gen_loss, critic_loss  

# Training loop  
EPOCHS = 20000  
BATCH_SIZE = 32  

for epoch in range(EPOCHS):  
    for _ in range(5):  
        idx = np.random.randint(0, scaled_features.shape[0], BATCH_SIZE)  
        real_batch = scaled_features[idx]  
        _, d_loss = train_step(real_batch)  
    
    idx = np.random.randint(0, scaled_features.shape[0], BATCH_SIZE)  
    real_batch = scaled_features[idx]  
    g_loss, _ = train_step(real_batch)  
    
    if epoch % 100 == 0:  
        print(f"Epoch {epoch}, G Loss: {g_loss:.4f}, D Loss: {d_loss:.4f}")  

# 4. Generate new samples  
def generate_samples(num_samples):  
    # Modify noise dimension to 11 to match the number of features  
    noise = tf.random.normal([num_samples, 11])  
    generated = generator(noise).numpy()  
    return custom_inverse_scale(generated, feature_ranges)  

# Generate 1000 samples  
new_samples = generate_samples(1000)  

# 5. Output results  
print(f"Number of generated samples: {len(new_samples)}")  
if len(new_samples) > 0:  
    print("Statistical information of generated samples:")  
    for i, feature_name in enumerate(feature_columns):  
        print(f"{feature_name}: min = {new_samples[:, i].min():.2f}, max = {new_samples[:, i].max():.2f}, mean = {new_samples[:, i].mean():.2f}")  

# Save samples to CSV file  
new_samples_df = pd.DataFrame(new_samples, columns=feature_columns)  
new_samples_df.to_csv("generated_thermal_response_samples.csv", index=False)  
print("Samples have been saved to generated_thermal_response_samples.csv")

