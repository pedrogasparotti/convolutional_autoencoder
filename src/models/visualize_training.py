import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras import layers, models, Input, regularizers
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from scipy.stats import lognorm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os
import pandas as pd

# Define paths for DOFs
BASE_PATH = '/Users/home/Documents/github/convolutional_autoencoder/data/processed/npy'
MODEL_PATH = '/Users/home/Documents/github/convolutional_autoencoder/models/autoencoder_best_model.keras'

DOF_PATHS = {
    'DOF_6': {
        'baseline': f'{BASE_PATH}/acc_vehicle_data_dof_6_baseline_train.npy',
        'healthy': f'{BASE_PATH}/acc_vehicle_data_dof_6_baseline_val.npy',
        'damage_5pc': f'{BASE_PATH}/acc_vehicle_data_dof_6_5pc.npy',
        'damage_10pc': f'{BASE_PATH}/acc_vehicle_data_dof_6_10pc.npy'
    },
    'DOF_5': {
        'baseline': f'{BASE_PATH}/acc_vehicle_data_dof_5_baseline_train.npy',
        'healthy': f'{BASE_PATH}/acc_vehicle_data_dof_5_baseline_val.npy',
        'damage_5pc': f'{BASE_PATH}/acc_vehicle_data_dof_5_5pc.npy',
        'damage_10pc': f'{BASE_PATH}/acc_vehicle_data_dof_5_10pc.npy'
    }
}

def build_autoencoder_single_model(input_shape=(44, 44, 1), latent_dim=128):
    inputs = Input(shape=input_shape, name='encoder_input')
    
    # Encoder
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    encoded = layers.MaxPooling2D((2, 2), name='encoded_layer')(x)
    
    # Decoder
    x = layers.Conv2DTranspose(256, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D((2, 2))(x)
    
    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D((2, 2))(x)
    
    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D((2, 2))(x)
    
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    return models.Model(inputs, decoded, name='Autoencoder')

def load_signals(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at {file_path}")
    return np.load(file_path)

def preprocess_data(data, input_shape=(44, 44, 1)):
    min_val = np.min(data)
    max_val = np.max(data)
    data = (data - min_val) / (max_val - min_val)
    data = data.reshape(-1, *input_shape)
    return tf.cast(data, tf.float32)

def train_autoencoder(data, input_shape=(44, 44, 1)):
    model = build_autoencoder_single_model(input_shape=input_shape, latent_dim=128)
    model.compile(optimizer='adam', loss='mae')
    
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    
    history = model.fit(
        train_data, train_data,
        epochs=50,
        batch_size=32,
        validation_data=(val_data, val_data),
        shuffle=True,
        verbose=1
    )
    return model, history

def compare_training_across_dofs(dof_paths):
    training_results = {}
    
    for dof, paths in dof_paths.items():
        print(f"\nTraining model for {dof}...")
        
        baseline_data = load_signals(paths['baseline'])
        baseline_data = preprocess_data(baseline_data, input_shape=(44, 44, 1))
        
        model, history = train_autoencoder(baseline_data)
        training_results[dof] = {
            'model': model,
            'history': history.history
        }
        
        plot_loss(history, dof)
    
    return training_results

def plot_loss(history, dof_label):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Training and Validation Loss - {dof_label}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MAE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{dof_label}_loss_plot.png')
    plt.show()

def plot_loss_comparison(training_results):
    plt.figure(figsize=(12, 8))
    for dof, result in training_results.items():
        history = result['history']
        plt.plot(history['loss'], label=f'{dof} - Training Loss')
        plt.plot(history['val_loss'], linestyle='--', label=f'{dof} - Validation Loss')
    
    plt.title('Training and Validation Loss Comparison Across DOFs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MAE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_loss_comparison.png')
    plt.show()

if __name__ == '__main__':
    # Train models and compare across DOFs
    training_results = compare_training_across_dofs(DOF_PATHS)
    
    # Plot the training and validation loss comparison
    plot_loss_comparison(training_results)
