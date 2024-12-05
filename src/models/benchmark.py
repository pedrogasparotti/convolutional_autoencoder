import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Get the absolute path of the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import functions from autoencoder.py
from models.autoencoder import (
    build_autoencoder_single_model,
    compile_autoencoder,
    get_callbacks,
    plot_model_architecture
)

# Define paths for DOFs
BASE_PATH = '/Users/home/Documents/github/convolutional_autoencoder/data/processed/npy'
DOF_PATHS = {
    'DOF_6': f'{BASE_PATH}/acc_vehicle_data_dof_6_baseline_train.npy',
    'DOF_5': f'{BASE_PATH}/acc_vehicle_data_dof_5_baseline_train.npy',
    'DOF_4': f'{BASE_PATH}/acc_vehicle_data_dof_5_baseline_train.npy',
    'DOF_1': f'{BASE_PATH}/acc_vehicle_data_dof_5_baseline_train.npy'
}

class AnomalyDetector(Model):
    def __init__(self, input_dim):
        super(AnomalyDetector, self).__init__()
        dim1 = min(input_dim, 128)
        dim2 = min(dim1, 64)
        dim3 = min(dim2, 32)
        
        self.encoder = tf.keras.Sequential([
            layers.Dense(dim1, activation="relu"),
            layers.Dense(dim2, activation="relu"),
            layers.Dense(dim3, activation="relu")
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(dim2, activation="relu"),
            layers.Dense(dim1, activation="relu"),
            layers.Dense(input_dim, activation="sigmoid")
        ])
    
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def load_data(data_path):
    """Loads data from a .npy file."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found at {data_path}")
    return np.load(data_path)

def preprocess_data(data, flatten=False):
    """Normalize data between 0 and 1 and optionally flatten."""
    min_val = np.min(data)
    max_val = np.max(data)
    data = (data - min_val) / (max_val - min_val)
    if flatten:
        data = data.reshape(data.shape[0], -1)  # Flatten for dense layers
    return data

def train_anomaly_detector(data, input_dim):
    """Train the AnomalyDetector model."""
    model = AnomalyDetector(input_dim)
    model.compile(optimizer='adam', loss='mae', metrics=['accuracy'])
    
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

def train_convolutional_autoencoder(data, input_shape):
    """Train the Convolutional Autoencoder model."""
    autoencoder = build_autoencoder_single_model(input_shape=input_shape)
    autoencoder = compile_autoencoder(
        autoencoder,
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
        loss='mae'
    )
    
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    
    history = autoencoder.fit(
        train_data, train_data,
        batch_size=32,
        epochs=50,
        validation_data=(val_data, val_data),
        shuffle=True,
        verbose=1
    )
    return autoencoder, history

def compare_training_across_models(dof_paths):
    """Train and compare both models across DOFs."""
    histories = {'autoencoder': [], 'anomaly_detector': []}
    dof_labels = []
    
    for dof, data_path in dof_paths.items():
        print(f"\nTraining models for {dof}...")
        
        # Load and preprocess data
        data = load_data(data_path)
        conv_data = preprocess_data(data)
        dense_data = preprocess_data(data, flatten=True)
        
        # Train Convolutional Autoencoder
        autoencoder, auto_history = train_convolutional_autoencoder(conv_data, input_shape=(44, 44, 1))
        histories['autoencoder'].append(auto_history.history)
        
        # Train AnomalyDetector
        anomaly_detector, anomaly_history = train_anomaly_detector(dense_data, input_dim=dense_data.shape[1])
        histories['anomaly_detector'].append(anomaly_history.history)
        
        # Save DOF label
        dof_labels.append(dof)
    
    return histories, dof_labels

def plot_loss_comparison(histories, dof_labels):
    """
    Plot training and validation loss for both models across DOFs.
    Titles, labels, and legends are adjusted for better visualization in Portuguese (BR).
    """
    for model_name, model_histories in histories.items():
        plt.figure(figsize=(14, 10))
        for dof_label, history in zip(dof_labels, model_histories):
            plt.plot(history['loss'], label=f'{dof_label} - Treinamento')
            plt.plot(history['val_loss'], linestyle='--', label=f'{dof_label} - Validação')

        plt.title(f'Perda de Treinamento e Validação ({model_name})', fontsize=22, weight='bold')
        plt.xlabel('Épocas', fontsize=18, weight='bold')
        plt.ylabel('Perda (MAE)', fontsize=18, weight='bold')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(fontsize=14)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = f'{model_name}_comparacao_perda.png'
        plt.savefig(output_file)
        print(f'Plot salvo: {output_file}')
        plt.show()

if __name__ == '__main__':
    # Train and compare models across DOFs
    histories, dof_labels = compare_training_across_models(DOF_PATHS)
    
    # Plot loss comparisons for both models
    plot_loss_comparison(histories, dof_labels)