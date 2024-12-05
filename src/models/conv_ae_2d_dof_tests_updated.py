import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# Define paths for DOFs
BASE_PATH = '/Users/home/Documents/github/convolutional_autoencoder/data/processed/npy'

DOF_PATHS = {
    'DOF_6': {
        'baseline': f'{BASE_PATH}/acc_vehicle_data_dof_6_baseline_train.npy',
        'healthy': f'{BASE_PATH}/acc_vehicle_data_dof_6_baseline_val.npy'
    },
    'DOF_5': {
        'baseline': f'{BASE_PATH}/acc_vehicle_data_dof_5_baseline_train.npy',
        'healthy': f'{BASE_PATH}/acc_vehicle_data_dof_5_baseline_val.npy'
    },
    'DOF_4': {
        'baseline': f'{BASE_PATH}/acc_vehicle_data_dof_4_baseline_train.npy',
        'healthy': f'{BASE_PATH}/acc_vehicle_data_dof_4_baseline_val.npy'
    },
    'DOF_1': {
        'baseline': f'{BASE_PATH}/acc_vehicle_data_dof_1_baseline_train.npy',
        'healthy': f'{BASE_PATH}/acc_vehicle_data_dof_1_baseline_val.npy'
    }
}

def load_signals(file_path):
    """Load signals from .npy file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at {file_path}")
    return np.load(file_path)

def preprocess_data(data):
    """Normalize data between 0 and 1."""
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val)

def train_autoencoder(data, input_shape):
    """Train the autoencoder model for the given data."""
    autoencoder = tf.keras.models.load_model('/Users/home/Documents/github/convolutional_autoencoder/models/autoencoder_best_model.keras')
    
    # Prepare data
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # Train the model
    history = autoencoder.fit(
        train_data, train_data,
        epochs=50,
        batch_size=32,
        validation_data=(val_data, val_data),
        shuffle=True,
        verbose=1
    )
    return history

def plot_loss_comparison(histories, dof_labels):
    """Plot training and validation loss for all DOFs."""
    plt.figure(figsize=(12, 8))
    for dof_label, history in zip(dof_labels, histories):
        plt.plot(history.history['loss'], label=f'{dof_label} - Training Loss')
        plt.plot(history.history['val_loss'], linestyle='--', label=f'{dof_label} - Validation Loss')
    
    plt.title('Training and Validation Loss Comparison Across DOFs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MAE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_loss_comparison.png')
    plt.show()

if __name__ == '__main__':
    histories = []
    dof_labels = []
    
    for dof, paths in DOF_PATHS.items():
        print(f"Processing {dof}...")
        
        # Load and preprocess data
        baseline_data = load_signals(paths['baseline'])
        baseline_data = preprocess_data(baseline_data)
        
        # Train model and save history
        history = train_autoencoder(baseline_data, input_shape=(baseline_data.shape[1], baseline_data.shape[2], 1))
        histories.append(history)
        dof_labels.append(dof)
    
    # Plot the training loss comparison across DOFs
    plot_loss_comparison(histories, dof_labels)
