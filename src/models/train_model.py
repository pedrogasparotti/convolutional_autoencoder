import os
import sys
import numpy as np
import tensorflow as tf
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
    'DOF_4': f'{BASE_PATH}/acc_vehicle_data_dof_4_baseline_train.npy',
    'DOF_1': f'{BASE_PATH}/acc_vehicle_data_dof_1_baseline_train.npy'
}

def load_data(data_path):
    """Loads data from a .npy file."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found at {data_path}")
    return np.load(data_path)

def scheduler(epoch, lr):
    """Learning rate scheduler function."""
    if epoch < 10:
        return lr
    else:
        return lr * 0.9

def train_and_evaluate_dof(dof, data_path):
    """Train and evaluate the autoencoder model for a specific DOF."""
    print(f"\nTraining for {dof}...")
    
    # Load and preprocess data
    data = load_data(data_path)
    if len(data.shape) == 3:
        data = data[..., np.newaxis]
    x_train, x_val = train_test_split(data, test_size=0.2, random_state=42)
    
    # Build and compile the model
    input_shape = (44, 44, 1)
    autoencoder = build_autoencoder_single_model(input_shape=input_shape)
    autoencoder = compile_autoencoder(
        autoencoder,
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
        loss='mae'
    )
    
    # Define callbacks
    model_save_path = f'/Users/home/Documents/github/convolutional_autoencoder/models/autoencoder_best_model_{dof}.keras'
    callbacks = get_callbacks(model_path=model_save_path, patience=10)
    lr_scheduler = LearningRateScheduler(scheduler)
    callbacks.append(lr_scheduler)

    log_dir = f"logs/autoencoder_{dof}"
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks.append(tensorboard_callback)

    # Train the model
    history = autoencoder.fit(
        x_train, x_train,
        batch_size=32,
        epochs=100,
        validation_data=(x_val, x_val),
        callbacks=callbacks
    )
    
    # Save training history
    history_path = f'/Users/home/Documents/github/convolutional_autoencoder/models/training_history_{dof}.npy'
    np.save(history_path, history.history)
    print(f"Training history for {dof} saved to {history_path}")
    
    # Visualize reconstruction
    reconstructed_signals = autoencoder.predict(x_val[:5])
    plot_reconstruction(x_val, reconstructed_signals, num_signals=5)
    
    return history.history

def plot_loss_comparison(histories, dof_labels):
    """Plot training and validation loss for all DOFs."""
    plt.figure(figsize=(12, 8))
    for dof, history in zip(dof_labels, histories):
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

def main():
    histories = []
    dof_labels = []

    for dof, data_path in DOF_PATHS.items():
        history = train_and_evaluate_dof(dof, data_path)
        histories.append(history)
        dof_labels.append(dof)

    # Plot loss comparison
    plot_loss_comparison(histories, dof_labels)

if __name__ == '__main__':
    main()
