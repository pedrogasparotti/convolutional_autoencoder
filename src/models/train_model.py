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

# Define data path
data_path = r'/Users/home/Documents/github/convolutional_autoencoder/data/processed/npy/acc_vehicle_data_dof_baseline_train.npy'

def load_data(data_path):
    """
    Loads data from a .npy file

    Parameters:
    - data_path: str
        Path to the .npy file containing the data.

    Returns:
    - numpy.ndarray
         data.
    """
    # Load data
    data = np.load(data_path)

    return data

def scheduler(epoch, lr):
    """
    Learning rate scheduler function. Reduces learning rate by 10% every 10 epochs.
    """
    if epoch < 10:
        return lr
    else:
        return lr * 0.9

def alternative_scheduler(epoch, lr):
    """
    Learning rate scheduler function.
    Less aggressive reduction: reduces learning rate by 5% every 20 epochs after epoch 20.
    Maintains a minimum learning rate threshold.
    """
    initial_lr = 0.001  # Initial learning rate
    min_lr = 1e-5      # Minimum learning rate threshold
    
    if epoch < 20:
        return initial_lr
    else:
        # Reduce by 5% every 20 epochs
        decay = 0.95 ** ((epoch - 20) // 20)
        new_lr = initial_lr * decay
        return max(new_lr, min_lr) 

def plot_reconstruction(original, reconstructed, num_signals=5):
    """
    Plots the original vs. reconstructed signals for comparison.
    """
    plt.figure(figsize=(15, 8))
    for i in range(num_signals):
        original_signal = original[i].flatten()
        reconstructed_signal = reconstructed[i].flatten()
        
        plt.subplot(num_signals, 2, 2 * i + 1)
        plt.plot(original_signal, label="Original")
        plt.title(f"Original Signal {i+1}")
        plt.xlabel("Element Index")
        plt.ylabel("Amplitude")

        plt.subplot(num_signals, 2, 2 * i + 2)
        plt.plot(reconstructed_signal, label="Reconstructed")
        plt.title(f"Reconstructed Signal {i+1}")
        plt.xlabel("Element Index")
        plt.ylabel("Amplitude")
        
    plt.tight_layout()
    plt.show()

def main():
    # Load and preprocess data
    data = load_data(data_path)
    if len(data.shape) == 3:
        data = data[..., np.newaxis]
    
    # First split: separate test set (e.g., 20% of total data)
    x_train_val, x_test = train_test_split(data, test_size=0.2, random_state=42)
    
    # Second split: separate validation set from remaining data
    x_train, x_val = train_test_split(x_train_val, test_size=0.2, random_state=42)
    
    # Build and compile the model
    input_shape = (44, 44, 1)
    autoencoder = build_autoencoder_single_model(input_shape=input_shape)
    autoencoder = compile_autoencoder(autoencoder,
                                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
                                    loss='mae')

    # Plot the model architecture
    architecture_file = r'/Users/home/Documents/github/convolutional_autoencoder/models/autoencoder_architecture.png'
    plot_model_architecture(autoencoder, architecture_file=architecture_file)

    # Define callbacks
    model_save_path = r'/Users/home/Documents/github/convolutional_autoencoder/models/autoencoder_best_model.keras'
    callbacks = get_callbacks(model_path=model_save_path, patience=10)
    lr_scheduler = LearningRateScheduler(scheduler)
    callbacks.append(lr_scheduler)

    # Add TensorBoard callback
    log_dir = "logs/autoencoder"
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks.append(tensorboard_callback)

    # Train the model with the callbacks and learning rate scheduler
    history = autoencoder.fit(
        x_train, x_train,
        batch_size=32,
        epochs=100,
        validation_data=(x_val, x_val),
        callbacks=callbacks
    )

    # Save the training history
    history_path = r'/Users/home/Documents/github/convolutional_autoencoder/models/training_history.npy'
    np.save(history_path, history.history)
    print(f"Training history saved to {history_path}")
    
    test_loss = autoencoder.evaluate(x_test, x_test)
    print(f"Test set loss: {test_loss}")
    # Visualize reconstruction results
    reconstructed_signals = autoencoder.predict(x_val[:5])
    plot_reconstruction(x_val, reconstructed_signals, num_signals=5)

if __name__ == '__main__':
    main()