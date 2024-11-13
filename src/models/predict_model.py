import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Paths
data_path = "/Users/home/Documents/github/convolutional_autoencoder/data/vbi_2d_healthy/acc_vehicle_data_dof_5.csv"
processed_data_path = "/Users/home/Documents/github/convolutional_autoencoder/data/processed/npy/healthy_accel_matrices.npy"
model_path = "/Users/home/Documents/github/convolutional_autoencoder/models/autoencoder_best_model.keras"


# Load the processed data
accel_matrices = np.load(processed_data_path)

# Load the trained autoencoder model
autoencoder = load_model(model_path)
print(f"Model loaded from {model_path}")

# Generate reconstructed signals
reconstructed_matrices = autoencoder.predict(accel_matrices)

def plot_input_vs_reconstructed(input_signal, reconstructed_signal, signal_index=0, title="Input vs. Reconstructed Signal"):
    """
    Plots the input signal and the reconstructed signal for comparison.
    
    Parameters:
    - input_signal (numpy.ndarray): The original input signal array.
    - reconstructed_signal (numpy.ndarray): The reconstructed signal array from the model.
    - signal_index (int): The index of the signal to plot (default is 0).
    - title (str): Title of the plot.
    """
    plt.figure(figsize=(12, 6))
    
    # Plot input signal
    plt.plot(input_signal[signal_index].flatten(), label="Input Signal", linewidth=1.5)
    
    # Plot reconstructed signal
    plt.plot(reconstructed_signal[signal_index].flatten(), label="Reconstructed Signal", linestyle='--', linewidth=1.5)
    
    # Plot styling
    plt.title(title)
    plt.xlabel("Time Steps")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()

# Choose a signal index to visualize
signal_index = 0  # You can change this to visualize different samples

# Plot the input signal and the reconstructed signal
plot_input_vs_reconstructed(accel_matrices, reconstructed_matrices, signal_index)