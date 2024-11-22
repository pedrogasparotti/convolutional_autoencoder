import numpy as np
import matplotlib.pyplot as plt

# Path to the normalized data file
data_path = '/Users/home/Documents/github/convolutional_autoencoder/data/processed/npy/healthy_acc_vehicle_data_dof_4.npy'

# Load the data
data = np.load(data_path)

# Check data shape
print(f"Data shape: {data.shape}")

# Select 5 signals to plot
num_signals = 10
selected_signals = data[:num_signals]

# Plot each signal after flattening
plt.figure(figsize=(12, 6))
for i, signal in enumerate(selected_signals):
    # Flatten the matrix and plot
    flattened_signal = signal.flatten()
    plt.plot(flattened_signal, label=f'Signal {i+1}')

# Labeling and legend
plt.title("Normalized Flattened Signals")
plt.xlabel("Element Index")
plt.ylabel("Normalized Amplitude")
plt.legend()
plt.show()
