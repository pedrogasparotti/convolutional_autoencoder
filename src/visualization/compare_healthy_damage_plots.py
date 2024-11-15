import numpy as np
import matplotlib.pyplot as plt

# Paths to the normalized data files
healthy_data_path = '/Users/home/Documents/github/convolutional_autoencoder/data/processed/npy/healthy_accel_matrices.npy'
damaged_data_path = '/Users/home/Documents/github/convolutional_autoencoder/data/processed/npy/ten_pc_damage_accel_matrices.npy'

# Load the data
healthy_data = np.load(healthy_data_path)
damaged_data = np.load(damaged_data_path)

# Check data shapes
print(f"Healthy data shape: {healthy_data.shape}")
print(f"Damaged data shape: {damaged_data.shape}")

# Number of signals to plot
num_signals = 5  # Adjust as needed based on your dataset size

# Plot Healthy Signals
plt.figure(figsize=(12, 6))
for i in range(num_signals):
    # Flatten the matrix and plot
    flattened_signal = healthy_data[i].flatten()
    plt.plot(flattened_signal, label=f'Healthy Signal {i+1}')
    
# Labeling and legend for healthy signals
plt.title("Normalized Flattened Healthy Signals")
plt.xlabel("Element Index")
plt.ylabel("Normalized Amplitude")
plt.legend()
plt.show()

# Plot Damaged Signals
plt.figure(figsize=(12, 6))
for i in range(num_signals):
    # Flatten the matrix and plot
    flattened_signal = damaged_data[i].flatten()
    plt.plot(flattened_signal, label=f'Damaged Signal {i+1}')
    
# Labeling and legend for damaged signals
plt.title("Normalized Flattened Damaged Signals")
plt.xlabel("Element Index")
plt.ylabel("Normalized Amplitude")
plt.legend()
plt.show()
