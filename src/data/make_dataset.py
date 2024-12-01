import numpy as np
import pandas as pd
import os

def load_csv_data(file_path):
    """
    Loads data from a CSV file.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data as a DataFrame.
    """
    try:
        dados = pd.read_csv(file_path)
        dados_cut = dados.iloc[:, 550:]
        shape = dados_cut.shape
        
        print(f"Successfully loaded data from {file_path}, shape of data is {shape}")
        return dados_cut
    

    except FileNotFoundError as e:
        print(f"Error loading CSV file: {e}")
        return None

def normalize_matrices(matrices):
    """
    Normalizes the matrix data to the range [0, 1].

    Parameters:
        matrices (np.ndarray): 3D or 4D matrix data to normalize.

    Returns:
        np.ndarray: Normalized matrices within the range [0, 1].
    """
    min_val = matrices.min()
    max_val = matrices.max()
    normalized_matrices = (matrices - min_val) / (max_val - min_val)  # normalize between [0, 1]
    return normalized_matrices

def transform_to_matrices(dados, reshape_dims, normalize=True):
    """
    Transforms the data from a DataFrame to a 3D matrix format and normalizes if specified.

    Parameters:
        dados (pd.DataFrame): DataFrame containing the data.
        reshape_dims (tuple): Target shape for reshaping the data.
        normalize (bool): Whether to normalize the data to the range [0, 1].

    Returns:
        np.ndarray: 3D matrix of reshaped and optionally normalized data.
    """
    # Convert DataFrame to a NumPy array
    accel_data = dados.to_numpy()

    # Check if total size matches the target reshape dimensions
    total_elements = np.prod(reshape_dims)

    if accel_data.size < total_elements:
        print("Warning: Not enough data to fill the target shape. Padding with zeros.")
        # Pad with zeros if needed
        accel_data = np.pad(accel_data.flatten(), (0, total_elements - accel_data.size), mode='constant')

    elif accel_data.size > total_elements:
        print("Warning: Too much data. Trimming excess elements.")
        # Trim excess elements
        accel_data = accel_data.flatten()[:total_elements]

    # Reshape to the target dimensions
    accel_matrices = accel_data.reshape(reshape_dims)

    # Normalize if specified
    if normalize:
        accel_matrices = normalize_matrices(accel_matrices)

    return accel_matrices

def save_matrices(matrices, output_path, file_name = "acc_vehicle_data_dof_6_10pc_dmg.npy"):
    """
    Saves the 3D matrix data to a .npy file.

    Parameters:
        matrices (np.ndarray): 3D matrix data to save.
        output_path (str): Directory to save the .npy file.
        file_name (str): Name of the .npy file.
    """
    os.makedirs(output_path, exist_ok=True)
    file_path = os.path.join(output_path, file_name)
    np.save(file_path, matrices)
    print(f"Matrix data saved to {file_path}")

# Paths
file_path = "/Users/home/Documents/github/convolutional_autoencoder/data/vbi_2d_10pc/acc_vehicle_data_dof_6_10pc_dmg.csv"

output_path = "/Users/home/Documents/github/convolutional_autoencoder/data/processed/npy"

# Process
dados = load_csv_data(file_path)
reshape_dims=(599, 44, 44)

if dados is not None:

    accel_matrices = transform_to_matrices(dados, reshape_dims)

    save_matrices(accel_matrices, output_path)
