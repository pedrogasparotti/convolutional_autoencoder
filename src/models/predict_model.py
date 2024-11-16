import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error
from scipy.stats import lognorm
import matplotlib.pyplot as plt
import os

def load_dae_model(model_path):
    """
    Loads the trained Deep Autoencoder (DAE) model.
    
    Parameters:
    - model_path (str): Path to the saved DAE model file.
    
    Returns:
    - model (tf.keras.Model): Loaded DAE model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"DAE model not found at {model_path}")
    model = load_model(model_path)
    print(f"DAE model loaded from {model_path}")
    return model

def calculate_maes(model, data, batch_size=32):
    """
    Calculate MAE values for a dataset using the autoencoder model.
    
    Parameters:
    model (tensorflow.keras.Model): Trained autoencoder model
    data (numpy.ndarray): Input data to evaluate
    batch_size (int): Batch size for predictions
    
    Returns:
    numpy.ndarray: Array of MAE values for each sample
    """
    # Ensure data has the correct number of dimensions
    if len(data.shape) == 3:  # (samples, height, width)
        data_input = np.expand_dims(data, axis=-1)
    else:
        data_input = data
    
    # Make predictions
    reconstructions = model.predict(data_input, batch_size=batch_size, verbose=0)
    
    # Ensure original data matches reconstruction dimensions for comparison
    if len(data.shape) == 3:
        reconstructions = np.squeeze(reconstructions, axis=-1)
    
    # Calculate MAE for each sample
    mae_values = np.mean(np.abs(data - reconstructions), axis=tuple(range(1, data.ndim)))
    
    return mae_values

def load_signals(healthy_path, anomalous_path):
    """
    Loads healthy and anomalous signal databases from .npy files.
    
    Parameters:
    - healthy_path (str): Path to the healthy signals .npy file.
    - anomalous_path (str): Path to the anomalous signals .npy file.
    
    Returns:
    - healthy_data (np.ndarray): Array of healthy signals.
    - anomalous_data (np.ndarray): Array of anomalous signals.
    """
    if not os.path.exists(healthy_path):
        raise FileNotFoundError(f"Healthy signals file not found at {healthy_path}")
    if not os.path.exists(anomalous_path):
        raise FileNotFoundError(f"Anomalous signals file not found at {anomalous_path}")
    
    healthy_data = np.load(healthy_path)
    anomalous_data = np.load(anomalous_path)
    
    print(f"Loaded {healthy_data.shape[0]} healthy signals and {anomalous_data.shape[0]} anomalous signals.")
    return healthy_data, anomalous_data

import numpy as np
from scipy import stats
from scipy.stats import lognorm
import matplotlib.pyplot as plt

def calculate_kl_divergence(mae_healthy, mae_damaged, num_points=1000):
    """
    Calculate KL divergence between two MAE populations assuming log-normal distributions.
    
    Parameters:
    mae_healthy (array-like): MAE values from healthy bridge condition
    mae_damaged (array-like): MAE values from damaged bridge condition
    num_points (int): Number of points to use in numerical integration
    
    Returns:
    float: KL divergence value
    dict: Distribution parameters and additional statistics
    """
    # Convert to numpy arrays and ensure positive values
    mae_healthy = np.array(mae_healthy)
    mae_damaged = np.array(mae_damaged)
    
    # Fit log-normal distributions to both datasets
    # For healthy condition
    shape_healthy, loc_healthy, scale_healthy = stats.lognorm.fit(mae_healthy)
    mu_healthy = np.log(scale_healthy)
    sigma_healthy = shape_healthy
    
    # For damaged condition
    shape_damaged, loc_damaged, scale_damaged = stats.lognorm.fit(mae_damaged)
    mu_damaged = np.log(scale_damaged)
    sigma_damaged = shape_damaged
    
    # Create points for numerical integration
    x = np.linspace(min(mae_healthy.min(), mae_damaged.min()),
                    max(mae_healthy.max(), mae_damaged.max()),
                    num_points)
    
    # Calculate PDFs
    pdf_healthy = stats.lognorm.pdf(x, shape_healthy, loc_healthy, scale_healthy)
    pdf_damaged = stats.lognorm.pdf(x, shape_damaged, loc_damaged, scale_damaged)
    
    # Remove zeros to avoid division by zero in log
    mask = (pdf_healthy > 0) & (pdf_damaged > 0)
    x = x[mask]
    pdf_healthy = pdf_healthy[mask]
    pdf_damaged = pdf_damaged[mask]
    
    # Calculate KL divergence numerically
    kl_div = np.sum(pdf_healthy * np.log(pdf_healthy / pdf_damaged)) * (x[1] - x[0])
    
    # Collect distribution parameters and statistics
    stats_dict = {
        'healthy_mu': mu_healthy,
        'healthy_sigma': sigma_healthy,
        'damaged_mu': mu_damaged,
        'damaged_sigma': sigma_damaged,
        'healthy_mean': np.mean(mae_healthy),
        'damaged_mean': np.mean(mae_damaged),
        'healthy_std': np.std(mae_healthy),
        'damaged_std': np.std(mae_damaged)
    }
    
    return kl_div, stats_dict

def plot_distributions(mae_healthy, mae_damaged, stats_dict, save_path=None):
    """
    Plot the fitted log-normal distributions for visual comparison.
    
    Parameters:
    mae_healthy (array-like): MAE values from healthy bridge condition
    mae_damaged (array-like): MAE values from damaged bridge condition
    stats_dict (dict): Dictionary containing distribution parameters
    save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Create points for plotting
    x = np.linspace(min(min(mae_healthy), min(mae_damaged)),
                    max(max(mae_healthy), max(mae_damaged)),
                    1000)
    
    # Plot histograms of actual data
    plt.hist(mae_healthy, bins=30, density=True, alpha=0.3, color='blue', label='Healthy (Data)')
    plt.hist(mae_damaged, bins=30, density=True, alpha=0.3, color='red', label='Damaged (Data)')
    
    plt.xlabel('MAE Values')
    plt.ylabel('Density')
    plt.title('Fitted Log-Normal Distributions of MAE Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

# Example usage in your main function:
def process_mae_distributions(healthy_maes, damaged_maes):
    """
    Process MAE distributions and calculate KL divergence.
    
    Parameters:
    healthy_maes (array-like): MAE values from healthy condition
    damaged_maes (array-like): MAE values from damaged condition
    
    Returns:
    float: KL divergence value
    dict: Distribution parameters and statistics
    """
    # Calculate KL divergence
    kl_div, stats = calculate_kl_divergence(healthy_maes, damaged_maes)
    
    # Plot distributions
    plot_distributions(healthy_maes, damaged_maes, stats)
    
    return kl_div, stats

def main():
    # Define paths
    model_path = r'/Users/home/Documents/github/convolutional_autoencoder/models/autoencoder_best_model.keras' 
    healthy_path = r'/Users/home/Documents/github/convolutional_autoencoder/data/processed/npy/healthy_accel_matrices_dof_4.npy'  # Replace with your healthy data path
    anomalous_path = r'/Users/home/Documents/github/convolutional_autoencoder/data/processed/npy/acc_vehicle_data_dof_4_DAMAGE.npy'  # Replace with your anomalous data path
    # Load the trained DAE model
    dae_model = load_dae_model(model_path)
    
    # Load healthy and anomalous signals
    healthy_data, anomalous_data = load_signals(healthy_path, anomalous_path)
    
    # Define the input shape based on the DAE model
    input_shape = dae_model.input_shape[1:]  # Exclude the batch dimension
    print(f"Model expects input shape: {input_shape}")
    
    # Calculate MAEs for both conditions
    healthy_maes = calculate_maes(dae_model, healthy_data)
    damaged_maes = calculate_maes(dae_model, anomalous_data)
    
    # Calculate KL divergence and plot distributions
    kl_div, stats = process_mae_distributions(healthy_maes, damaged_maes)
    
    print(f"KL Divergence: {kl_div:.4f}")
    print("\nDistribution Parameters:")
    for key, value in stats.items():
        print(f"{key}: {value:.4f}")

if __name__ == '__main__':
    main()