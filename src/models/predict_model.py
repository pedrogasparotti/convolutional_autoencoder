import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error
import scipy.stats as stats
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

def calculate_maes(model, data, batch_size=32):
    """
    Calculate MAE values for a dataset using the autoencoder model with Keras built-in function.
    
    Parameters:
    model (tensorflow.keras.Model): Trained autoencoder model
    data (numpy.ndarray): Input data to evaluate
    batch_size (int): Batch size for predictions
    
    Returns:
    numpy.ndarray: Array of MAE values for each sample
    """
    
    # Make predictions
    reconstructions = model.predict(data, batch_size=batch_size, verbose=0)
    
    # Initialize Keras Mean Absolute Error function
    mae_function = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
    
    # Calculate MAE for each sample
    mae_values = mae_function(data, reconstructions)
    
    return mae_values

def sample_mae_subsets(healthy_maes, damaged_maes, num_subsets=60, samples_per_subset=10):
    """
    Generate random subsets of MAE values for both healthy and damaged conditions.

    Parameters:
    healthy_maes (tf.Tensor): Tensor of MAE values for healthy samples.
    damaged_maes (tf.Tensor): Tensor of MAE values for damaged samples.
    num_subsets (int): Number of subsets to generate.
    samples_per_subset (int): Number of samples per subset.

    Returns:
    tuple: Two tensors containing the subsets for healthy and damaged conditions.
    """
    # Function to create samples for a given condition
    def create_samples(maes):
        return [tf.gather(maes, tf.random.uniform([samples_per_subset], maxval=len(maes), dtype=tf.int32))
                for _ in range(num_subsets)]

    # Create subsets for healthy and damaged conditions
    healthy_samples = create_samples(healthy_maes)
    damaged_samples = create_samples(damaged_maes)

    # Convert list of tensors into a stacked tensor for easy handling
    healthy_samples_tensor = tf.stack(healthy_samples)
    damaged_samples_tensor = tf.stack(damaged_samples)

    return healthy_samples_tensor, damaged_samples_tensor

def plot_mae_distributions(healthy_subsets, damaged_subsets, bins=50):
    """
    Plot the distributions of MAE values for healthy and damaged subsets using histograms.

    Parameters:
    healthy_subsets (tf.Tensor): Tensor of MAE subsets for healthy samples.
    damaged_subsets (tf.Tensor): Tensor of MAE subsets for damaged samples.
    bins (int): Number of bins for histogram.
    """
    # Flatten the tensors to get all sample MAE values in a single array
    all_healthy_maes = tf.reshape(healthy_subsets, [-1]).numpy()
    all_damaged_maes = tf.reshape(damaged_subsets, [-1]).numpy()

    # Create figure and axis objects
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot histograms
    ax.hist(all_healthy_maes, bins=bins, alpha=0.6, label='Healthy', color='green')
    ax.hist(all_damaged_maes, bins=bins, alpha=0.6, label='Damaged', color='red')

    # Set title and labels
    ax.set_title('Distribution of MAE Values')
    ax.set_xlabel('Mean Absolute Error (MAE)')
    ax.set_ylabel('Frequency')
    
    # Add a legend
    ax.legend()

    # Show the plot
    plt.show()

def plot_sample_distributions(healthy_samples, damaged_samples, bins=50):
    """
    Plot the distributions of healthy and damaged samples using histograms in separate subplots.

    Parameters:
    healthy_samples (list of tf.Tensor): List of tensors containing the healthy samples.
    damaged_samples (list of tf.Tensor): List of tensors containing the damaged samples.
    bins (int): Number of bins for histogram.
    """
    # Convert list of tensors to numpy arrays for plotting
    all_healthy = np.concatenate([sample.numpy().flatten() for sample in healthy_samples])
    all_damaged = np.concatenate([sample.numpy().flatten() for sample in damaged_samples])

    # Create figure and subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot healthy samples distribution
    axes[0].hist(all_healthy, bins=bins, color='green', alpha=0.7)
    axes[0].set_title('Distribution of Healthy Samples')
    axes[0].set_ylabel('Frequency')
    
    # Plot damaged samples distribution
    axes[1].hist(all_damaged, bins=bins, color='red', alpha=0.7)
    axes[1].set_title('Distribution of Damaged Samples')
    axes[1].set_ylabel('Frequency')
    axes[1].set_xlabel('Mean Absolute Error (MAE)')

    # Tight layout to ensure no overlap
    plt.tight_layout()
    plt.show()

def plot_distribution_subplots(samples, title_prefix, color, bins=50):
    """
    Plot distributions of samples using histograms in multiple subplots on a single figure.

    Parameters:
    samples (list of tf.Tensor): List of tensors containing the samples for plotting.
    title_prefix (str): Prefix for subplot titles to distinguish healthy from damaged.
    color (str): Color of the histogram.
    bins (int): Number of bins for histogram.
    """
    num_samples = len(samples)
    cols = 2  # Number of columns in subplot grid
    rows = (num_samples + 1) // cols  # Calculate rows needed

    # Create figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 3), sharex=True, sharey=True)
    axes = axes.flatten()  # Flatten to simplify indexing

    # Plot each sample distribution
    for i, sample in enumerate(samples):
        data = sample.numpy().flatten()
        axes[i].hist(data, bins=bins, color=color, alpha=0.7)
        axes[i].set_title(f'{title_prefix} Sample {i+1}')

    # Hide unused subplots if any
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()

def fit_lognormal_parameters(samples):
    """
    Fits log-normal distribution parameters for a list of sample tensors.

    Parameters:
    samples (list of tf.Tensor): List of tensors containing the samples.

    Returns:
    list of tuple: Each tuple contains the mu (mean) and sigma (std deviation) of
                   the log-normal distribution for each sample.
    """
    parameters = []
    for sample in samples:
        # Flatten the sample tensor and take the natural log of the data
        data = np.log(sample.numpy().flatten() + 1e-9)  # add a small value to avoid log(0)

        # Fit a normal distribution to the log of the data
        mu, sigma = stats.norm.fit(data)

        # Store the parameters
        parameters.append((mu, sigma))

    return parameters

def kl_divergence(mu1, sigma1, mu2, sigma2):
    """
    Calculate the Kullback-Leibler divergence between two normal distributions.

    Parameters:
    mu1 (float): Mean of the first normal distribution.
    sigma1 (float): Standard deviation of the first normal distribution.
    mu2 (float): Mean of the second normal distribution.
    sigma2 (float): Standard deviation of the second normal distribution.

    Returns:
    float: The KL divergence from the second distribution to the first.
    """
    term1 = np.log(sigma2 / sigma1)
    term2 = (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2)
    return term1 + term2 - 0.5

# Calculate KL divergence for each pair of healthy and damaged distributions
def calculate_di(healthy_params, damaged_params):
    dis = []
    for (mu1, sigma1), (mu2, sigma2) in zip(healthy_params, damaged_params):
        # Convert log-normal parameters to normal parameters for underlying distributions
        kl_div = kl_divergence(mu1, sigma1, mu2, sigma2)
        dis.append(kl_div)
    return dis

def plot_lognormal_fits(samples, parameters, title_prefix):
    """
    Plots histograms of the sample data alongside their fitted log-normal distribution curves.

    Parameters:
    samples (list of tf.Tensor): List of tensors containing the samples.
    parameters (list of tuple): Each tuple contains the mu (mean) and sigma (std deviation)
                                of the log-normal distribution for each sample.
    title_prefix (str): Prefix for the plot titles to distinguish between healthy and damaged.
    """
    num_samples = len(samples)
    cols = 2  # Number of columns in subplot grid
    rows = (num_samples + 1) // cols  # Calculate rows needed

    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 4), sharex=True, sharey=True)
    axes = axes.flatten()  # Flatten to simplify indexing

    for i, (sample, (mu, sigma)) in enumerate(zip(samples, parameters)):
        # Flatten the sample tensor and plot histogram
        data = sample.numpy().flatten()
        axes[i].hist(data, bins=30, density=True, alpha=0.6, color='blue', label='Empirical Data')

        # Generate points for the log-normal distribution
        dist_space = np.linspace(min(data), max(data), 100)
        # Calculate the pdf of the log-normal distribution
        pdf = stats.lognorm.pdf(dist_space, sigma, scale=np.exp(mu))
        axes[i].plot(dist_space, pdf, 'r-', lw=2, label='Log-normal Fit')

        # Set title and labels
        axes[i].set_title(f'{title_prefix} Sample {i+1}')
        axes[i].legend()

    # Hide unused subplots if any
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def compare_lognormal_pairs(healthy_params, damaged_params):
    """
    Plots 10 graphs comparing healthy and damaged log-normal distributions, displayed in a 5x2 grid,
    with shaded areas under the curves.

    Parameters:
    healthy_params (list of tuple): Each tuple contains the mu and sigma for healthy samples.
    damaged_params (list of tuple): Each tuple contains the mu and sigma for damaged samples.
    """
    num_plots = 10  # Total number of plots
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))  # 5 columns x 2 rows layout

    # Flatten the axes array for easier indexing
    axes = axes.flatten()

    # Define the range of the data to plot
    x_values = np.linspace(0, 0.2, 100)  # Adjusted range as needed

    for i, ax in enumerate(axes):
        if i < num_plots:  # Only plot for the number of available parameters
            # Healthy distribution
            mu, sigma = healthy_params[i]
            pdf = stats.lognorm.pdf(x_values, sigma, scale=np.exp(mu))
            ax.plot(x_values, pdf, 'g-', label=f'Healthy Sample {i+1}')
            ax.fill_between(x_values, 0, pdf, color='green', alpha=0.3)  # Adding shaded area

            # Damaged distribution
            mu, sigma = damaged_params[i]
            pdf = stats.lognorm.pdf(x_values, sigma, scale=np.exp(mu))
            ax.plot(x_values, pdf, 'r-', label=f'Damaged Sample {i+1}')
            ax.fill_between(x_values, 0, pdf, color='red', alpha=0.3)  # Adding shaded area

            # Set titles and labels
            ax.set_title(f'Comparison of Sample {i+1}')
            ax.set_xlabel('Mean Absolute Error (MAE)')
            ax.set_ylabel('Probability Density')
            ax.legend()
        else:
            ax.axis('off')  # Turn off unused axes

    plt.tight_layout()
    plt.show()

def main():
    # Define paths
    model_path = r'/Users/home/Documents/github/convolutional_autoencoder/models/autoencoder_best_model.keras'
    healthy_path = r'/Users/home/Documents/github/convolutional_autoencoder/data/processed/npy/healthy_accel_matrices_dof_4.npy'
    anomalous_path = r'/Users/home/Documents/github/convolutional_autoencoder/data/processed/npy/acc_vehicle_data_dof_4_DAMAGE.npy'
    # Load the trained DAE model
    dae_model = load_dae_model(model_path)
    
    # Load healthy and anomalous signals
    healthy_data, anomalous_data = load_signals(healthy_path, anomalous_path)

    healthy_data_mae = calculate_maes(dae_model, healthy_data)
    anomalous_data_mae = calculate_maes(dae_model, anomalous_data)

    # Generate subsets for healthy and damaged MAEs
    healthy_subsets, damaged_subsets = sample_mae_subsets(healthy_data_mae, anomalous_data_mae)

    # Sample 60 random entries from each subset
    healthy_samples = [tf.gather(subset, tf.random.shuffle(tf.range(tf.shape(subset)[0]))[:10]) for subset in healthy_subsets]
    damaged_samples = [tf.gather(subset, tf.random.shuffle(tf.range(tf.shape(subset)[0]))[:10]) for subset in damaged_subsets]

    healthy_params = fit_lognormal_parameters(healthy_samples)
    damaged_params = fit_lognormal_parameters(damaged_samples)

    damage_indices = calculate_di(healthy_params, damaged_params)
    print("Damage Indices (KL Divergence):", damage_indices)
    
    # plot_lognormal_fits(healthy_samples, healthy_params, 'Healthy')
    # plot_lognormal_fits(damaged_samples, damaged_params, 'Damaged')

    # compare_lognormal_pairs(healthy_params, damaged_params)

    # plot_mae_distributions(healthy_samples, damaged_samples)

    # plot_distribution_subplots(healthy_samples, 'Healthy', 'green')
    # plot_distribution_subplots(damaged_samples, 'Damaged', 'red')

if __name__ == '__main__':
    main()