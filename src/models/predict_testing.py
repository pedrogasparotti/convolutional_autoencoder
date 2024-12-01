import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanAbsoluteError
from scipy.stats import lognorm
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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

def load_signals(path):
    """
    Loads healthy and anomalous signal databases from .npy files.
    
    Parameters:
    - healthy_path (str): Path to the healthy signals .npy file.
    - anomalous_path (str): Path to the anomalous signals .npy file.
    
    Returns:
    - healthy_data (np.ndarray): Array of healthy signals.
    - anomalous_data (np.ndarray): Array of anomalous signals.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Healthy signals file not found at {path}")
    
    data = np.load(path)
    
    print(f"Loaded {data.shape[0]} signals.")
    return data

def calculate_maes(model, data):
    """
    Calculate Mean Absolute Errors between the original data and the reconstructed data.
    """
    # Add channel dimension if missing
    if len(data.shape) == 3:
        data = np.expand_dims(data, axis=-1)  # Now data.shape is (batch_size, height, width, channels)
    
    reconstructed = model.predict(data)
    
    # Calculate MAEs
    maes = np.mean(np.abs(data - reconstructed), axis=(1, 2, 3))  # Mean over height, width, channels
    return maes

def calculate_maes_samples(model, data):
    """
    Calculate Mean Absolute Errors (MAEs) for autoencoder predictions.
    
    Parameters:
    model: tensorflow.keras.Model
        The trained autoencoder model
    data: numpy.ndarray
        Array of signals with shape (n_samples, 44, 44)
        
    Returns:
    numpy.ndarray
        Array of MAE values for each sample
    """
    # Make predictions
    predictions = model.predict(data, verbose=0)
    predictions = np.squeeze(predictions)

    # Calculate MAE for each sample
    # This will automatically handle the (44,44) dimensions for each sample
    maes = np.mean(np.abs(data - predictions), axis=(1,2))
    
    return maes

def fit_mae_distribution(maes):
    """
    Fit a log-normal distribution to the MAEs and return the parameters of the underlying normal distribution.
    
    Returns:
    mu (float): Mean of the underlying normal distribution (ln(scale)).
    sigma (float): Standard deviation (shape parameter) of the underlying normal distribution.
    """
    # Ensure MAEs are positive
    maes = np.maximum(maes, 1e-9)
    
    # Fit a log-normal distribution with fixed loc=0
    shape, loc, scale = stats.lognorm.fit(maes, floc=0)
    
    # Parameters of the underlying normal distribution
    sigma = shape  # Shape parameter is the standard deviation of ln(MAEs)
    mu = np.log(scale)  # Mean of ln(MAEs)
    
    return mu, sigma

def sample_maes(maes, num_samples, sample_size):
    """
    Generate a list of samples by randomly sampling from MAEs.
    Each sample is a NumPy array of size 'sample_size'.
    """
    samples = []
    maes_array = np.array(maes)
    for _ in range(num_samples):
        # Ensure MAEs are positive
        maes_array = maes_array[maes_array > 0]
        sample = np.random.choice(maes_array, size=sample_size, replace=False)
        samples.append(sample)
    return samples

def calculate_sample_damage_indexes(samples, baseline_params):
    """
    Calculate damage indexes by comparing samples against the baseline distribution.
    """
    # Get distribution parameters for each sample
    sample_params = [stats.norm.fit(sample) for sample in samples]
    
    # Baseline distribution parameters
    baseline_mu, baseline_sigma = baseline_params 
    
    # Calculate KL divergence for samples vs baseline
    damage_indexes = [kl_divergence(mu, sigma, baseline_mu, baseline_sigma) 
                      for mu, sigma in sample_params]
    
    return np.array(damage_indexes)

def sample_mae_subsets(maes, num_subsets=60, samples_per_subset=10):
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
    samples = create_samples(maes)

    # Convert list of tensors into a stacked tensor for easy handling
    samples_tensor = tf.stack(samples)

    return samples_tensor

def plot_distribution_fit(data, dist, params, dist_name, sample_name):
    """
    Plot histogram of MAEs with fitted distribution.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot histogram of actual data
    plt.hist(data, bins='auto', density=True, alpha=0.7, 
             label=f'Actual {sample_name} MAEs')
    
    # Plot fitted distribution
    x = np.linspace(min(data), max(data), 100)
    plt.plot(x, dist.pdf(x, *params), 'r-', lw=2, 
             label=f'Fitted {dist_name} Distribution')
    
    plt.title(f'Distribution of {sample_name} MAEs')
    plt.xlabel('MAE')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def kl_divergence(mu1, sigma1, mu2, sigma2):
    """
    Calculate the KL divergence between two log-normal distributions.
    
    Parameters:
    mu1 (float): Mean of the first underlying normal distribution.
    sigma1 (float): Standard deviation of the first underlying normal distribution.
    mu2 (float): Mean of the second underlying normal distribution.
    sigma2 (float): Standard deviation of the second underlying normal distribution.
    
    Returns:
    float: The KL divergence from the first distribution to the second.
    """
    term1 = np.log(sigma2 / sigma1)
    term2 = (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2)
    return term1 + term2 - 0.5

# Calculate KL divergence for each pair of healthy and damaged distributions
def calculate_di(damaged_params, healthy_params):
    dis = []
    for (mu1, sigma1), (mu2, sigma2) in zip(healthy_params, damaged_params):
        # Convert log-normal parameters to normal parameters for underlying distributions
        kl_div = kl_divergence(mu1, sigma1, mu2, sigma2)
        dis.append(kl_div)
    return dis

def calculate_sample_damage_indexes(samples, baseline_params):
    """
    Calculate damage indexes by comparing samples against the baseline distribution.
    
    Parameters:
    samples: list of numpy.ndarray
        List of MAE samples to compare against baseline.
    baseline_params: tuple
        Parameters (mu, sigma) of the baseline distribution fitted to all healthy data.
    
    Returns:
    numpy.ndarray
        Array of damage indexes for the samples.
    """
    baseline_mu, baseline_sigma = baseline_params
    
    # Get distribution parameters for each sample
    sample_params = []
    for sample in samples:
        # Fit log-normal distribution to each sample
        mu, sigma = fit_mae_distribution(sample)
        sample_params.append((mu, sigma))
    
    # Calculate KL divergence for each sample vs baseline
    damage_indexes = [
        kl_divergence(mu, sigma, baseline_mu, baseline_sigma) 
        for mu, sigma in sample_params
    ]
    
    return np.array(damage_indexes)

def plot_lognormal_distribution(mu, sigma, data=None, bins=50, figsize=(10, 6)):
    """
    Plot the log-normal distribution with given parameters and optionally overlay histogram of data.
    
    Parameters:
    mu (float): Location parameter of the log-normal distribution
    sigma (float): Scale parameter of the log-normal distribution
    data (array-like, optional): Original data to overlay as histogram
    bins (int, optional): Number of bins for histogram
    figsize (tuple, optional): Figure size
    
    Returns:
    tuple: Figure and axes objects
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Generate points for the PDF curve
    x = np.linspace(0, max(data) if data is not None else np.exp(mu + 3*sigma), 1000)
    pdf = lognorm.pdf(x, sigma, scale=np.exp(mu))
    
    # Plot the PDF
    ax.plot(x, pdf, 'r-', lw=2, label='Log-normal PDF')
    
    # If data is provided, plot histogram
    if data is not None:
        ax.hist(data, bins=bins, density=True, alpha=0.5, color='blue', label='Data histogram')
    
    # Add labels and title
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title(f'Log-normal Distribution (μ={mu:.2f}, σ={sigma:.2f})')
    ax.legend()
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Tight layout to prevent label cutoff
    plt.tight_layout()
    
    return fig, ax

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

def load_val_signals(path):
    """
    Loads healthy and anomalous signal databases from .npy files.
    
    Parameters:
    - healthy_path (str): Path to the healthy signals .npy file.
    - anomalous_path (str): Path to the anomalous signals .npy file.
    
    Returns:
    - healthy_data (np.ndarray): Array of healthy signals.
    - anomalous_data (np.ndarray): Array of anomalous signals.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Healthy signals file not found at {path}")
    
    data = np.load(path)
    
    print(f"Loaded {data.shape[0]} signals.")
    return data

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

def plot_damage_indices(healthy_indices, damaged_indices, figsize=(12, 6)):
    """
    Create a scatter plot of damage indices for healthy and damaged samples.
    
    Parameters:
    healthy_indices: numpy.ndarray - Damage indices for healthy samples
    damaged_indices: numpy.ndarray - Damage indices for damaged samples
    figsize: tuple - Figure size (width, height)
    """
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create x-coordinates (just for visualization)
    healthy_x = np.random.normal(0, 0.1, len(healthy_indices))
    damaged_x = np.random.normal(1, 0.1, len(damaged_indices))
    
    # Create scatter plots
    plt.scatter(healthy_x, healthy_indices, c='blue', alpha=0.5, label='Healthy')
    plt.scatter(damaged_x, damaged_indices, c='red', alpha=0.5, label='Damaged')
    
    # Add mean lines
    plt.axhline(y=np.mean(healthy_indices), color='blue', linestyle='--', alpha=0.5)
    plt.axhline(y=np.mean(damaged_indices), color='red', linestyle='--', alpha=0.5)
    
    # Customize plot
    plt.title('Damage Indices: Healthy vs Damaged Samples')
    plt.xlabel('Sample Type')
    plt.ylabel('Damage Index')
    plt.xticks([0, 1], ['Healthy', 'Damaged'])
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add statistics annotation
    stats_text = f'Healthy: μ={np.mean(healthy_indices):.2f}, σ={np.std(healthy_indices):.2f}\n'
    stats_text += f'Damaged: μ={np.mean(damaged_indices):.2f}, σ={np.std(damaged_indices):.2f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return plt.gcf(), plt.gca()

def plot_damage_indices_three_sets(healthy_di, damaged_di, damaged_10pc_di, labels=['Healthy', 'Damaged', '10% Damaged']):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create box plots for all three datasets
    bp = ax.boxplot([healthy_di, damaged_di, damaged_10pc_di], 
                labels=labels,
                patch_artist=True)
    
    # Customize colors
    colors = ['lightgreen', 'lightcoral', 'lightskyblue']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Damage Index')
    ax.set_title('Damage Index Comparison')
    ax.grid(True)
    
    return fig, ax

def plot_multiple_roc_curves(healthy_di, damaged_di, damaged_10pc_di):
    plt.figure(figsize=(10, 6))
    
    # Calculate ROC curves for both damage scenarios
    y_true_five = ['Damaged'] * len(damaged_di) + ['Healthy'] * len(healthy_di)
    y_scores_five = damaged_di + healthy_di
    
    y_true_10pc = ['Damaged'] * len(damaged_10pc_di) + ['Healthy'] * len(healthy_di)
    y_scores_10pc = damaged_10pc_di + healthy_di
    
    # Plot ROC curve for Damage Five
    fpr_five, tpr_five, _ = roc_curve([1 if y == 'Damaged' else 0 for y in y_true_five], y_scores_five)
    roc_auc_five = auc(fpr_five, tpr_five)
    plt.plot(fpr_five, tpr_five, color='darkorange', lw=2, 
             label=f'Damage Five ROC (area = {roc_auc_five:.2f})')
    
    # Plot ROC curve for 10% Damage
    fpr_10pc, tpr_10pc, _ = roc_curve([1 if y == 'Damaged' else 0 for y in y_true_10pc], y_scores_10pc)
    roc_auc_10pc = auc(fpr_10pc, tpr_10pc)
    plt.plot(fpr_10pc, tpr_10pc, color='blue', lw=2, 
             label=f'10% Damage ROC (area = {roc_auc_10pc:.2f})')
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

def create_labeled_dataset(healthy_samples, damaged_5pc_samples, damaged_10pc_samples):
    """
    Create a labeled dataset where each sample consists of Mean Absolute Errors (MAEs)
    and an associated label indicating the condition (healthy, 5% damage, or 10% damage).
    
    Parameters:
    - healthy_samples: list of numpy.ndarray
        List of MAE samples for the healthy condition.
    - damaged_5pc_samples: list of numpy.ndarray
        List of MAE samples for the 5% damage condition.
    - damaged_10pc_samples: list of numpy.ndarray
        List of MAE samples for the 10% damage condition.
    
    Returns:
    - dataset: pandas.DataFrame
        A DataFrame containing MAE samples and their associated labels.
    """
    # Initialize lists to store data and labels
    data = []
    labels = []

    # Label mapping
    label_mapping = {'healthy': 0, '5% damage': 1, '10% damage': 2}

    # Process healthy samples
    for sample in healthy_samples:
        # Each sample is an array of MAEs

        mean_mae = np.mean(sample)
        data.append(mean_mae)
        labels.append(label_mapping['healthy'])

    # Process 5% damage samples
    for sample in damaged_5pc_samples:
        mean_mae = np.mean(sample)
        data.append(mean_mae)
        labels.append(label_mapping['5% damage'])

    # Process 10% damage samples
    for sample in damaged_10pc_samples:
        mean_mae = np.mean(sample)
        data.append(mean_mae)
        labels.append(label_mapping['10% damage'])

    # Create a DataFrame
    dataset = pd.DataFrame({'damage_indexes': data, 'Condition': labels})

    return dataset

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

def plot_damage_analysis(x_data, y_data, threshold=3):
    """Create the damage analysis plot with all components."""
    plt.figure(figsize=(12, 8))
    
    # Plot the main data points
    plt.scatter(x_data, y_data, color='black', alpha=0.5, s=30, label='Data points')
    
    # Generate and plot the trend line
    x_line = np.linspace(0, 10, 100)
    y_line = 0.5 * x_line + 1
    plt.plot(x_line, y_line, color='green', label='β₀', linewidth=2)
    
    # Plot the decision threshold
    plt.axhline(y=threshold, color='blue', linestyle='-', label='Decision threshold')
    
    # Generate and plot the false calls region
    x_false = np.linspace(0, 2, 100)
    y_false = 3 * np.exp(-x_false) + 1
    plt.plot(x_false, y_false, color='red', label='False calls')
    
    # Add annotations
    plt.annotate('ε', xy=(4, 2.5), xytext=(4, 2.5), color='green', fontsize=12)
    plt.annotate('β₁', xy=(6, 4), xytext=(6, 4), color='green', fontsize=12)
    
    # Add noise annotation
    plt.annotate('noise in the\nabsence of\ndiscontinuity', 
                xy=(1, 4), xytext=(1, 4),
                color='red', fontsize=10)
    
    # Create the ellipse around main data cluster
    from matplotlib.patches import Ellipse
    ellipse = Ellipse(xy=(6, 3.5), width=4, height=2, 
                     angle=-30, fill=False, color='black')
    plt.gca().add_patch(ellipse)
    
    # Labels and title
    plt.xlabel('discontinuity size (arbitrary units)')
    plt.ylabel('response (arbitrary units)')
    plt.title('Damage Index versus Damage Size')
    
    # Add FALSE CALLS text box
    plt.text(0.5, 1, 'FALSE CALLS', color='red', 
            bbox=dict(facecolor='white', edgecolor='red'))
    
    # Set plot limits and grid
    plt.xlim(-0.5, 10)
    plt.ylim(0, 6)
    plt.grid(True, alpha=0.3)
    
    return plt.gcf()

# Define paths  
model_path = r'/Users/home/Documents/github/convolutional_autoencoder/models/autoencoder_best_model.keras'  # trained model
baseline_path = r'/Users/home/Documents/github/convolutional_autoencoder/data/processed/npy/acc_vehicle_data_dof_6_baseline.npy'  # baseline trained data
healthy_path = r'/Users/home/Documents/github/convolutional_autoencoder/data/processed/npy/acc_vehicle_data_dof_6_baseline_val.npy'  # healthy cases novel data
anomalous_path_5pc = r'/Users/home/Documents/github/convolutional_autoencoder/data/processed/npy/acc_vehicle_data_dof_6_5pc.npy'  # 5% damage
anomalous_path_10pc = r'/Users/home/Documents/github/convolutional_autoencoder/data/processed/npy/acc_vehicle_data_dof_6_10pc_dmg.npy'  # 10% damage

# Load the trained DAE model
autoencoder_model = load_dae_model(model_path)

# Load baseline, healthy, and anomalous signals
baseline_data = load_signals(baseline_path)
healthy_data = load_signals(healthy_path)
anomalous_data_5pc = load_signals(anomalous_path_5pc)
anomalous_data_10pc = load_signals(anomalous_path_10pc)

def visualize_reconstruction(model, data, num_samples=5):
    indices = np.random.choice(len(data), num_samples, replace=False)
    original_samples = data[indices]
    reconstructed_samples = model.predict(original_samples)
    
    for i in range(num_samples):
        plt.figure(figsize=(8, 4))
        
        # Original
        plt.subplot(1, 2, 1)
        plt.imshow(original_samples[i].squeeze(), cmap='gray')
        plt.title('Original')
        
        # Reconstructed
        plt.subplot(1, 2, 2)
        plt.imshow(reconstructed_samples[i].squeeze(), cmap='gray')
        plt.title('Reconstructed')
        
        plt.show()

# Visualize for healthy data
plot_reconstruction(autoencoder_model, healthy_data)

# Visualize for damaged data
plot_reconstruction(autoencoder_model, anomalous_data_5pc)