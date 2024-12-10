import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanAbsoluteError
from scipy.stats import lognorm
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import os
import pandas as pd
from sklearn.cluster import KMeans
import seaborn as sns

def generate_damage_index_report(damage_indices, conditions):
    """
    Generates a report on the fluctuations of damage indices for each class.
    
    Parameters:
    - damage_indices (numpy.ndarray): Array of damage indices.
    - conditions (numpy.ndarray): Array of condition labels corresponding to each damage index.
    
    Returns:
    - report_df (pd.DataFrame): DataFrame containing the statistical summary for each class.
    """
    # Create a DataFrame for analysis
    df = pd.DataFrame({'Damage_Index': damage_indices, 'Condition': conditions})
    
    # Group by condition and calculate descriptive statistics
    report = df.groupby('Condition')['Damage_Index'].agg([
        'mean', 'std', 'min', 'max', 
        lambda x: np.percentile(x, 25),  # Q1
        'median',
        lambda x: np.percentile(x, 75),  # Q3
    ]).rename(columns={
        '<lambda_0>': 'Q1',
        '<lambda_1>': 'Q3'
    })
    
    # Add interquartile range (IQR)
    report['IQR'] = report['Q3'] - report['Q1']
    
    # Reset index for better visualization
    report_df = report.reset_index()
    
    print("\nResumo Estatístico dos Índices de Dano por Classe:")
    print(report_df)
    
    # Plotting the summary
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Condition', y='Damage_Index', data=df, palette='Set2')
    plt.title('Distribuição dos Índices de Dano por Classe', fontsize=18, weight='bold')
    plt.xlabel('Classe', fontsize=14)
    plt.ylabel('Índice de Dano', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return report_df

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
    return healthy_data, anomalous_data

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

def calculate_maes(model, data):
    """Calculate mean absolute errors for the given data and model."""
    predictions = model.predict(data, verbose=0)
    # Reshape predictions to match the input data shape
    predictions = predictions[:, :, np.newaxis] if predictions.ndim == 2 else predictions
    # Calculate MAE along the required axes
    maes = np.mean(np.abs(data - predictions), axis=(1, 2))
    return maes

def fit_mae_distribution(maes, plot=False):
    """
    Fit a lognormal distribution to MAE values and extract mu and sigma parameters.
    
    Parameters:
    maes: numpy.ndarray
        Array of Mean Absolute Error values
        
    Returns:
    tuple:
        - mu (float): Location parameter of the underlying normal distribution
        - sigma (float): Scale parameter of the underlying normal distribution
        - shape (float): Shape parameter s of the fitted lognormal distribution
        - loc (float): Location parameter of the fitted lognormal distribution
        - scale (float): Scale parameter of the fitted lognormal distribution
    """
    # Fit the lognormal distribution to the MAE values
    # The lognorm.fit function returns (s, loc, scale) where s is the shape parameter
    shape, loc, scale = stats.lognorm.fit(maes, floc=0)  # floc=0 fixes location to 0
    
    # Convert lognormal parameters to normal distribution parameters
    # For lognormal distribution:
    # sigma = shape
    # mu = log(scale)
    sigma = shape
    mu = np.log(scale)
    
    # Calculate goodness of fit using Kolmogorov-Smirnov test
    ks_statistic, p_value = stats.kstest(maes, 'lognorm', args=(shape, loc, scale))
    
    return mu, sigma

def plot_distribution_fit(data, dist, params):
    """
    Plot histogram of MAEs with fitted distribution.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot histogram of actual data
    plt.hist(data, bins='auto', density=True, alpha=0.7, label='Actual MAEs')
    
    # Plot fitted distribution
    x = np.linspace(min(data), max(data), 100)
    plt.plot(x, dist.pdf(x, *params), 'r-', lw=2, label=f'Fitted {dist.name}')
    
    plt.title('Distribution of Healthy Signal MAEs')
    plt.xlabel('MAE')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def stratified_sampling(data, sample_size, bins=10, random_state=42):
    """
    Perform stratified sampling to ensure a representative subset.
    
    Parameters:
        data (numpy.ndarray): Input data.
        sample_size (int): Number of samples to draw.
        bins (int): Number of bins for stratification.
        random_state (int): Random seed for reproducibility.
        
    Returns:
        numpy.ndarray: A representative sample of the data.
    """
    rng = np.random.default_rng(random_state)
    
    # Bin the data into equal intervals
    bin_edges = np.linspace(np.min(data), np.max(data), bins + 1)
    digitized = np.digitize(data, bin_edges)
    
    # Proportional sampling from each bin
    sampled_data = []
    for b in range(1, bins + 1):
        bin_data = data[digitized == b]
        if len(bin_data) > 0:
            bin_sample_size = int(sample_size * (len(bin_data) / len(data)))
            bin_sample = rng.choice(bin_data, size=bin_sample_size, replace=False)
            sampled_data.append(bin_sample)
    
    return np.concatenate(sampled_data)


def remove_outliers_iqr(data):
    """
    Remove outliers from data using the IQR method.
    
    Parameters:
        data (numpy.ndarray): Input data.
        
    Returns:
        numpy.ndarray: Data with outliers removed.
    """
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    
    # Define bounds for non-outlier data
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter the data
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
    return filtered_data

def sample_mae_subsets(maes, num_subsets=20, samples_per_subset=30):
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

def analyze_individual_distributions(mae_samples):
    """
    Analyze each sample of MAEs separately to get their distribution parameters.
    
    Parameters:
    mae_samples: list of numpy.ndarray
        List of samples, where each sample contains MAE values
        
    Returns:
    list of tuples
        List of (mu, sigma) pairs for each sample
    """
    distributions_params = []
    
    for sample in mae_samples:
        # Fit normal distribution to this sample
        mu, sigma = stats.norm.fit(sample)
        distributions_params.append((mu, sigma))
    
    return distributions_params

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

def calculate_sample_damage_indexes(samples, baseline_params):
    """
    Calculate damage indexes by comparing any set of samples against the baseline distribution.
    
    Parameters:
    samples: list of numpy.ndarray
        List of MAE samples to compare against baseline
    baseline_params: tuple
        Parameters of the baseline distribution fitted to all healthy data
    baseline_dist: scipy.stats distribution
        The baseline distribution object
        
    Returns:
    numpy.ndarray
        Array of damage indexes for the samples
    """

    # Get distribution parameters for each sample
    sample_params = [stats.norm.fit(sample) for sample in samples]
    
    # Convert baseline parameters to normal distribution parameters
    baseline_mu, baseline_sigma = baseline_params 
    
    # Calculate KL divergence for samples vs baseline
    damage_indexes = [kl_divergence(mu, sigma, baseline_mu, baseline_sigma) 
                     for mu, sigma in sample_params]
    
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

# Alternative version with sequential scatter plot
def plot_mae_distribution_scatter_alltogether(healthy_maes, anomalous_maes_5pc, anomalous_maes_10pc, baseline_maes, figsize=(12, 8)):
    """
    Create sequential scatter plots of all MAE values.
    
    Parameters:
        healthy_maes (np.array): MAEs from healthy data
        anomalous_maes_5pc (np.array): MAEs from 5% damage data
        anomalous_maes_10pc (np.array): MAEs from 10% damage data
        baseline_maes (np.array): MAEs from baseline data
        figsize (tuple): Figure size (width, height)
    """
    plt.figure(figsize=figsize)
    
    datasets = {
        'Healthy': (healthy_maes, 'green'),
        'Anomalous 5%': (anomalous_maes_5pc, 'red'),
        'Anomalous 10%': (anomalous_maes_10pc, 'red'),
        'Baseline': (baseline_maes, 'red')
    }
    
    # Create scatter plots
    for label, (data, color) in datasets.items():
        indices = np.arange(len(data))
        plt.scatter(indices, data, 
                   alpha=0.5, 
                   c=color, 
                   label=f'{label} MAEs',
                   s=30)
    
    plt.title('Sequential MAE Values')
    plt.xlabel('Sample Index')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
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

def plot_damage_indices(healthy_indices, damaged_indices, figsize=(12, 6)):
    """
    Cria um gráfico de dispersão dos índices de dano para amostras saudáveis e danificadas.

    Parâmetros:
    healthy_indices: numpy.ndarray - Índices de dano para amostras saudáveis.
    damaged_indices: numpy.ndarray - Índices de dano para amostras danificadas.
    figsize: tuple - Tamanho da figura (largura, altura).
    """
    plt.figure(figsize=figsize)

    # Cria coordenadas x (apenas para visualização)
    healthy_x = np.random.normal(0, 0.1, len(healthy_indices))
    damaged_x = np.random.normal(1, 0.1, len(damaged_indices))

    # Gráficos de dispersão
    plt.scatter(healthy_x, healthy_indices, c='blue', alpha=0.5, label='Saudável')
    plt.scatter(damaged_x, damaged_indices, c='red', alpha=0.5, label='Danificado')

    # Adiciona linhas de média
    plt.axhline(y=np.mean(healthy_indices), color='blue', linestyle='--', alpha=0.5)
    plt.axhline(y=np.mean(damaged_indices), color='red', linestyle='--', alpha=0.5)

    # Personalização do gráfico
    plt.title('Índices de Dano: Saudável vs Danificado', fontsize=18, weight='bold')
    plt.xlabel('Tipo de Amostra', fontsize=14)
    plt.ylabel('Índice de Dano', fontsize=14)
    plt.xticks([0, 1], ['Saudável', 'Danificado'], fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)

    # Adiciona estatísticas
    stats_text = f'Saudável: μ={np.mean(healthy_indices):.2f}, σ={np.std(healthy_indices):.2f}\n'
    stats_text += f'Danificado: μ={np.mean(damaged_indices):.2f}, σ={np.std(damaged_indices):.2f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    return plt.gcf(), plt.gca()

def plot_roc_curve(healthy_di, damaged_di):
    """
    Plota a curva ROC para classificação baseada no índice de dano.

    Parâmetros:
    healthy_di: numpy.ndarray
        Índices de dano para amostras saudáveis.
    damaged_di: numpy.ndarray
        Índices de dano para amostras danificadas.
    """
    # Combina índices de dano e cria rótulos
    scores = np.concatenate([healthy_di, damaged_di])
    labels = np.concatenate([np.zeros(len(healthy_di)), np.ones(len(damaged_di))])

    # Calcula a curva ROC
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)  # Área sob a curva ROC

    # Plota a curva ROC
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Linha diagonal
    plt.title("Curva ROC: Saudável vs Danificado", fontsize=18, weight='bold')
    plt.xlabel("Taxa de Falsos Positivos (FPR)", fontsize=14)
    plt.ylabel("Taxa de Verdadeiros Positivos (TPR)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.show()


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

def create_dataset_from_damage_indices(healthy_di, damage_5pc_di, damaged_10pc_di):
    """
    Create a labeled dataset from damage indexes.

    Parameters:
    healthy_di: numpy.ndarray
        Array of damage indexes for healthy samples.
    damage_5pc_di: numpy.ndarray
        Array of damage indexes for 5% damage samples.
    damaged_10pc_di: numpy.ndarray
        Array of damage indexes for 10% damage samples.

    Returns:
    dataset: pandas.DataFrame
        A DataFrame containing damage indexes and their associated labels.
    """
    import pandas as pd

    # Create DataFrames for each condition
    df_healthy = pd.DataFrame({'Damage_Index': healthy_di, 'Condition': 'Healthy'})
    df_5pc = pd.DataFrame({'Damage_Index': damage_5pc_di, 'Condition': '5% Damage'})
    df_10pc = pd.DataFrame({'Damage_Index': damaged_10pc_di, 'Condition': '10% Damage'})

    # Combine all DataFrames
    dataset = pd.concat([df_healthy, df_5pc, df_10pc], ignore_index=True)

    # Map conditions to numerical labels
    label_mapping = {'Healthy': 0, '5% Damage': 1, '10% Damage': 2}
    dataset['Label'] = dataset['Condition'].map(label_mapping)

    return dataset

def plot_mae_distribution_scatter(baseline_maes, healthy_maes, anomalous_maes_5pc, anomalous_maes_10pc, figsize=(12, 8)):
    """
    Create a vertical scatter plot showing the distribution of all MAE values.
    
    Parameters:
        healthy_maes (np.array): MAEs from healthy data
        anomalous_maes_5pc (np.array): MAEs from 5% damage data
        anomalous_maes_10pc (np.array): MAEs from 10% damage data
        baseline_maes (np.array): MAEs from baseline data
        figsize (tuple): Figure size (width, height)
    """
    plt.figure(figsize=figsize)
    
    # Define datasets with their properties
    datasets = {
        'Baseline': (baseline_maes, 'green', 1),
        'Healthy': (healthy_maes, 'blue', 2),
        'Anomalous 5%': (anomalous_maes_5pc, 'orange', 3),
        'Anomalous 10%': (anomalous_maes_10pc, 'red', 4)
    }
    
    # Create scatter plots for each dataset
    for label, (data, color, x_pos) in datasets.items():
        # Create jittered x positions
        x_jittered = np.random.normal(x_pos, 0.04, size=len(data))
        
        plt.scatter(x_jittered, data, 
                   alpha=0.5, 
                   c=color, 
                   label=f'{label} MAEs',
                   s=30)
    
    plt.title('Distribution of MAE Values')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.xlabel('Datasets')
    
    # Set x-ticks at the center of each distribution
    plt.xticks([1, 2, 3, 4], ['Baseline', 'Healthy', 'Anomalous 5%', 'Anomalous 10%'])
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.show()


def plot_all_mae_distributions(healthy_maes, anomalous_maes_5pc, anomalous_maes_10pc, baseline_maes, dist=stats.lognorm):
    """
    Plot histograms and fitted distributions for all MAE datasets.
    
    Parameters:
        healthy_maes (np.array): MAEs from healthy data
        anomalous_maes_5pc (np.array): MAEs from 5% damage data
        anomalous_maes_10pc (np.array): MAEs from 10% damage data
        baseline_maes (np.array): MAEs from baseline data
        dist (scipy.stats distribution): Distribution to fit (default: lognorm)
    """
    plt.figure(figsize=(12, 8))
    
    # Define data sets and their properties
    datasets = {
        'Healthy': (healthy_maes, 'blue', 0.3),
        #'Anomalous 5%': (anomalous_maes_5pc, 'orange', 0.3),
        #'Anomalous 10%': (anomalous_maes_10pc, 'red', 0.3),
        'Baseline': (baseline_maes, 'green', 0.3)
    }
    
    # Plot histograms and fitted distributions for each dataset
    for label, (data, color, alpha) in datasets.items():
        # Fit distribution
        params = dist.fit(data)
        
        # Plot histogram
        plt.hist(data, bins='auto', density=True, alpha=alpha, 
                color=color, label=f'{label} MAEs')
        
        # Plot fitted distribution
        x = np.linspace(min(data), max(data), 100)
        plt.plot(x, dist.pdf(x, *params), '-', 
                color=color, lw=2, label=f'Fitted {label}')
    
    plt.title('Distribution Comparison of MAEs')
    plt.xlabel('Mean Absolute Error (MAE)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add log scale for better visualization if needed
    plt.yscale('log')
    
    # Adjust layout to prevent label clipping
    plt.tight_layout()
    plt.show()
    
    # Print distribution parameters
    print("\nDistribution Parameters:")
    for label, (data, _, _) in datasets.items():
        params = dist.fit(data)
        print(f"{label}: {params}")

def normalize_damage_indexes(damage_indices):
    min_val, max_val = np.min(damage_indices), np.max(damage_indices)
    range_min, range_max = (0,1)
    normalized = (damage_indices - min_val) / (max_val - min_val)
    return normalized * (range_max - range_min) + range_min

def preprocess_for_autoencoder(data_path):
    """Load, flatten, and reshape data to fit the autoencoder's input requirements."""
    # Load the data from the given path
    data = np.load(data_path)  # Ensure the file is a .npy or compatible format
    # Flatten each sample from (44, 44) to (1936,)
    flattened_data = data.reshape(data.shape[0], -1)
    # Reshape to (599, 1936, 1)
    reshaped_data = flattened_data[:, :, np.newaxis]
    return reshaped_data

def main():
    # Define paths
    model_path = r'/Users/home/Documents/github/convolutional_autoencoder/models/best_model_dof_4_CONV1D.keras'
    healthy_path = r'/Users/home/Documents/github/convolutional_autoencoder/data/processed/npy/acc_vehicle_data_dof_4_val.npy'
    baseline_path = r'/Users/home/Documents/github/convolutional_autoencoder/data/processed/npy/acc_vehicle_data_dof_4_baseline.npy'
    anomalous_path_5pc = r'/Users/home/Documents/github/convolutional_autoencoder/data/processed/npy/acc_vehicle_data_dof_4_5pc.npy'
    anomalous_path_10pc = r'/Users/home/Documents/github/convolutional_autoencoder/data/processed/npy/acc_vehicle_data_dof_4_10pc.npy'

    # Load the trained DAE model
    autoencoder_model = load_dae_model(model_path)

    # Load baseline, healthy, and anomalous signals
    baseline_data = preprocess_for_autoencoder(baseline_path)
    healthy_data = preprocess_for_autoencoder(healthy_path)
    anomalous_data_5pc = preprocess_for_autoencoder(anomalous_path_5pc)
    anomalous_data_10pc = preprocess_for_autoencoder(anomalous_path_10pc)

    # Calculate MAEs for all datasets
    healthy_maes = calculate_maes(autoencoder_model, healthy_data)
    anomalous_maes_5pc = calculate_maes(autoencoder_model, anomalous_data_5pc)
    anomalous_maes_10pc = calculate_maes(autoencoder_model, anomalous_data_10pc)
    baseline_maes = calculate_maes(autoencoder_model, baseline_data)

    # Generate subsets for all datasets
    healthy_subsets = sample_mae_subsets(healthy_maes)
    damaged_subsets_5pc = sample_mae_subsets(anomalous_maes_5pc)
    damaged_10pc_subsets = sample_mae_subsets(anomalous_maes_10pc)

    sample_size = 30

    # Remove outliers from baseline MAEs
    filtered_baseline_maes = remove_outliers_iqr(baseline_maes)

    # Generate a representative subset after removing outliers
    representative_baseline = stratified_sampling(filtered_baseline_maes, sample_size)

    # Fit baseline distribution
    baseline_params = fit_mae_distribution(representative_baseline, plot=False)

    # Sample from each subset
    healthy_samples = [tf.gather(subset, tf.random.shuffle(tf.range(tf.shape(subset)[0]))[:10]) 
                      for subset in healthy_subsets]
    
    damaged_samples = [tf.gather(subset, tf.random.shuffle(tf.range(tf.shape(subset)[0]))[:10]) 
                      for subset in damaged_subsets_5pc]
    
    damaged_10pc_samples = [tf.gather(subset, tf.random.shuffle(tf.range(tf.shape(subset)[0]))[:10]) 
                          for subset in damaged_10pc_subsets]

    # Convert to numpy arrays
    healthy_samples_np = [tensor.numpy() for tensor in healthy_samples]
    damaged_samples_np = [tensor.numpy() for tensor in damaged_samples]
    damaged_10pc_samples_np = [tensor.numpy() for tensor in damaged_10pc_samples]

    # Calculate damage indices
    healthy_di = calculate_sample_damage_indexes(healthy_samples_np, baseline_params)
    damaged_di_5pc = calculate_sample_damage_indexes(damaged_samples_np, baseline_params)
    damaged_di_10pc = calculate_sample_damage_indexes(damaged_10pc_samples_np, baseline_params)

    # Concatenate and normalize damage indices
    damage_indices = np.concatenate([healthy_di, damaged_di_5pc, damaged_di_10pc])
    normalized_damage_indices = normalize_damage_indexes(damage_indices)

    conditions = np.concatenate([
        np.full(len(healthy_di), 'Healthy'),
        np.full(len(damaged_di_5pc), '5% Damage'),
        np.full(len(damaged_di_10pc), '10% Damage')
    ])

    # Create a DataFrame
    dataset = pd.DataFrame({
        'Damage_Index': normalized_damage_indices,
        'Condition': conditions
    })

    print(dataset.tail())

    # Prepare data for clustering
    X = dataset[['Damage_Index']].values  # Using only Damage_Index as the feature

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)
    cluster_labels = kmeans.labels_

    # Add cluster labels to the dataset
    dataset['Cluster'] = cluster_labels

    # Map clusters to actual damage conditions based on mean Damage_Index in each cluster
    cluster_mapping = {}
    for cluster in np.unique(cluster_labels):
        # Get the mean Damage_Index for the cluster
        mean_di = dataset[dataset['Cluster'] == cluster]['Damage_Index'].mean()
        cluster_mapping[cluster] = mean_di

    # Sort clusters by mean Damage_Index
    sorted_clusters = sorted(cluster_mapping.items(), key=lambda x: x[1])

    # Assign cluster labels to conditions
    cluster_to_condition = {}
    conditions_order = ['Healthy', '5% Damage', '10% Damage']
    for idx, (cluster, _) in enumerate(sorted_clusters):
        cluster_to_condition[cluster] = conditions_order[idx]

    # Map the clusters to conditions
    dataset['Predicted_Condition'] = dataset['Cluster'].map(cluster_to_condition)

    # Evaluate the clustering performance
    print("Classification Report:")
    print(classification_report(dataset['Condition'], dataset['Predicted_Condition'], target_names=['Healthy', '5% Damage', '10% Damage']))

    # Confusion Matrix
    cm = confusion_matrix(dataset['Condition'], dataset['Predicted_Condition'], labels=['Healthy', '5% Damage', '10% Damage'])
    cm_df = pd.DataFrame(cm, index=['Healthy', '5% Damage', '10% Damage'],
                        columns=['Predicted Healthy', 'Predicted 5% Damage', 'Predicted 10% Damage'])

    # Plot the confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Condition')
    plt.xlabel('Predicted Condition')
    plt.show()

    # Visualize the clusters
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Damage_Index', y=np.zeros_like(dataset['Damage_Index']), hue='Predicted_Condition', data=dataset, palette='viridis', s=100)
    plt.title('KMeans Clustering for Damage Index')
    plt.xlabel('Damage Index')
    plt.yticks([])
    plt.show()
    # Healthy vs 5% Damage
    # Prepare data
    di_h_vs_5pc = np.concatenate([healthy_di, damaged_di_5pc])
    labels_h_vs_5pc = np.concatenate([np.zeros(len(healthy_di)), np.ones(len(damaged_di_5pc))])

    # Compute ROC curve and AUC
    fpr_h_vs_5pc, tpr_h_vs_5pc, _ = roc_curve(labels_h_vs_5pc, di_h_vs_5pc)
    auc_h_vs_5pc = auc(fpr_h_vs_5pc, tpr_h_vs_5pc)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr_h_vs_5pc, tpr_h_vs_5pc, color='darkorange',
             lw=2, label='Curva ROC (área = %0.2f)' % auc_h_vs_5pc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('Curva ROC: saudável vs dano de 5%')
    plt.xlabel('Taxa de falsos positivos')
    plt.ylabel('Taxa de verdadeiros positivos')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

    # Healthy vs 10% Damage
    # Prepare data
    di_h_vs_10pc = np.concatenate([healthy_di, damaged_di_10pc])
    labels_h_vs_10pc = np.concatenate([np.zeros(len(healthy_di)), np.ones(len(damaged_di_10pc))])

    # Compute ROC curve and AUC
    fpr_h_vs_10pc, tpr_h_vs_10pc, _ = roc_curve(labels_h_vs_10pc, di_h_vs_10pc)
    auc_h_vs_10pc = auc(fpr_h_vs_10pc, tpr_h_vs_10pc)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr_h_vs_10pc, tpr_h_vs_10pc, color='green',
             lw=2, label='ROC curve (area = %0.2f)' % auc_h_vs_10pc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('Curva ROC: saudável vs dano de 10%')
    plt.xlabel('Taxa de falsos positivos')
    plt.ylabel('Taxa de verdadeiros positivos')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()
    
    # Generate the damage index report
    report_df = generate_damage_index_report(normalized_damage_indices, conditions)

    # Save the report to a CSV file
    report_file = "damage_index_report.csv"
    report_df.to_csv(report_file, index=False)
    print(f"Report saved to: {report_file}")

if __name__ == '__main__':
    main()