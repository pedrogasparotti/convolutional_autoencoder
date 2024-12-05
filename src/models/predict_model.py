import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanAbsoluteError
from scipy.stats import lognorm
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os
import pandas as pd

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
    plt.hist(data, bins='auto', density=True, alpha=0.7, label='MAEs reais')
    
    # Plot fitted distribution
    x = np.linspace(min(data), max(data), 100)
    plt.plot(x, dist.pdf(x, *params), 'r-', lw=2, label=f'Distribuição ajustada ({dist.name})')
    
    plt.title('Distribuição de MAEs dos Sinais Saudáveis')
    plt.xlabel('MAE')
    plt.ylabel('Densidade')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

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
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.linspace(0, max(data) if data is not None else np.exp(mu + 3*sigma), 1000)
    pdf = lognorm.pdf(x, sigma, scale=np.exp(mu))
    
    ax.plot(x, pdf, 'r-', lw=2, label='PDF Log-normal')
    
    if data is not None:
        ax.hist(data, bins=bins, density=True, alpha=0.5, color='blue', label='Histograma dos Dados')
    
    ax.set_xlabel('Valor')
    ax.set_ylabel('Densidade')
    ax.set_title(f'Distribuição Log-normal (μ={mu:.2f}, σ={sigma:.2f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
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
    """
    plt.figure(figsize=figsize)
    
    healthy_x = np.random.normal(0, 0.1, len(healthy_indices))
    damaged_x = np.random.normal(1, 0.1, len(damaged_indices))
    
    plt.scatter(healthy_x, healthy_indices, c='blue', alpha=0.5, label='Saudáveis')
    plt.scatter(damaged_x, damaged_indices, c='red', alpha=0.5, label='Danificados')
    
    plt.axhline(y=np.mean(healthy_indices), color='blue', linestyle='--', alpha=0.5)
    plt.axhline(y=np.mean(damaged_indices), color='red', linestyle='--', alpha=0.5)
    
    plt.title('Índices de Dano: Amostras Saudáveis vs Danificadas')
    plt.xlabel('Tipo de Amostra')
    plt.ylabel('Índice de Dano')
    plt.xticks([0, 1], ['Saudáveis', 'Danificados'])
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    stats_text = f'Saudáveis: μ={np.mean(healthy_indices):.2f}, σ={np.std(healthy_indices):.2f}\n'
    stats_text += f'Danificados: μ={np.mean(damaged_indices):.2f}, σ={np.std(damaged_indices):.2f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return plt.gcf(), plt.gca()

def plot_roc_curve(healthy_di, damaged_di):
    """
    Plots the ROC curve for damage index-based classification.
    """
    scores = np.concatenate([healthy_di, damaged_di])
    labels = np.concatenate([np.zeros(len(healthy_di)), np.ones(len(damaged_di))])
    
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.title("Curva ROC: Saudáveis vs Danificados")
    plt.xlabel("Taxa de Falsos Positivos (FPR)")
    plt.ylabel("Taxa de Verdadeiros Positivos (TPR)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.5)
    plt.show()

def plot_damage_indices_three_sets(healthy_di, damaged_di, damaged_10pc_di, labels=['Saudável', 'Dano de 5%', 'Dano de 15%']):
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
    ax.set_title('Comparativo do índice de dano')
    ax.grid(True)
    
    return fig, ax

def plot_multiple_roc_curves(healthy_di, damaged_di, damaged_10pc_di):
    plt.figure(figsize=(10, 6))
    
    y_true_five = ['Danificado'] * len(damaged_di) + ['Saudável'] * len(healthy_di)
    y_scores_five = damaged_di + healthy_di
    
    y_true_10pc = ['Danificado'] * len(damaged_10pc_di) + ['Saudável'] * len(healthy_di)
    y_scores_10pc = damaged_10pc_di + healthy_di
    
    fpr_five, tpr_five, _ = roc_curve([1 if y == 'Danificado' else 0 for y in y_true_five], y_scores_five)
    roc_auc_five = auc(fpr_five, tpr_five)
    plt.plot(fpr_five, tpr_five, color='darkorange', lw=2, 
             label=f'Curva ROC para 5% de Danos (AUC = {roc_auc_five:.2f})')
    
    fpr_10pc, tpr_10pc, _ = roc_curve([1 if y == 'Danificado' else 0 for y in y_true_10pc], y_scores_10pc)
    roc_auc_10pc = auc(fpr_10pc, tpr_10pc)
    plt.plot(fpr_10pc, tpr_10pc, color='blue', lw=2, 
             label=f'Curva ROC para 10% de Danos (AUC = {roc_auc_10pc:.2f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curvas ROC Comparativas')
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

def main():
    
    # Define paths
    model_path = r'/Users/home/Documents/github/convolutional_autoencoder/models/autoencoder_best_model.keras'
    healthy_path = r'/Users/home/Documents/github/convolutional_autoencoder/data/processed/npy/acc_vehicle_data_dof_6_baseline_val.npy'
    baseline_path = r'/Users/home/Documents/github/convolutional_autoencoder/data/processed/npy/acc_vehicle_data_dof_6_baseline_train.npy'
    anomalous_path_5pc = r'/Users/home/Documents/github/convolutional_autoencoder/data/processed/npy/acc_vehicle_data_dof_6_5pc.npy'
    anomalous_path_10pc = r'/Users/home/Documents/github/convolutional_autoencoder/data/processed/npy/acc_vehicle_data_dof_6_10pc.npy'

    # Load the trained DAE model
    autoencoder_model = load_dae_model(model_path)

    # Load baseline, healthy and anomalous signals
    baseline_data = load_val_signals(baseline_path)
    healthy_data = load_val_signals(healthy_path)
    anomalous_data_5pc = load_val_signals(anomalous_path_5pc)
    anomalous_data_10pc = load_val_signals(anomalous_path_10pc)

    # Calculate MAEs for all datasets
    healthy_maes = calculate_maes(autoencoder_model, healthy_data)
    anomalous_maes_5pc = calculate_maes(autoencoder_model, anomalous_data_5pc)
    anomalous_maes_10pc = calculate_maes(autoencoder_model, anomalous_data_10pc)
    baseline_maes = calculate_maes(autoencoder_model, baseline_data)

    # Fit baseline distribution
    baseline_params = fit_mae_distribution(baseline_maes, plot=False)

    # Generate subsets for all datasets
    healthy_subsets = sample_mae_subsets(healthy_maes)
    damaged_subsets_5pc = sample_mae_subsets(anomalous_maes_5pc)
    damaged_10pc_subsets = sample_mae_subsets(anomalous_maes_10pc)

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
    damaged_10pc_di = calculate_sample_damage_indexes(damaged_10pc_samples_np, baseline_params)

    # Plot damage indices for all three datasets
    fig, ax = plot_damage_indices_three_sets(healthy_di, damaged_di_5pc, damaged_10pc_di,
                                             labels=['Healthy', '5% Damage', '10% Damage'])
    plt.show()

    # Create labeled dataset from damage indices
    dataset = create_dataset_from_damage_indices(healthy_di, damaged_di_5pc, damaged_10pc_di)

    print(dataset.tail())

    # Prepare data for SVM
    X = dataset[['Damage_Index']].values
    y = dataset['Condition'].values

    # Split data into training and testing sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Feature scaling
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and train the SVM classifier
    from sklearn.svm import SVC
    import seaborn as sns
    svm_classifier = SVC(kernel='rbf', random_state=42)
    svm_classifier.fit(X_train_scaled, y_train)

    # Make predictions on the test set
    y_pred = svm_classifier.predict(X_test_scaled)

    # Evaluate the classifier
    from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Healthy', '5% Damage', '10% Damage']))

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=['Healthy', '5% Damage', '10% Damage'],
                         columns=['Predicted Healthy', 'Predicted 5% Damage', 'Predicted 10% Damage'])

    # Plot the confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Condition')
    plt.xlabel('Predicted Condition')
    plt.show()

    # Now, plot ROC curves for Healthy vs 5% Damage and Healthy vs 10% Damage

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
             lw=2, label='Curva ROC (area = %0.2f)' % auc_h_vs_5pc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('Curva ROC: caso saudável vs dano de 5%')
    plt.xlabel('Taxa de falsos negativos')
    plt.ylabel('Taxa de verdadeiros positivos')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

    # Healthy vs 10% Damage
    # Prepare data
    di_h_vs_10pc = np.concatenate([healthy_di, damaged_10pc_di])
    labels_h_vs_10pc = np.concatenate([np.zeros(len(healthy_di)), np.ones(len(damaged_10pc_di))])

    # Compute ROC curve and AUC
    fpr_h_vs_10pc, tpr_h_vs_10pc, _ = roc_curve(labels_h_vs_10pc, di_h_vs_10pc)
    auc_h_vs_10pc = auc(fpr_h_vs_10pc, tpr_h_vs_10pc)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr_h_vs_10pc, tpr_h_vs_10pc, color='green',
            lw=2, label='Curva ROC (area = %0.2f)' % auc_h_vs_5pc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('Curva ROC: caso saudável vs dano de 10%')
    plt.xlabel('Taxa de falsos negativos')
    plt.ylabel('Taxa de verdadeiros positivos')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()