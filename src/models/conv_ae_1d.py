import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
import pandas as pd
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datetime import datetime
from tensorflow.keras.layers import LeakyReLU
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datetime import datetime

class Conv1DAutoencoder:
    def __init__(self, input_shape=(1936, 1)):
        self.input_shape = input_shape
        self.model = self._build_model()
        
    def _build_model(self):
        # Encoder Input
        inputs = Input(shape=(1936, 1), name='encoder_input')
        
        # Encoder
        x = layers.Conv1D(32, 3, strides=2, padding='same', activation=None)(inputs)
        x = LeakyReLU(alpha=0.3)(x)
        
        x = layers.Conv1D(64, 3, strides=2, padding='same', activation=None)(x)
        x = LeakyReLU(alpha=0.3)(x)
        
        x = layers.Conv1D(128, 3, strides=2, padding='same', activation=None)(x)
        x = LeakyReLU(alpha=0.3)(x)
        
        # Decoder
        x = layers.Conv1DTranspose(64, 3, strides=2, padding='same', activation=None)(x)
        x = LeakyReLU(alpha=0.3)(x)
        
        x = layers.Conv1DTranspose(32, 3, strides=2, padding='same', activation=None)(x)
        x = LeakyReLU(alpha=0.3)(x)
        
        # Output layer
        decoded = layers.Conv1DTranspose(1, 3, strides=2, padding='same', activation='linear')(x)
        
        # Create model
        autoencoder = models.Model(inputs, decoded, name='Simplified_Conv1D_Autoencoder')
        
        return autoencoder 
    
    def compile_model(self, learning_rate=0.0001):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='mae')
        return self.model
     
def load_and_preprocess_dof_data(file_path, sequence_length=1936, noise_factor=0):
    """
    Load and preprocess data for a single DOF, adding noise and normalizing to [0, 1].
    
    Parameters:
    - file_path (str): Path to the CSV file containing the data.
    - sequence_length (int): Length of the sequence to extract.
    - noise_factor (float): Percentage of the maximum data value to use as noise.
    
    Returns:
    - data_normalized (np.ndarray): Data normalized to [0, 1] with shape (samples, sequence_length, 1).
    """
    # Load data
    data = pd.read_csv(file_path)
    data = data.iloc[:, -sequence_length:]  # Take last sequence_length columns
    data = data.values
    
    # Add noise to the raw data
    noise = np.random.normal(loc=0, scale=noise_factor * np.max(data), size=data.shape)
    data_with_noise = data + noise
    
    # Normalize the noisy data to the range [0, 1]
    data_min = np.min(data_with_noise, axis=1, keepdims=True)
    data_max = np.max(data_with_noise, axis=1, keepdims=True)
    data_normalized = (data_with_noise - data_min) / (data_max - data_min)
    
    return data_normalized.reshape(-1, sequence_length, 1)

def train_dof_model(dof_number, x_train, x_val, project_dir):
    """Train model for a specific DOF"""
    # Create model
    model = Conv1DAutoencoder(input_shape=(1936, 1))
    model = model.compile_model()
    
    # Setup callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(project_dir, 'models', f'best_model_dof_{dof_number}_CONV1D.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.9,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train
    history = model.fit(
        x_train, x_train,
        epochs=150,
        batch_size=32,
        validation_data=(x_val, x_val),
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

def plot_training_curves(histories, dof_numbers, project_dir):
    """Plot training curves for all DOFs"""
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'purple']
    for i, (history, dof) in enumerate(zip(histories, dof_numbers)):
        plt.plot(history.history['loss'], 
                label=f'DOF {dof} - Train',
                color=colors[i],
                linestyle='-')
        plt.plot(history.history['val_loss'],
                label=f'DOF {dof} - Val',
                color=colors[i],
                linestyle='--')
    
    plt.xlabel('Épocas', fontsize=12)
    plt.ylabel('Erro médio absoluto de reconstrução', fontsize=16)
    plt.yscale('log')
    plt.title('Histórico de treinamento dos diferentes graus de liberdade', fontsize=20)
    plt.legend(fontsize=10)
    plt.grid(True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(project_dir, 'models', f'training_curves_{timestamp}.png'),
                dpi=300, bbox_inches='tight')
    plt.show()

def plot_reconstructions(models, test_data, dof_numbers, project_dir):
    """Plot reconstruction examples with overlaid signals"""
    fig, axes = plt.subplots(len(models), 1, figsize=(15, 5*len(models)))
    
    # Make axes a list if there's only one subplot
    if len(models) == 1:
        axes = [axes]
    
    # Colors and styles
    original_style = dict(color='blue', linewidth=1.5, label='Original', alpha=0.8)
    recon_style = dict(color='red', linewidth=1.5, linestyle='--', label='Reconstruído', alpha=0.8)
    
    for i, (model, dof, data) in enumerate(zip(models, dof_numbers, test_data)):
        # Get reconstructions
        reconstruction = model.predict(data[:1])
        
        # Plot both signals on same subplot
        axes[i].plot(data[0].flatten(), **original_style)
        axes[i].plot(reconstruction[0].flatten(), **recon_style)
        
        # Customize subplot
        axes[i].set_title(f'DOF {dof}', fontsize=13)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlabel('Posição', fontsize=10)
        axes[i].set_ylabel('Amplitude', fontsize=10)
        axes[i].legend(fontsize=12)
        
        # Adjust ticks font size
        axes[i].tick_params(axis='both', which='major', labelsize=10)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(project_dir, 'models', f'reconstructions_{timestamp}.png'),
                dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Setup
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_dir = os.path.join('/Users/home/Documents/github/convolutional_autoencoder/data/dataset')
    
    # DOFs to process
    dof_numbers = [4]
    
    # Store results
    histories = []
    models = []
    test_datasets = []
    
    # Process each DOF
    for dof in dof_numbers:
        print(f"\nProcessing DOF {dof}")
        
        # Load data
        file_path = os.path.join(data_dir, f'acc_vehicle_data_dof_{dof}.csv')
        data = load_and_preprocess_dof_data(file_path)
        
        # Split data
        x_train_val, x_test = train_test_split(data, test_size=0.2, random_state=42)
        x_train, x_val = train_test_split(x_train_val, test_size=0.2, random_state=42)
        
        # Train model
        model, history = train_dof_model(dof, x_train, x_val, project_dir)
        
        # Store results
        histories.append(history)
        models.append(model)
        test_datasets.append(x_test)
    
    # Plot results
    plot_training_curves(histories, dof_numbers, project_dir)
    plot_reconstructions(models, test_datasets, dof_numbers, project_dir)

if __name__ == "__main__":
    main()