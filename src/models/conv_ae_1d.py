import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datetime import datetime
from tensorflow.keras import regularizers

class Conv1DAutoencoder:
    def __init__(self, input_shape=(1936, 1), dropout_rate=0.25, reg_lambda=0.001):
        self.input_shape = input_shape
        self.dropout_rate = dropout_rate
        self.reg_lambda = reg_lambda
        self.model = self._build_model()
        
    def _build_model(self):
        inputs = Input(shape=self.input_shape, name='encoder_input')
        
        # Encoder
        x = layers.Conv1D(32, kernel_size=3, strides=2, padding='same', 
                          kernel_regularizer=regularizers.l2(self.reg_lambda))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        x = layers.Conv1D(64, kernel_size=3, strides=2, padding='same', 
                          kernel_regularizer=regularizers.l2(self.reg_lambda))(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        x = layers.Conv1D(128, kernel_size=3, strides=2, padding='same', 
                          kernel_regularizer=regularizers.l2(self.reg_lambda))(x)
        x = layers.BatchNormalization()(x)
        encoded = layers.ReLU()(x)
        
        # Decoder
        x = layers.Conv1DTranspose(128, kernel_size=3, strides=2, padding='same')(encoded)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        x = layers.Conv1DTranspose(64, kernel_size=3, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        x = layers.Conv1DTranspose(32, kernel_size=3, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        outputs = layers.Conv1D(1, kernel_size=3, padding='same', activation='linear')(x)
        
        return models.Model(inputs, outputs, name='Conv1D_Autoencoder')
    
def load_and_preprocess_dof_data(file_path, sequence_length=1936):
    """Load and preprocess data for a single DOF"""
    data = pd.read_csv(file_path)
    data = data.iloc[:, -sequence_length:]  # Take last sequence_length columns
    data = data.values
    
    # Normalize per sample
    data_mean = np.mean(data, axis=1, keepdims=True)
    data_std = np.std(data, axis=1, keepdims=True)
    data_std = np.where(data_std == 0, 1e-6, data_std)
    data_normalized = (data - data_mean) / data_std
    
    return data_normalized.reshape(-1, sequence_length, 1), (data_mean, data_std)

def train_dof_model(dof_number, x_train, x_val, project_dir):
    """Train model for a specific DOF"""
    # Create model
    model = Conv1DAutoencoder(input_shape=(1936, 1))
    model = model.compile_model()
    
    # Setup callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(project_dir, 'models', f'best_model_dof_{dof_number}_{timestamp}.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train
    history = model.fit(
        x_train, x_train,
        epochs=1000,
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
    
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('MAE Loss', fontsize=12)
    plt.yscale('log')
    plt.title('Training History for Different DOFs', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(project_dir, 'models', f'training_curves_{timestamp}.png'),
                dpi=300, bbox_inches='tight')
    plt.show()

def plot_reconstructions(models, test_data, dof_numbers, project_dir):
    """Plot reconstruction examples for each DOF"""
    fig, axes = plt.subplots(len(models), 2, figsize=(15, 5*len(models)))
    
    for i, (model, dof, data) in enumerate(zip(models, dof_numbers, test_data)):
        # Get reconstructions
        reconstruction = model.predict(data[:1])
        
        # Plot original
        axes[i, 0].plot(data[0].flatten(), label='Original')
        axes[i, 0].set_title(f'DOF {dof} - Original')
        axes[i, 0].grid(True)
        
        # Plot reconstruction
        axes[i, 1].plot(reconstruction[0].flatten(), label='Reconstructed')
        axes[i, 1].set_title(f'DOF {dof} - Reconstructed')
        axes[i, 1].grid(True)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(project_dir, 'models', f'reconstructions_{timestamp}.png'),
                dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Setup
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_dir = os.path.join(project_dir, 'data', 'vbi_baseline_train')
    
    # DOFs to process
    dof_numbers = [1, 4, 5, 6]
    
    # Store results
    histories = []
    models = []
    test_datasets = []
    
    # Process each DOF
    for dof in dof_numbers:
        print(f"\nProcessing DOF {dof}")
        
        # Load data
        file_path = os.path.join(data_dir, f'acc_vehicle_data_dof_{dof}_healthy.csv')
        data, norm_params = load_and_preprocess_dof_data(file_path)
        
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