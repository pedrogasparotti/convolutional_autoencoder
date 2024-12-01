import os
import sys
import time
import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path for imports
project_root = Path('/Users/home/Documents/github/convolutional_autoencoder')
sys.path.append(str(project_root))

# Import the alternative autoencoder
from models.alternative_autoencoder import build_improved_autoencoder, compile_improved_autoencoder

def load_and_preprocess_data(data_path):
    """Load and preprocess the data."""
    data = np.load(data_path)
    if len(data.shape) == 3:
        data = data[..., np.newaxis]
    return data

def create_training_dirs():
    """Create necessary directories for training artifacts."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_dir = project_root / 'training_outputs' / timestamp
    
    dirs = {
        'models': base_dir / 'models',
        'logs': base_dir / 'logs',
        'plots': base_dir / 'plots'
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
        
    return dirs, timestamp

def plot_training_history(history, plot_dir):
    """Plot and save training history."""
    plt.figure(figsize=(12, 4))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Learning rate plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['lr'], label='Learning Rate')
    plt.title('Learning Rate over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(plot_dir / 'training_history.png')
    plt.close()

def plot_reconstructions(model, x_val, plot_dir, n_samples=5):
    """Plot original vs reconstructed samples."""
    reconstructions = model.predict(x_val[:n_samples])
    
    plt.figure(figsize=(15, 3*n_samples))
    for i in range(n_samples):
        # Original
        plt.subplot(n_samples, 2, 2*i + 1)
        plt.imshow(x_val[i].squeeze(), cmap='viridis')
        plt.title(f'Original {i+1}')
        plt.axis('off')
        
        # Reconstruction
        plt.subplot(n_samples, 2, 2*i + 2)
        plt.imshow(reconstructions[i].squeeze(), cmap='viridis')
        plt.title(f'Reconstructed {i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(plot_dir / 'reconstructions.png')
    plt.close()

def get_callbacks(dirs, timestamp):
    """Configure training callbacks."""
    callbacks = [
        # Model checkpoint
        tf.keras.callbacks.ModelCheckpoint(
            filepath=dirs['models'] / 'best_model.keras',
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        
        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir=dirs['logs'],
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        ),
        
        # CSV Logger
        tf.keras.callbacks.CSVLogger(
            dirs['logs'] / 'training_log.csv'
        ),
        
        # Learning rate scheduler with warmup and cosine decay
        tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: 
            0.001 * (epoch + 1) / 10 if epoch < 10  # Warmup
            else 0.001 * (1 + np.cos(np.pi * (epoch - 10) / 90)) / 2  # Cosine decay
        )
    ]
    return callbacks

def main():
    # Configuration
    data_path = '/Users/home/Documents/github/convolutional_autoencoder/data/processed/npy/acc_vehicle_data_dof_6_baseline.npy'
    input_shape = (44, 44, 1)
    batch_size = 32
    epochs = 100
    
    # Create directories
    dirs, timestamp = create_training_dirs()
    print(f"Training outputs will be saved to {dirs['models'].parent}")
    
    # Load and split data
    print("Loading and preprocessing data...")
    data = load_and_preprocess_data(data_path)
    x_train, x_val = train_test_split(data, test_size=0.2, random_state=42)
    
    # Build and compile model
    print("Building and compiling model...")
    model = build_improved_autoencoder(input_shape=input_shape)
    model = compile_improved_autoencoder(model, use_sgd=True)  # Using SGD with momentum
    
    # Configure callbacks
    callbacks = get_callbacks(dirs, timestamp)
    
    # Train model
    print("Starting training...")
    history = model.fit(
        x_train, x_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, x_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save(dirs['models'] / 'final_model.keras')
    
    # Plot training history
    print("Generating training plots...")
    plot_training_history(history, dirs['plots'])
    
    # Plot reconstructions
    plot_reconstructions(model, x_val, dirs['plots'])
    
    # Save training configuration
    config = {
        'timestamp': timestamp,
        'input_shape': input_shape,
        'batch_size': batch_size,
        'epochs': epochs,
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'best_val_loss': float(min(history.history['val_loss'])),
        'data_path': str(data_path)
    }
    
    with open(dirs['models'] / 'training_config.txt', 'w') as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Training completed! All outputs saved to {dirs['models'].parent}")
    
    return history, model, dirs

if __name__ == '__main__':
    # Enable mixed precision training
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    # Set memory growth for GPUs
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    history, model, dirs = main()