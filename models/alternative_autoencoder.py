import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam

# Import your autoencoder model
from models.autoencoder import build_deep_autoencoder  # Update the import path as per your file structure

# Define paths
data_path = r'/Users/home/Documents/github/convolutional_autoencoder/data/processed/npy/healthy_accel_matrices_dof_5.npy'
model_save_path = r'/Users/home/Documents/github/convolutional_autoencoder/models/autoencoder_best_model.keras'
log_dir = r'/Users/home/Documents/github/convolutional_autoencoder/logs'

# Define training parameters
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
LATENT_DIM = 128  # Adjust based on experimentation


def load_data(data_path):
    """
    Loads data from a .npy file and flattens it.
    """
    data = np.load(data_path)
    print(f"Loaded data shape: {data.shape}")
    
    # Flatten the input if it's a matrix
    if len(data.shape) > 2:
        data = data.reshape(data.shape[0], -1)
        print(f"Flattened data shape: {data.shape}")
    
    return data


def main():
    # Load and preprocess data
    data = load_data(data_path)
    
    # Split data into training and validation sets
    x_train, x_val = train_test_split(data, test_size=VALIDATION_SPLIT, random_state=42)
    print(f"Training data shape: {x_train.shape}")
    print(f"Validation data shape: {x_val.shape}")
    
    # Build the deep autoencoder model
    input_dim = x_train.shape[1]
    autoencoder = build_deep_autoencoder(input_dim=input_dim, encoding_dim=LATENT_DIM)
    autoencoder.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse')
    print("Model compiled successfully")

    # Define callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(filepath=model_save_path, monitor='val_loss', save_best_only=True, verbose=1),
        TensorBoard(log_dir=log_dir, histogram_freq=1)
    ]

    # Train the model
    history = autoencoder.fit(
        x_train, x_train,  # Input and target are the same
        validation_data=(x_val, x_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    print("Training completed successfully")

    # Save the training history for analysis
    history_path = os.path.join(log_dir, 'training_history.npy')
    np.save(history_path, history.history)
    print(f"Training history saved to {history_path}")

    # Evaluate reconstruction on validation data
    reconstructed = autoencoder.predict(x_val[:5])
    print(f"Reconstructed shape: {reconstructed.shape}")
    
    # Add your visualization or reconstruction analysis code here


if __name__ == '__main__':
    main()