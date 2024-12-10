import tensorflow as tf
from tensorflow.keras import layers, models, Input, regularizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.regularizers import l2
import numpy as np

def build_autoencoder_single_model(input_shape=(44, 44, 1), latent_dim=128):
    inputs = Input(shape=input_shape, name='encoder_input')

    # Encoder
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)  # Output: (22, 22, 64)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)  # Output: (11, 11, 128)

    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same', name='encoded_layer')(x)  # Output: (6, 6, 256)

    # Flatten and bottleneck (latent space)
    x = layers.Flatten()(x)
    x = layers.Dense(latent_dim, activation='relu')(x)  # Bottleneck layer

    # Decoder
    x = layers.Conv2DTranspose(256, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D((2, 2))(x)  # Output: (12, 12, 256)

    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D((2, 2))(x)  # Output: (24, 24, 128)

    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D((2, 2))(x)

    # Adjust to match the original input shape (44, 44)
    x = layers.Cropping2D(((2, 2), (2, 2)))(x)  # Crop to (44, 44, 64)

    # Final layer with sigmoid activation for output range [0, 1]
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='decoder_output')(x)  # Output: (44, 44, 1)

    autoencoder = models.Model(inputs, decoded, name='Autoencoder')
    return autoencoder

def compile_autoencoder(model, 
                        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0), 
                        loss='mae'):
    """
    Compiles the autoencoder model.

    Parameters:
    - model (Model): The autoencoder model to compile.
    - optimizer (str or Optimizer): Optimizer to use.
    - loss (str or Loss): Loss function to use.
    
    Returns:
    - model (Model): The compiled autoencoder model.
    """
    model.compile(optimizer=optimizer, loss=loss)
    return model

def get_callbacks(model_path='autoencoder_best_model.keras', patience=10):
    """
    Returns the callbacks for training.
    
    Parameters:
    - model_path (str): Path to save the best model.
    - patience (int): Early stopping patience.
    
    Returns:
    - callbacks (list): List of callbacks.
    """
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        model_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    return [early_stopping, checkpoint]

def plot_model_architecture(model, architecture_file='autoencoder_architecture.png'):
    """
    Saves the model architecture diagram to a file.
    
    Parameters:
    - model (Model): The model whose architecture to plot.
    - architecture_file (str): Path to save the architecture diagram.
    """
    plot_model(model, to_file=architecture_file, show_shapes=True, show_layer_names=True)