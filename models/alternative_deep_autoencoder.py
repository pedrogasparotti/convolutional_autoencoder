import tensorflow as tf
from tensorflow.keras import layers, models, Input

def build_deep_autoencoder(input_dim=1936, encoding_dim=128):
    """
    Build a standard deep autoencoder using fully connected layers.
    
    Parameters:
    - input_dim: int
        The dimensionality of the input data (e.g., flattened image size).
    - encoding_dim: int
        The dimensionality of the bottleneck (latent space).
    
    Returns:
    - autoencoder: tf.keras.Model
        The complete autoencoder model.
    """
    # Input layer
    inputs = Input(shape=(input_dim,), name='encoder_input')

    # Encoder
    x = layers.Dense(1024, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(encoding_dim, activation='relu', name='latent_layer')(x)  # Bottleneck

    # Decoder
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    decoded = layers.Dense(input_dim, activation='sigmoid', name='decoder_output')(x)

    # Define the autoencoder
    autoencoder = models.Model(inputs, decoded, name='DeepAutoencoder')
    
    return autoencoder

# Input size for 44x44 images (flattened)
input_dim = 44 * 44

# Build the autoencoder
autoencoder = build_deep_autoencoder(input_dim=input_dim, encoding_dim=128)
autoencoder.summary()

# Compile the autoencoder
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss='mae')  # Use 'mae' or 'mse' for reconstruction loss

# Generate synthetic data for demonstration
import numpy as np
x_train = np.random.rand(1000, input_dim)  # Example: 1000 samples of size 1936
x_val = np.random.rand(200, input_dim)

# Train the model
history = autoencoder.fit(
    x_train, x_train,  # Input and target are the same
    epochs=50,
    batch_size=32,
    validation_data=(x_val, x_val)
)

# Extract the encoder part of the model
encoder = tf.keras.Model(inputs=autoencoder.input,
                         outputs=autoencoder.get_layer('latent_layer').output)

# Generate latent representations for input data
latent_representations = encoder.predict(x_train)
print("Latent representations shape:", latent_representations.shape)