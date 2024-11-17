import tensorflow as tf
import numpy as np

# Define input tensor: (batch_size, height, width, channels)
input_tensor = tf.random.normal([1, 44, 44, 1])  # Example latent input

# components
flatten = tf.keras.layers.Flatten()
latent_dense = tf.keras.layers.Dense(5 * 5 * 64, activation='relu')  # Smaller latent representation
reshape = tf.keras.layers.Reshape((5, 5, 64))  # Reshape to (5, 5, 64)

# encoder layers
encoder_layers = [
    # First block: shape 44x44x1
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'),

    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'),

    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same"),
    
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same"),

    tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'),

    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'),

    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same")
]

# encoder layers
decoder_layers = [
    
    tf.keras.layers.Conv2DTranspose(256, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.UpSampling2D((2, 2)),

    tf.keras.layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.UpSampling2D((2, 2)),

    tf.keras.layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='valid'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.UpSampling2D((2, 2)),

    tf.keras.layers.Conv2DTranspose(1, (3, 3), padding='same', activation='sigmoid'),

]

x = input_tensor

# Debugging the encoder step by step
for i, layer in enumerate(encoder_layers):
    x = layer(x)
    print(f"After layer {i + 1} ({layer.__class__.__name__}):", x.shape)

# Debugging the decoder step by step
print("Input tensor shape:", input_tensor.shape)

# Simulate latent space processing (flatten + dense + reshape)
x = flatten(input_tensor)  # Flatten the input tensor
print("After flatten:", x.shape)

x = latent_dense(x)  # Dense layer expands to (6 * 6 * 64)
print("After latent dense:", x.shape)

x = reshape(x)  # Reshape to (6, 6, 64)
print("After reshape:", x.shape)

# Pass through each decoder layer and check the shapes
for i, layer in enumerate(decoder_layers):
    x = layer(x)
    print(f"After layer {i + 1} ({layer.__class__.__name__}):", x.shape)