import keras_tuner as kt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.layers import Input
from tensorflow.keras.regularizers import l2
import os
import sys
import matplotlib.pyplot as plt
import keras

# Define the build_autoencoder function
def build_autoencoder(
    conv_activations=['relu', 'relu', 'relu'],
    kernel_regularizer=0.01,
    latent_dim=256
):
    inputs = layers.Input(shape=input_shape, name='encoder_input')
    x = inputs

    # Encoder
    for i in range(len(conv_filters)):
        x = layers.Conv2D(
            filters=conv_filters[i],
            kernel_size=conv_kernel_sizes[i],
            activation=conv_activations[i],
            padding='same',
            kernel_regularizer=regularizers.l2(kernel_regularizer)
        )(x)
        x = layers.BatchNormalization()(x)
        if dropout_rates[i] > 0:
            x = layers.Dropout(dropout_rates[i])(x)
        x = layers.MaxPooling2D(pool_sizes[i], padding='same')(x)

    # Calculate shape before flattening (static calculation)
    # After 3 poolings with (2,2): ceil(44/2^3) = ceil(44/8) = 6
    shape_before_flattening = [
        6,  # height
        6,  # width
        conv_filters[-1]  # channels
    ]

    # Bottleneck
    x = layers.Flatten()(x)
    x = layers.Dense(latent_dim, activation='relu', name='encoded_layer')(x)

    # Decoder
    decoder_units = np.prod(shape_before_flattening)  # 6*6*256=9216
    x = layers.Dense(decoder_units, activation='relu')(x)
    x = layers.Reshape(shape_before_flattening)(x)

    for i in reversed(range(len(conv_filters))):
        x = layers.Conv2DTranspose(
            filters=conv_filters[i],
            kernel_size=conv_kernel_sizes[i],
            activation=conv_activations[i],
            padding='same',
            kernel_regularizer=regularizers.l2(kernel_regularizer)
        )(x)
        x = layers.BatchNormalization()(x)
        if dropout_rates[i] > 0:
            x = layers.Dropout(dropout_rates[i])(x)
        x = layers.UpSampling2D(pool_sizes[i])(x)

    # Adjust dimensions to match input shape (44, 44, 1)
    # After 3 upsamplings: 6→12→24→48
    # Crop 2 from each side to get back to 44
    x = layers.Cropping2D(((2, 2), (2, 2)))(x)  # (48, 48, filters) → (44, 44, filters)

    # Output layer with sigmoid activation for [0, 1] range
    outputs = layers.Conv2D(
        filters=1,
        kernel_size=conv_kernel_sizes[0],
        activation='sigmoid',
        padding='same',
        name='decoder_output'
    )(x)

    autoencoder = models.Model(inputs, outputs, name='Autoencoder')
    return autoencoder

# Define the model_builder function
def model_builder(hp):
    # Hyperparameter definitions
    conv_filters = [
        hp.Int(f'conv_filters_{i}', min_value=32, max_value=256, step=32)
        for i in range(3)
    ]
    conv_kernel_sizes = [
        hp.Choice(f'conv_kernel_size_{i}', values=[3, 5])
        for i in range(3)
    ]
    conv_activations = [
        hp.Choice(f'conv_activation_{i}', values=['relu', 'leaky_relu', 'elu'])
        for i in range(3)
    ]
    dropout_rates = [
        hp.Float(f'dropout_rate_{i}', min_value=0.0, max_value=0.5, step=0.1)
        for i in range(3)
    ]
    kernel_regularizer = hp.Float('kernel_regularizer', min_value=1e-5, max_value=1e-2, sampling='log')
    latent_dim = hp.Int('latent_dim', min_value=64, max_value=512, step=64)

    # Build the model with the hyperparameters
    autoencoder = build_autoencoder(
        input_shape=(44, 44, 1),
        conv_filters=conv_filters,
        conv_kernel_sizes=conv_kernel_sizes,
        conv_activations=conv_activations,
        pool_sizes=[(2, 2)] * 3,
        dropout_rates=dropout_rates,
        kernel_regularizer=kernel_regularizer,
        latent_dim=latent_dim
    )

    # Compile the model
    autoencoder.compile(
        optimizer='adam',
        loss='mae',
        metrics=['mae']
    )

    return autoencoder

# Initialize the tuner
tuner = kt.RandomSearch(
    model_builder,
    objective='val_loss',
    max_trials=20,
    executions_per_trial=1,
    directory='hyperparam_tuning',
    project_name='autoencoder_optimization'
)

# Define data path
data_path = r'/Users/home/Documents/github/convolutional_autoencoder/data/processed/npy/healthy_accel_matrices_dof_4.npy'

def load_data(data_path):
    """
    Loads data from a .npy file.
    """
    data = np.load(data_path)
    return data

# Load and preprocess data
data = load_data(data_path)

# Ensure that the data shape is compatible
# Expected shape: (num_samples, 44, 44, 1)
if data.ndim == 3:
    x_data = data.reshape(-1, 44, 44, 1).astype('float32')  # Reshape and convert to float32
elif data.ndim == 4 and data.shape[-1] == 1:
    x_data = data.astype('float32')
else:
    raise ValueError(f"Unexpected data shape: {data.shape}")

# Normalize between 0 and 1
x_data /= np.max(x_data)

# Split data into training and validation sets
x_train, x_val = train_test_split(x_data, test_size=0.2, random_state=42)

# Start hyperparameter search
tuner.search(
    x_train, x_train,
    epochs=20,
    batch_size=32,
    validation_data=(x_val, x_val),
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    ]
)

# Retrieve the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"""
The hyperparameter search is complete. The optimal number of filters in the first Conv layer is {best_hps.get('conv_filters_0')},
the second Conv layer is {best_hps.get('conv_filters_1')}, and the third Conv layer is {best_hps.get('conv_filters_2')}.
The optimal kernel sizes are {best_hps.get('conv_kernel_size_0')}, {best_hps.get('conv_kernel_size_1')}, {best_hps.get('conv_kernel_size_2')}.
The optimal activation functions are {best_hps.get('conv_activation_0')}, {best_hps.get('conv_activation_1')}, {best_hps.get('conv_activation_2')}.
The optimal dropout rates are {best_hps.get('dropout_rate_0')}, {best_hps.get('dropout_rate_1')}, {best_hps.get('dropout_rate_2')}.
The optimal kernel regularizer is {best_hps.get('kernel_regularizer')}.
The optimal latent dimension is {best_hps.get('latent_dim')}.
""")

# Build the model with the best hyperparameters
model = tuner.hypermodel.build(best_hps)

# Train the best model with early stopping
history = model.fit(
    x_train, x_train,
    epochs=50,
    batch_size=32,
    validation_data=(x_val, x_val),
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]
)

# Evaluate the model on validation data
val_loss, val_mae = model.evaluate(x_val, x_val)
print(f'Validation MAE: {val_mae}')

# Make predictions
decoded_signals = model.predict(x_val)

# Display original and reconstructed signals
n = 5  # Number of signals to display
selected_signals = x_val[:n]

plt.figure(figsize=(12, 6))
for i in range(n):
    # Original signals
    plt.subplot(2, n, i + 1)
    plt.imshow(selected_signals[i].reshape(44, 44), cmap='gray')
    plt.title('Original')
    plt.axis('off')

    # Reconstructed signals
    plt.subplot(2, n, i + n + 1)
    plt.imshow(decoded_signals[i].reshape(44, 44), cmap='gray')
    plt.title('Reconstructed')
    plt.axis('off')

plt.tight_layout()
plt.show()
