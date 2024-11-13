import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.utils import plot_model
from tensorflow.keras.regularizers import l2

def build_autoencoder(input_shape=(38, 66, 1)):
    inputs = Input(shape=input_shape, name='encoder_input')

    # Encoder
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)  # (19, 33, 64)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)  # (10, 17, 128)

    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same', name='encoded_layer')(x)  # (5, 9, 256)

    # Decoder
    x = layers.Conv2DTranspose(256, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D((2, 2))(x)  # (10, 18, 256)

    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D((2, 2))(x)  # (20, 36, 128)

    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D((2, 2))(x)  # (40, 72, 64)

    # Adjust to match the original shape (38, 66, 64)
    x = layers.Cropping2D(((1, 1), (3, 3)))(x)  # Crop to (38, 66, 64)

    # Final layer with tanh activation for output range [-1, 1]
    decoded = layers.Conv2D(1, (3, 3), activation='tanh', padding='same', name='decoder_output')(x)  # (38, 66, 1)
    
    autoencoder = models.Model(inputs, decoded, name='Enhanced_Autoencoder_with_Regularization')
    return autoencoder

def compile_autoencoder(model, optimizer='adam', loss='mse'):
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
