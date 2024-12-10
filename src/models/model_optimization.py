import tensorflow as tf
import keras_tuner as kt
import os
from datetime import datetime
import numpy as np
from tensorflow.keras import layers, models, Input, regularizers
from tensorflow.keras.regularizers import l1, l2, l1_l2
from sklearn.model_selection import train_test_split

class EnhancedTrainingTuner(kt.HyperModel):
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape

    def build(self, hp):
        """
        Builds the autoencoder model with enhanced hyperparameter search space.
        
        Parameters:
        - hp (HyperParameters): Hyperparameters to tune.
        
        Returns:
        - Model: Compiled autoencoder model.
        """
        # Architecture hyperparameters
        initial_filters = hp.Int('initial_filters', 32, 128, step=32)
        n_conv_layers = hp.Int('n_conv_layers', 2, 4)
        kernel_sizes = hp.Choice('kernel_size', [3, 5, 7])
        latent_dim = hp.Int('latent_dim', 64, 512, step=64)
        
        # Regularization hyperparameters
        reg_type = hp.Choice('regularization_type', ['none', 'l1', 'l2', 'l1_l2'])
        reg_strength = hp.Float('reg_strength', 1e-6, 1e-2, sampling='log')
        dropout_rate = hp.Float('dropout_rate', 0.0, 0.5, step=0.1)
        
        # Define regularizer based on type
        if reg_type == 'l1':
            regularizer = l1(reg_strength)
        elif reg_type == 'l2':
            regularizer = l2(reg_strength)
        elif reg_type == 'l1_l2':
            regularizer = l1_l2(l1=reg_strength, l2=reg_strength)
        else:
            regularizer = None

        # Training hyperparameters
        learning_rate = hp.Float('learning_rate', 1e-5, 1e-2, sampling='log')
        optimizer_choice = hp.Choice('optimizer', ['adam', 'adamw', 'radam', 'rmsprop'])
        batch_norm = hp.Boolean('batch_normalization')
        activation = hp.Choice('activation', ['relu', 'elu', 'leaky_relu'])

        # Build the model
        inputs = Input(shape=self.input_shape, name='encoder_input')
        x = inputs

        # Encoder
        for i in range(n_conv_layers):
            filters = initial_filters * (2 ** i)
            x = layers.Conv2D(
                filters=filters,
                kernel_size=kernel_sizes,
                padding='same',
                kernel_regularizer=regularizer
            )(x)
            
            if activation == 'leaky_relu':
                x = layers.LeakyReLU(alpha=0.2)(x)
            else:
                x = layers.Activation(activation)(x)
            
            if batch_norm:
                x = layers.BatchNormalization()(x)
            
            if dropout_rate > 0:
                x = layers.Dropout(dropout_rate)(x)
            
            x = layers.MaxPooling2D((2, 2), padding='same')(x)

        # Flatten and bottleneck
        x = layers.Flatten()(x)
        x = layers.Dense(latent_dim, kernel_regularizer=regularizer)(x)
        if activation == 'leaky_relu':
            x = layers.LeakyReLU(alpha=0.2)(x)
        else:
            x = layers.Activation(activation)(x)

        # Reshape for decoder
        # Calculate the shape after encoding
        encoder_shape = x.shape
        decoder_dense_shape = np.prod(self.input_shape) // (4 ** n_conv_layers)
        x = layers.Dense(decoder_dense_shape, kernel_regularizer=regularizer)(x)
        if batch_norm:
            x = layers.BatchNormalization()(x)
        
        # Calculate reshape dimensions
        reshape_dim = int(np.sqrt(decoder_dense_shape))
        x = layers.Reshape((reshape_dim, reshape_dim, 1))(x)

        # Decoder
        for i in range(n_conv_layers - 1, -1, -1):
            filters = initial_filters * (2 ** i)
            x = layers.Conv2DTranspose(
                filters=filters,
                kernel_size=kernel_sizes,
                padding='same',
                kernel_regularizer=regularizer
            )(x)
            
            if activation == 'leaky_relu':
                x = layers.LeakyReLU(alpha=0.2)(x)
            else:
                x = layers.Activation(activation)(x)
            
            if batch_norm:
                x = layers.BatchNormalization()(x)
            
            if dropout_rate > 0:
                x = layers.Dropout(dropout_rate)(x)
            
            x = layers.UpSampling2D((2, 2))(x)

        # Final output layer
        decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='decoder_output')(x)
        
        # Ensure output shape matches input shape
        if decoded.shape[1:3] != self.input_shape[0:2]:
            decoded = layers.Resizing(self.input_shape[0], self.input_shape[1])(decoded)

        model = models.Model(inputs, decoded, name='Enhanced_Autoencoder')

        # Optimizer selection and compilation
        if optimizer_choice == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_choice == 'adamw':
            optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate)
        elif optimizer_choice == 'radam':
            optimizer = tf.keras.optimizers.RectifiedAdam(learning_rate=learning_rate)
        else:  # rmsprop
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

        # Loss function selection
        loss = hp.Choice('loss_function', ['mae', 'mse', 'huber'])
        if loss == 'huber':
            loss = tf.keras.losses.Huber()

        model.compile(optimizer=optimizer, loss=loss)
        return model

def run_enhanced_hyperparameter_optimization(x_train, x_val, project_dir, input_shape=(44, 44, 1)):
    """
    Runs enhanced hyperparameter optimization using Keras Tuner.
    
    Parameters:
    - x_train (np.ndarray): Training dataset
    - x_val (np.ndarray): Validation dataset
    - project_dir (str): Directory to save results
    - input_shape (tuple): Shape of input images
    
    Returns:
    - tuple: (best_model, best_hps, history)
    """
    tuner = kt.BayesianOptimization(
        hypermodel=EnhancedTrainingTuner(input_shape),
        objective=kt.Objective('val_loss', direction='min'),
        max_trials=50,  # Increased number of trials
        directory=os.path.join(project_dir, 'hyperparam_tuning'),
        project_name='enhanced_autoencoder_tuning',
        overwrite=True
    )

    # Define callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]

    # Search strategy
    tuner.search(
        x_train, x_train,
        validation_data=(x_val, x_val),
        epochs=100,  # Increased max epochs
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    # Get best hyperparameters and retrain
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.hypermodel.build(best_hps)
    
    # Final training with best parameters
    history = best_model.fit(
        x_train, x_train,
        validation_data=(x_val, x_val),
        epochs=150,  # Extended final training
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    return best_model, best_hps, history

def main():
    # Load and prepare data
    data_path = r'/Users/home/Documents/github/convolutional_autoencoder/data/processed/npy/acc_vehicle_data_dof_5_baseline.npy'
    data = np.load(data_path)
    
    # Data splits with stratification if applicable
    x_train_val, x_test = train_test_split(data, test_size=0.2, random_state=42)
    x_train, x_val = train_test_split(x_train_val, test_size=0.2, random_state=42)
    
    # Run optimization
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    best_model, best_hps, history = run_enhanced_hyperparameter_optimization(
        x_train, x_val, project_dir
    )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results(best_model, best_hps, project_dir, timestamp)

if __name__ == "__main__":
    main()