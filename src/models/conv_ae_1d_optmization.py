import os
import tensorflow as tf
import keras_tuner as kt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime
from tensorflow.keras import layers, models, Input

class ConvAutoencoderHyperModel(kt.HyperModel):
    def __init__(self, input_shape=(1936, 1)):
        self.input_shape = input_shape

    def build(self, hp):
        inputs = Input(shape=self.input_shape, name='encoder_input')
        
        # Hyperparameters for regularization
        l2_reg = hp.Float('l2_reg', 1e-6, 1e-3, sampling='log')
        dropout_rate = hp.Float('dropout_rate', 0.0, 0.5, step=0.1)
        use_batch_norm = hp.Boolean('use_batch_norm')
        
        # Encoder
        x = layers.Conv1D(32, kernel_size=3, strides=2, padding='same',
                         kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(inputs)
        if use_batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        x = layers.Conv1D(64, kernel_size=3, strides=2, padding='same',
                         kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
        if use_batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        x = layers.Conv1D(128, kernel_size=3, strides=2, padding='same',
                         kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
        if use_batch_norm:
            x = layers.BatchNormalization()(x)
        encoded = layers.ReLU()(x)
        
        # Decoder
        x = layers.Conv1DTranspose(128, kernel_size=3, strides=2, padding='same',
                                 kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(encoded)
        if use_batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        x = layers.Conv1DTranspose(64, kernel_size=3, strides=2, padding='same',
                                 kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
        if use_batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        x = layers.Conv1DTranspose(32, kernel_size=3, strides=2, padding='same',
                                 kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
        if use_batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        outputs = layers.Conv1D(1, kernel_size=3, padding='same', activation='linear')(x)
        
        model = models.Model(inputs, outputs, name='Conv1D_Autoencoder')
        
        # Optimizer hyperparameters
        learning_rate = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
        beta_1 = hp.Float('beta_1', 0.8, 0.999)
        beta_2 = hp.Float('beta_2', 0.8, 0.999)
        clipnorm = hp.Float('clipnorm', 0.1, 2.0, sampling='log')
        
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            clipnorm=clipnorm
        )
        
        model.compile(optimizer=optimizer, loss='mae')
        return model

def load_and_preprocess_dof_data(file_path, sequence_length=1936):
    data = pd.read_csv(file_path)
    data = data.iloc[:, -sequence_length:]
    data = data.values
    
    data_mean = np.mean(data, axis=1, keepdims=True)
    data_std = np.std(data, axis=1, keepdims=True)
    data_std = np.where(data_std == 0, 1e-6, data_std)
    data_normalized = (data - data_mean) / data_std
    
    return data_normalized.reshape(-1, sequence_length, 1), (data_mean, data_std)

def run_bayesian_optimization(dof_number, x_train, x_val, project_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tuner_dir = os.path.join(project_dir, 'tuner_logs', f'dof_{dof_number}_{timestamp}')
    
    tuner = kt.BayesianOptimization(
        hypermodel=ConvAutoencoderHyperModel(),
        objective='val_loss',
        max_trials=50,
        num_initial_points=10,
        directory=tuner_dir,
        project_name=f'autoencoder_dof_{dof_number}'
    )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
    
    print(f"\nStarting Bayesian Optimization for DOF {dof_number}")
    tuner.search(
        x_train, x_train,
        validation_data=(x_val, x_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr]
    )
    
    # Get best hyperparameters and model
    best_hp = tuner.get_best_hyperparameters(1)[0]
    best_model = tuner.get_best_models(1)[0]
    
    # Save best hyperparameters
    hp_file = os.path.join(project_dir, 'models', f'best_hp_dof_{dof_number}_{timestamp}.txt')
    with open(hp_file, 'w') as f:
        f.write("Best Hyperparameters:\n")
        for param in best_hp.values:
            f.write(f"{param}: {best_hp.values[param]}\n")
    
    return best_model, best_hp

def main():
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_dir = os.path.join(project_dir, 'data', 'vbi_baseline_train')
    
    dof_numbers = [1, 4, 5, 6]
    results = {}
    
    for dof in dof_numbers:
        print(f"\nProcessing DOF {dof}")
        
        # Load and prepare data
        file_path = os.path.join(data_dir, f'acc_vehicle_data_dof_{dof}_healthy.csv')
        data, norm_params = load_and_preprocess_dof_data(file_path)
        
        x_train_val, x_test = train_test_split(data, test_size=0.2, random_state=42)
        x_train, x_val = train_test_split(x_train_val, test_size=0.2, random_state=42)
        
        # Run optimization
        best_model, best_hp = run_bayesian_optimization(dof, x_train, x_val, project_dir)
        
        # Store results
        results[dof] = {
            'model': best_model,
            'hyperparameters': best_hp,
            'test_loss': best_model.evaluate(x_test, x_test, verbose=0)
        }
        
        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(project_dir, 'models', f'best_model_dof_{dof}_{timestamp}.keras')
        best_model.save(model_path)
    
    # Print summary of results
    print("\nOptimization Results Summary:")
    for dof in dof_numbers:
        print(f"\nDOF {dof}:")
        print(f"Test Loss: {results[dof]['test_loss']}")
        print("Best Hyperparameters:")
        for param in results[dof]['hyperparameters'].values:
            print(f"  {param}: {results[dof]['hyperparameters'].values[param]}")

if __name__ == "__main__":
    main()