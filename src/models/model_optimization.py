import tensorflow as tf
import keras_tuner as kt
import os
from datetime import datetime
import numpy as np
from tensorflow.keras import mixed_precision
from sklearn.model_selection import train_test_split
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from models.autoencoder import build_autoencoder_single_model

class TrainingTuner(kt.HyperModel):
    def __init__(self, input_shape=(44, 44, 1), latent_dim=128):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        
    def build(self, hp):
        # Base model architecture remains the same
        model = build_autoencoder_single_model(
            input_shape=self.input_shape,
            latent_dim=self.latent_dim
        )
        
        # Hyperparameters to optimize
        init_lr = hp.Float('initial_learning_rate', 1e-4, 1e-2, sampling='log')
        
        # Optimizer parameters
        optimizer_type = hp.Choice('optimizer_type', ['adam', 'sgd'])
        
        if optimizer_type == 'adam':
            beta_1 = hp.Float('adam_beta_1', 0.8, 0.99)
            beta_2 = hp.Float('adam_beta_2', 0.99, 0.9999)
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=init_lr,
                beta_1=beta_1,
                beta_2=beta_2,
                clipnorm=1.0
            )
        else:
            momentum = hp.Float('momentum', 0.8, 0.99)
            nesterov = hp.Boolean('nesterov')
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=init_lr,
                momentum=momentum,
                nesterov=nesterov,
                clipnorm=1.0
            )

        # Learning rate schedule
        use_lr_schedule = hp.Boolean('use_lr_schedule')
        if use_lr_schedule:
            schedule_type = hp.Choice('lr_schedule_type', ['exponential', 'cosine'])
            if schedule_type == 'exponential':
                decay_rate = hp.Float('decay_rate', 0.8, 0.99)
                decay_steps = hp.Int('decay_steps', 1000, 10000, step=1000)
                lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                    init_lr, decay_steps=decay_steps, decay_rate=decay_rate
                )
                optimizer.learning_rate = lr_schedule
            else:  # cosine
                decay_steps = hp.Int('decay_steps', 1000, 10000, step=1000)
                lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
                    init_lr, decay_steps=decay_steps
                )
                optimizer.learning_rate = lr_schedule

        # Compile model
        model.compile(optimizer=optimizer, loss='mae')
        return model

def create_lr_schedule(hp):
    """Creates learning rate schedule based on hyperparameters"""
    schedule_type = hp.Choice('lr_schedule_type', ['exponential', 'cosine', 'step'])
    initial_lr = hp.Float('initial_lr', 1e-4, 1e-2, sampling='log')
    
    if schedule_type == 'exponential':
        decay_rate = hp.Float('decay_rate', 0.8, 0.99)
        return tf.keras.optimizers.schedules.ExponentialDecay(
            initial_lr, decay_steps=1000, decay_rate=decay_rate
        )
    elif schedule_type == 'cosine':
        decay_steps = hp.Int('decay_steps', 1000, 10000, step=1000)
        return tf.keras.optimizers.schedules.CosineDecay(
            initial_lr, decay_steps=decay_steps
        )
    else:  # step
        return tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=[1000, 2000, 3000],
            values=[initial_lr, initial_lr*0.1, initial_lr*0.01, initial_lr*0.001]
        )

def run_hyperparameter_optimization(x_train, x_val, project_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tuner_dir = os.path.join(project_dir, 'tuner_logs', timestamp)
    os.makedirs(tuner_dir, exist_ok=True)

    tuner = kt.BayesianOptimization(
        hypermodel=TrainingTuner(),
        objective=kt.Objective("val_loss", direction="min"),
        max_trials=30,
        num_initial_points=5,
        directory=tuner_dir,
        project_name='training_optimization'
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    # Search
    tuner.search(
        x_train, x_train,
        validation_data=(x_val, x_val),
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )

    # Get best hyperparameters
    best_hps = tuner.get_best_hyperparameters(1)[0]
    print("\nBest Hyperparameters:")
    for hp in best_hps.values:
        print(f"{hp}: {best_hps.values[hp]}")

    # Train best model
    best_model = tuner.hypermodel.build(best_hps)
    history = best_model.fit(
        x_train, x_train,
        validation_data=(x_val, x_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )

    return best_model, best_hps, history

def save_results(best_model, best_hps, project_dir, timestamp):
    """Save the best model and hyperparameters"""
    # Save model
    model_path = os.path.join(project_dir, 'models', f'best_model_{timestamp}.keras')
    best_model.save(model_path)
    
    # Save hyperparameters
    hp_path = os.path.join(project_dir, 'models', f'best_hyperparameters_{timestamp}.txt')
    with open(hp_path, 'w') as f:
        f.write("Best Hyperparameters:\n")
        for hp in best_hps.values:
            f.write(f"{hp}: {best_hps.values[hp]}\n")

def load_data(data_path):
    """
    Loads data from a .npy file

    Parameters:
    - data_path: str
        Path to the .npy file containing the data.

    Returns:
    - numpy.ndarray
         data.
    """
    # Load data
    data = np.load(data_path)

    return data

def main():

    data_path = r'/Users/home/Documents/github/convolutional_autoencoder/data/processed/npy/acc_vehicle_data_dof_6_train.npy'

    data = load_data(data_path)

    # First split: separate test set (e.g., 20% of total data)
    x_train_val, x_test = train_test_split(data, test_size=0.2, random_state=42)
    
    # Second split: separate validation set from remaining data
    x_train, x_val = train_test_split(x_train_val, test_size=0.2, random_state=42)
    
    # Run optimization
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    best_model, best_hps, history = run_hyperparameter_optimization(x_train, x_val, project_dir)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results(best_model, best_hps, project_dir, timestamp)

if __name__ == "__main__":
    main()