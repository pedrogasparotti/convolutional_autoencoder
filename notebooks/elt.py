import scipy.io
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_dataset():

    # Load the .mat files
    results_mat = scipy.io.loadmat('results.mat')
    dano_labels_mat = scipy.io.loadmat('dano_labels.mat')

    # Extract data from .mat files
    results = np.squeeze(results_mat['results'])
    dano_labels = dano_labels_mat['dano_labels']

    # Create a DataFrame
    df_results = pd.DataFrame(results.T)
    df_results = df_results.iloc[:]
    df_dano_labels = pd.DataFrame(dano_labels.T)

    # Create one-hot encoding for damage cases
    df_dano_labels['damage_case'] = df_dano_labels[0].apply(lambda x: 'y_0' if x == 1 else ('y_1' if x == 0.95 else 'y_2'))
    df_dano_one_hot = pd.get_dummies(df_dano_labels['damage_case'])


    # Ensure one-hot encoded columns are integers
    df_dano_one_hot = df_dano_one_hot.astype(int)

    # Combine the DataFrames
    df_combined = pd.concat([df_results, df_dano_one_hot], axis=1)

    # Drop the original 'dano' and 'damage_case' columns as they are now encoded
    #df_combined.drop([0, 'damage_case'], axis=1, inplace=True)

    # Discard the first 10,000 columns
    #df_combined_reduced = df_combined.iloc[:, 10004:]

    # df_plot = df_filtered
    # df_plot.drop(['y_0', 'y_1', 'y_2'], axis=1, inplace=True)

    return df_combined

def split_dataset(dados):
    
    # Separate the data based on the classes
    y_0_data = dados[dados['y_0'] == 1]
    y_1_data = dados[dados['y_1'] == 1]
    y_2_data = dados[dados['y_2'] == 1]

    # Shuffle the instances within each class
    y_0_data = y_0_data.sample(frac=1, random_state=42)
    y_1_data = y_1_data.sample(frac=1, random_state=42)
    y_2_data = y_2_data.sample(frac=1, random_state=42)

    # Select 100 instances of each class for the testing dataset
    y_0_test = y_0_data.iloc[:100]
    y_1_test = y_1_data.iloc[:100]
    y_2_test = y_2_data.iloc[:100]

    # Concatenate the testing data from all classes
    test_data = pd.concat([y_0_test, y_1_test, y_2_test])

    # Remove the one hot encoding columns to get the testing features and labels
    X_test = test_data.drop(['y_0', 'y_1', 'y_2'], axis=1)
    y_test = test_data[['y_0', 'y_1', 'y_2']]

    remaining_data = dados.drop(test_data.index)
    # Split the remaining data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
    remaining_data.drop(['y_0', 'y_1', 'y_2'], axis=1),
    remaining_data[['y_0', 'y_1', 'y_2']],
    test_size=0.2,  # You can adjust the validation set size as needed
    random_state=42
)
    return X_test, y_test, X_train, X_val, y_train, y_val