import pandas as pd
import matplotlib.pyplot as plt

def load_csv_data(file_path):
    """
    Loads data from a CSV file and performs basic checks.

    Parameters:
        file_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Loaded data as a DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data from {file_path}")
        
        # Display the first few rows of the DataFrame
        print("First few rows of data:")
        print(df.head())
        
        # Check the dimensions of the DataFrame
        print(f"\nData dimensions: {df.shape}")
        
        # Basic statistics for numeric columns
        print("\nBasic statistics:")
        print(df.describe())
        
        # Check for missing values
        missing_values = df.isnull().sum().sum()
        print(f"\nTotal missing values: {missing_values}")
        
        if missing_values > 0:
            print("Warning: Data contains missing values.")
        
        return df
    except FileNotFoundError as e:
        print(f"Error loading CSV file: {e}")
        return None

def plot_first_five_rows(df):
    """
    Plots the first 5 rows of data in the DataFrame as line plots.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data to plot.
    """
    if df is not None and len(df) >= 5:
        print("Plotting the first 5 rows of data...")
        plt.figure(figsize=(10, 6))
        for i in range(5):
            plt.plot(df.columns, df.iloc[i], label=f'Row {i+1}')
        
        plt.xlabel('Columns')
        plt.ylabel('Values')
        plt.title('Line Plot of First 5 Rows')
        plt.legend()
        plt.show()
    else:
        print("DataFrame does not contain enough rows to plot the first 5 rows.")

# Usage
file_path = "/Users/home/Documents/github/convolutional_autoencoder/data/vbi_2d_healthy/acc_vehicle_data_dof_4.csv"
dados = load_csv_data(file_path)

# Plot the first 5 rows if data is loaded
if dados is not None:
    plot_first_five_rows(dados)