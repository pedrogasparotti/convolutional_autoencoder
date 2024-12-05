import pandas as pd
import matplotlib.pyplot as plt

# File path to the data
file_path = "/Users/home/Documents/github/convolutional_autoencoder/fluctuation_stats.csv"

# Load the data into a DataFrame
df = pd.read_csv(file_path)

# Ensure numeric columns are correctly typed
df['batch_index'] = pd.to_numeric(df['batch_index'], errors='coerce')
df['std_dev'] = pd.to_numeric(df['std_dev'], errors='coerce')
df['median'] = pd.to_numeric(df['median'], errors='coerce')

# Drop rows with NaN values resulting from conversion issues
df = df.dropna(subset=['batch_index', 'std_dev', 'median'])

# Plot the Moving Average of DI for each case
window_size = 5  # Define the window size for the moving average

plt.figure(figsize=(14, 7))

for case in df['case'].unique():
    case_data = df[df['case'] == case]
    batch_indices = case_data['batch_index']
    median_values = case_data['median']

    # Calculate the moving average
    moving_avg = median_values.rolling(window=window_size, center=True).mean()

    # Plot the moving average
    plt.plot(batch_indices, moving_avg, label=f'{case} (Moving Avg)', linewidth=2)

# Customize the plot
plt.title('Média Móvel do Indicador de Dano (DI)', fontsize=16)
plt.xlabel('Passagem', fontsize=14)
plt.ylabel('Valor Médio do DI', fontsize=14)
plt.grid(True)
plt.tight_layout()

# Display the plot
plt.show()