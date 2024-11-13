import matplotlib.pyplot as plt

def plot_input_vs_reconstructed(input_signal, reconstructed_signal, signal_index=0, title="Input vs. Reconstructed Signal"):
    """
    Plots the input signal and the reconstructed signal for comparison.
    
    Parameters:
    - input_signal (numpy.ndarray): The original input signal array.
    - reconstructed_signal (numpy.ndarray): The reconstructed signal array from the model.
    - signal_index (int): The index of the signal to plot (default is 0).
    - title (str): Title of the plot.
    """
    plt.figure(figsize=(12, 6))
    
    # Plot input signal
    plt.plot(input_signal[signal_index], label="Input Signal", linewidth=1.5)
    
    # Plot reconstructed signal
    plt.plot(reconstructed_signal[signal_index], label="Reconstructed Signal", linestyle='--', linewidth=1.5)
    
    # Plot styling
    plt.title(title)
    plt.xlabel("Time Steps")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()