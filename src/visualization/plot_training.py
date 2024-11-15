import matplotlib.pyplot as plt
import numpy as np

# Load the training history
training_history = np.load('/Users/home/Documents/github/convolutional_autoencoder/models/training_history.npy', allow_pickle=True).item()

# Plot training and validation loss
plt.plot(training_history['loss'], label='Training Loss')
plt.plot(training_history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.show()
