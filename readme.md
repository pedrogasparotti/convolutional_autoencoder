# Signal reconstruction and anomaly detection with unsupervised machine learning 

This project focuses on the reconstruction and classification of accelerometer signals using convolutional autoencoders. The primary goal is to detect anomalies and classify signals as either healthy or damaged by analyzing their patterns and characteristics.

## Overview

- **Signal Processing**: Load, preprocess, and normalize accelerometer data.
- **Data Transformation**: Convert 1D signals into 2D matrices suitable for convolutional neural networks.
- **Visualization**: Plot and compare healthy and damaged signals to identify distinguishing features.
- **Outlier Detection**: Implement methods to detect anomalous signals within the dataset.
- **Modeling**: Use convolutional autoencoders for signal reconstruction and anomaly detection.

   ## Directory Structure

![image](https://github.com/user-attachments/assets/eb2d9a3e-2ed2-45bb-aabc-196dcd05bef6)


## Getting Started

### Prerequisites

- Python 3.10 (the version I currently use to ensure proper tensorflow function)
- NumPy
- Pandas (I only use pandas to make sanity checking in some data transformation processes) 
- Matplotlib (of course matplotlib)
- TensorFlow or PyTorch (for autoencoder implementation)

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name

### Create a Virtual Environment


python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

### Install Dependencies

pip install -r requirements.txt

## Usage

`python src/data/make_dataset_pd.py` --> load and normalize the accelerometer data

`cd src/visualization/` --> directory with tools to visualize signal

`python src/models/train_autoencoder.py` --> train the autoencoder model

`python src/visualization/plot_training.py` --> visualize your training history

I have developed the whole code in macOs using M2 Pro chip, training takes less than 5 minutes (code runs on WSL2 Ubuntu as well)

### License

This project is licensed under the MIT License.

Contact me if you want to talk about AI with someone that will not use GPT to answer you (for a change) ;)
