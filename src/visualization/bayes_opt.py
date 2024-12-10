import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import scipy.stats as stats
import seaborn as sns

class BayesianOptimizationVisualizer:
    def __init__(self):
        # Define the true objective function (example: optimization of learning rate)
        self.bounds = (0.0001, 0.1)  # Learning rate bounds
        
    def objective_function(self, x):
        """Simulated objective function (validation loss)"""
        return -(1.5 * np.sin(13 * x) * np.sin(27 * x) + 0.5) + 0.3 * np.random.randn()
    
    def expected_improvement(self, X, X_sample, Y_sample, gpr, xi=0.01):
        """Compute the expected improvement acquisition function"""
        mu, sigma = gpr.predict(X.reshape(-1, 1), return_std=True)
        
        mu_sample_opt = np.min(Y_sample)
        
        with np.errstate(divide='warn'):
            imp = mu_sample_opt - mu - xi
            Z = imp / sigma
            ei = imp * stats.norm.cdf(Z) + sigma * stats.norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
            
        return ei
    
    def plot_optimization(self, n_iterations=10):
        """Plot the Bayesian optimization process"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        
        # Generate input space for visualization
        X = np.linspace(self.bounds[0], self.bounds[1], 1000).reshape(-1, 1)
        
        # Initialize samples
        X_sample = np.array([]).reshape(-1, 1)
        Y_sample = np.array([])
        
        # Define GP kernel
        kernel = ConstantKernel(1.0) * RBF(length_scale=0.01)
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        
        # Colors for iterations
        colors = plt.cm.viridis(np.linspace(0, 1, n_iterations))
        
        for i in range(n_iterations):
            # If no samples, sample randomly
            if X_sample.shape[0] == 0:
                X_next = np.array([[np.random.uniform(self.bounds[0], self.bounds[1])]])
            else:
                # Fit GP
                gpr.fit(X_sample, Y_sample)
                
                # Compute EI
                ei = self.expected_improvement(X, X_sample, Y_sample, gpr)
                X_next = X[np.argmax(ei)].reshape(-1, 1)
            
            # Sample objective function
            Y_next = self.objective_function(X_next[0])
            
            # Add sample to observed points
            X_sample = np.vstack((X_sample, X_next)) if X_sample.size else X_next
            Y_sample = np.append(Y_sample, Y_next)
            
            # Fit GP to plot
            if X_sample.shape[0] > 1:
                gpr.fit(X_sample, Y_sample)
                mu, sigma = gpr.predict(X, return_std=True)
                ei = self.expected_improvement(X, X_sample, Y_sample, gpr)
                
                # Plot GP and acquisition function
                ax1.clear()
                ax2.clear()
                
                # Plot GP
                ax1.plot(X, -self.objective_function(X), 'k--', label='Função objetiva')
                ax1.plot(X, -mu, color=colors[i], label=f'GP média iteração {i+1}')
                ax1.fill_between(X.ravel(), 
                               -(mu + 2*sigma), 
                               -(mu - 2*sigma), 
                               color=colors[i], alpha=0.1)
                ax1.scatter(X_sample, -Y_sample, c='red', marker='o', label='ObservaçÕes')
                ax1.scatter(X_next, -Y_next, c='black', marker='*', 
                          s=200, label='Próxima amostra')
                ax1.set_title('Processo gaussiano de observação')
                ax1.set_xlabel('Taxa de aprendizado')
                ax1.set_ylabel('Validação de custo negativa')
                ax1.legend()
                
                # Plot acquisition function
                ax2.plot(X, ei, color=colors[i], label=f'EI iter {i+1}')
                ax2.scatter(X_next, 0, c='black', marker='*', s=200)
                ax2.set_title('EI (Expected Improvement)')
                ax2.set_xlabel('Taxa de aprendizado')
                ax2.set_ylabel('EI')
                ax2.legend()
            
            plt.tight_layout()
            plt.pause(0.5)
            plt.savefig(f"gauss{i}")
        
        # Return best found value and location
        best_idx = np.argmin(Y_sample)
        return X_sample[best_idx], Y_sample[best_idx]

# Create and run visualization
visualizer = BayesianOptimizationVisualizer()
best_x, best_y = visualizer.plot_optimization(n_iterations=10)
print(f"\nBest learning rate found: {best_x[0]:.6f}")
print(f"Best validation loss: {-best_y:.6f}")