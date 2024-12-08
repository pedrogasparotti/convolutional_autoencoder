import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Exemplo 1: K-Means com duas variáveis (2D)
np.random.seed(42)
X_2D = np.random.rand(200, 2) * 10  # Dados aleatórios 2D
kmeans_2D = KMeans(n_clusters=3, random_state=42)
y_pred_2D = kmeans_2D.fit_predict(X_2D)

# Exemplo 2: K-Means com uma variável (1D)
X_1D = np.random.rand(100, 1) * 10
kmeans_1D = KMeans(n_clusters=3, random_state=42)
y_pred_1D = kmeans_1D.fit_predict(X_1D)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plotagem do exemplo 1 (2D)
ax1 = axes[0]
ax1.scatter(X_2D[:, 0], X_2D[:, 1], c=y_pred_2D, cmap='viridis', edgecolor='k', s=80, alpha=0.8)
ax1.scatter(kmeans_2D.cluster_centers_[:, 0], kmeans_2D.cluster_centers_[:, 1], 
            s=200, c='red', edgecolor='k', marker='X', label='Centroides')
ax1.set_xlabel('Característica 1', fontsize=14, labelpad=10)
ax1.set_ylabel('Característica 2', fontsize=14, labelpad=10)
ax1.set_title('K-Means com duas variáveis', fontsize=16)
ax1.grid(color='lightgray', linestyle='--', linewidth=0.5)
ax1.legend(loc='upper right', fontsize=12)

# Plotagem do exemplo 2 (1D)
ax2 = axes[1]
ax2.scatter(X_1D, np.zeros_like(X_1D), c=y_pred_1D, cmap='viridis', edgecolor='k', s=80, alpha=0.8)
# Plotar centroides (apenas no eixo X, no Y=0)
ax2.scatter(kmeans_1D.cluster_centers_, np.zeros_like(kmeans_1D.cluster_centers_), 
            s=200, c='red', edgecolor='k', marker='X', label='Centroides')
ax2.set_xlabel('Característica única', fontsize=14, labelpad=10)
ax2.set_yticks([])  # Sem eixo y significativo para 1D
ax2.set_title('K-Means com uma variável', fontsize=16)
ax2.grid(color='lightgray', linestyle='--', linewidth=0.5)
ax2.legend(loc='upper right', fontsize=12)

plt.tight_layout()
plt.show()
