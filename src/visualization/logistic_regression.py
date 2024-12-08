import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

# Gerando dados de exemplo (2 classes, 2 features)
X, y = make_classification(
    n_samples=200, 
    n_features=2, 
    n_informative=2, 
    n_redundant=0, 
    n_clusters_per_class=1, 
    random_state=42
)

# Ajustando o modelo de regressão logística
modelo = LogisticRegression()
modelo.fit(X, y)

# Gerando uma malha para plotar a fronteira de decisão
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 200), 
    np.linspace(y_min, y_max, 200)
)
Z = modelo.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plotagem
plt.figure(figsize=(10, 6))

# Contorno preenchido para a fronteira de decisão
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')

# Plotando os pontos de cada classe
plt.scatter(
    X[y == 0, 0], X[y == 0, 1], 
    color='dodgerblue', edgecolor='k', 
    alpha=0.8, s=60, label='Classe 0'
)
plt.scatter(
    X[y == 1, 0], X[y == 1, 1], 
    color='crimson', edgecolor='k', 
    alpha=0.8, s=60, label='Classe 1'
)

# Rótulos dos eixos em português
plt.xlabel('Característica 1', fontsize=14, labelpad=10)
plt.ylabel('Característica 2', fontsize=14, labelpad=10)

# Legenda
plt.legend(loc='upper right', fontsize=12, frameon=True, facecolor='whitesmoke')

# Ajustes de grid e ticks
plt.grid(color='lightgray', linestyle='--', linewidth=0.5, alpha=0.7)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()
plt.show()
