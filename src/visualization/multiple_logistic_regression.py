import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Gerar dados sintéticos com quatro classes
X, y = make_blobs(
    n_samples=400,
    centers=4,
    n_features=2,
    cluster_std=1.5,
    random_state=42
)

# Definir os modelos
modelos = [
    ("Regressão logística multinomial", LogisticRegression(multi_class='multinomial', solver='lbfgs')),
    ("SVC (One-vs-One)", SVC(decision_function_shape='ovo')),
    ("Random forest", RandomForestClassifier(n_estimators=100, random_state=42))
]

# Ajustar cada modelo
for nome, modelo in modelos:
    modelo.fit(X, y)

# Criação da malha para visualizar as fronteiras
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

cores = ['dodgerblue', 'crimson', 'limegreen', 'gold']
classes = np.unique(y)

for ax, (nome, modelo) in zip(axes, modelos):
    # Predizer na malha
    Z = modelo.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plotar contorno preenchido
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='rainbow')
    
    # Plotar os pontos de cada classe
    for classe, cor in zip(classes, cores):
        ax.scatter(
            X[y == classe, 0], X[y == classe, 1],
            color=cor, edgecolor='k', s=60, alpha=0.8, label=f'Classe {classe}'
        )
        
    ax.set_xlabel('Característica 1', fontsize=12, labelpad=10)
    ax.set_ylabel('Característica 2', fontsize=12, labelpad=10)
    ax.set_title(nome, fontsize=14)
    ax.grid(color='lightgray', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.tick_params(labelsize=10)
    if ax is axes[0]:
        ax.legend(loc='upper right', fontsize=10, frameon=True, facecolor='whitesmoke')

plt.tight_layout()
plt.show()
