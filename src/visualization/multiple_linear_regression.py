import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D

# Geração de dados de exemplo
np.random.seed(42)
X1 = np.random.rand(50, 1) * 10   # Valores entre 0 e 10 para X1
X2 = np.random.rand(50, 1) * 10   # Valores entre 0 e 10 para X2
Y = 4 * X1 + 2 * X2 + 5 + np.random.randn(50, 1) * 3  # Modelo linear com ruído

X = np.hstack([X1, X2])  # Matriz de atributos

# Ajuste do modelo de regressão linear múltipla
modelo = LinearRegression()
modelo.fit(X, Y)

# Criação de uma grade de pontos para a superfície de regressão
X1_grid = np.linspace(0, 10, 20)
X2_grid = np.linspace(0, 10, 20)
X1_grid, X2_grid = np.meshgrid(X1_grid, X2_grid)

# Previsões do modelo na grade
Y_pred = modelo.predict(np.column_stack([X1_grid.ravel(), X2_grid.ravel()]))
Y_pred = Y_pred.reshape(X1_grid.shape)

# Erro médio quadrático
mse = mean_squared_error(Y, modelo.predict(X))

# Plotagem em 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Pontos de dados reais
ax.scatter(X1, X2, Y, color='dodgerblue', edgecolor='k', alpha=0.7, s=80, label='Dados Reais')

# Superfície de regressão
ax.plot_surface(X1_grid, X2_grid, Y_pred, color='crimson', alpha=0.5, edgecolor='none', label='Plano de Regressão')

# Personalizações do gráfico
# Sem título acima do gráfico, conforme solicitado

ax.set_xlabel('Valor da Variável Independente X1', fontsize=12, labelpad=10)
ax.set_ylabel('Valor da Variável Independente X2', fontsize=12, labelpad=10)
ax.set_zlabel('Valor da Variável Dependente (Y)', fontsize=12, labelpad=10)

ax.tick_params(axis='both', which='major', labelsize=10)
ax.legend(loc='upper left', fontsize=12)

# Ajuste do layout
plt.tight_layout()
plt.show()
