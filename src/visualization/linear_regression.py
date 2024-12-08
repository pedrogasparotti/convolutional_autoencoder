import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Dados de exemplo
np.random.seed(42)
x = np.random.rand(50, 1) * 10  # Valores entre 0 e 10
y = 3 * x + 7 + np.random.randn(50, 1) * 5  # Linha com ruído

# Ajustando o modelo de regressão linear
modelo = LinearRegression()
modelo.fit(x, y)

# Previsões
x_pred = np.linspace(0, 10, 100).reshape(-1, 1)
y_pred = modelo.predict(x_pred)

# Erro médio quadrático
mse = mean_squared_error(y, modelo.predict(x))

# Visualização
plt.figure(figsize=(10, 6))

# Scatter de pontos reais
plt.scatter(
    x, y, 
    color='dodgerblue', 
    edgecolor='k', 
    alpha=0.7, 
    s=80, 
    label='Dados Reais'
)

# Linha de regressão
plt.plot(
    x_pred, y_pred, 
    color='crimson', 
    linewidth=2.5, 
    linestyle='--', 
    label='Regressão Linear'
)

# Remover título acima do gráfico
# plt.title('Ajuste de Regressão Linear', ... ) # Removido

# Rótulos dos eixos (em português)
plt.xlabel('Valor da variável independente (X)', fontsize=18, labelpad=10)
plt.ylabel('Valor da variável dependente (Y)', fontsize=18, labelpad=10)

# Legenda
plt.legend(loc='upper left', fontsize=12, frameon=True, shadow=True, facecolor='whitesmoke')

# Grade clara
plt.grid(color='lightgray', linestyle='--', linewidth=0.5, alpha=0.7)

# Ajustes de ticks
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Linhas horizontais e verticais auxiliares
plt.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.6)
plt.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.6)

plt.tight_layout()
plt.show()
