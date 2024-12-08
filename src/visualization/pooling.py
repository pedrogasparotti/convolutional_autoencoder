import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Configuração de estilo
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14

# Criar matriz de entrada (6x6)
entrada = np.array([
    [1, 1, 2, 4, 5, 6],
    [2, 2, 3, 5, 7, 8],
    [3, 3, 4, 6, 8, 9],
    [4, 4, 5, 7, 9, 10],
    [5, 5, 6, 8, 10, 11],
    [6, 6, 7, 9, 11, 12]
])

# Função de max pooling
def max_pooling(matriz, tamanho=2, stride=2):
    h, w = matriz.shape
    saida_h = (h - tamanho) // stride + 1
    saida_w = (w - tamanho) // stride + 1
    saida = np.zeros((saida_h, saida_w))
    
    for i in range(0, saida_h):
        for j in range(0, saida_w):
            janela = matriz[i*stride:i*stride+tamanho, 
                          j*stride:j*stride+tamanho]
            saida[i, j] = np.max(janela)
    return saida

# Aplicar max pooling
saida = max_pooling(entrada)

# Criar visualização
plt.figure(figsize=(12, 5))

# Plotar matriz de entrada
plt.subplot(121)
plt.imshow(entrada, cmap='winter')
plt.title('Matriz de entrada', pad=20, fontweight='bold')
plt.colorbar(label='Intensidade')
plt.grid(True)

# Adicionar valores numéricos
for i in range(entrada.shape[0]):
    for j in range(entrada.shape[1]):
        plt.text(j, i, entrada[i, j],
                ha="center", va="center", 
                color="red" if entrada[i, j] > np.mean(entrada) else "orange",
                fontsize=14,  # Aumentado o tamanho da fonte
                fontweight='extra bold')

# Adicionar grade para mostrar as regiões de pooling
ax = plt.gca()
for i in range(0, entrada.shape[0], 2):
    for j in range(0, entrada.shape[1], 2):
        rect = Rectangle((j-0.5, i-0.5), 2, 2, fill=False, 
                        edgecolor='red', linewidth=2)
        ax.add_patch(rect)

# Plotar resultado do Max Pooling
plt.subplot(122)
plt.imshow(saida, cmap='winter')
plt.title('Resultado do max pooling (2x2)', pad=20, fontweight='bold')
plt.colorbar(label='Intensidade')
plt.grid(True)

# Adicionar valores numéricos no resultado
for i in range(saida.shape[0]):
    for j in range(saida.shape[1]):
        plt.text(j, i, f'{saida[i, j]:.0f}',
                ha="center", va="center",
                color="red" if saida[i, j] > np.mean(saida) else "orange",
                fontsize=14,  # Aumentado o tamanho da fonte
                fontweight='extra bold')

plt.tight_layout()
plt.show()

# Imprimir valores
print("\nMatriz de Entrada:")
print(entrada)
print("\nResultado do Max Pooling:")
print(saida)