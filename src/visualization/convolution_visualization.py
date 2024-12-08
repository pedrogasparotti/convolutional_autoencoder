import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Configurar fonte padrão maior para os textos em português
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14

# Criar imagem de entrada (5x5)
imagem_entrada = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
])

# Definir o kernel (3x3)
kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

# Função para realizar a convolução
def convolucao2d(imagem, kernel):
    forma_saida = (imagem.shape[0] - kernel.shape[0] + 1,
                  imagem.shape[1] - kernel.shape[1] + 1)
    saida = np.zeros(forma_saida)
    
    for i in range(forma_saida[0]):
        for j in range(forma_saida[1]):
            saida[i, j] = np.sum(
                imagem[i:i+kernel.shape[0], j:j+kernel.shape[1]] * kernel
            )
    return saida

# Computar convolução
saida = convolucao2d(imagem_entrada, kernel)

# Criar visualização
plt.figure(figsize=(15, 5))

# Plotar imagem de entrada
plt.subplot(131)
plt.imshow(imagem_entrada, cmap='gray')
plt.title('Imagem de Entrada', pad=20, fontweight='bold')
plt.colorbar(label='Intensidade')
plt.grid(True)

# Plotar kernel
plt.subplot(132)
plt.imshow(kernel, cmap='RdBu')
plt.title('Kernel (Filtro)', pad=20, fontweight='bold')
plt.colorbar(label='Peso')
plt.grid(True)

# Plotar saída
plt.subplot(133)
plt.imshow(saida, cmap='gray')
plt.title('Resultado da Convolução', pad=20, fontweight='bold')
plt.colorbar(label='Intensidade')
plt.grid(True)

# Adicionar retângulo para mostrar a janela de convolução
ax = plt.gca()
ret = Rectangle((0, 0), 3, 3, linewidth=2, edgecolor='r', facecolor='none')
ax.add_patch(ret)

plt.tight_layout()
plt.show()

# Imprimir os valores numéricos
print("\nMatriz de Entrada:")
print(imagem_entrada)
print("\nKernel (Filtro):")
print(kernel)
print("\nResultado da Convolução:")
print(saida)