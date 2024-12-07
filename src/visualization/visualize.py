import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def funcao_simplificada(x, y):
    """
    Função simplificada: f(x, y) = sqrt(x^2 + y^2) + 0.5 * sin(2 * sqrt(x^2 + y^2))
    """
    r = np.sqrt(x**2 + y**2)
    return r + 0.5 * np.sin(2 * r)

def gradiente_simplificada(x, y, h=1e-7):
    """
    Calcula o gradiente da função simplificada usando diferenças finitas.
    """
    grad_x = (funcao_simplificada(x + h, y) - funcao_simplificada(x - h, y)) / (2 * h)
    grad_y = (funcao_simplificada(x, y + h) - funcao_simplificada(x, y - h)) / (2 * h)
    return np.array([grad_x, grad_y])

# Criação do grid para visualização
x = np.linspace(-5, 5, 200)
y = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(x, y)
Z = funcao_simplificada(X, Y)

# Parâmetros do Adam
taxa_aprendizado = 0.1  # Taxa de aprendizado inicial
beta1 = 0.9  # Decaimento para o momento (primeiro momento)
beta2 = 0.999  # Decaimento para a escala RMS (segundo momento)
epsilon = 1e-8  # Termo para evitar divisões por zero
max_iteracoes = 1000
tolerancia = 1e-6

# Novo ponto inicial
ponto_inicial = np.array([7.3, -4.0])  # Atualize o ponto inicial aqui
posicao_atual = ponto_inicial
m = np.zeros_like(posicao_atual)  # Inicializar o momento
v = np.zeros_like(posicao_atual)  # Inicializar o RMS
caminho = [posicao_atual.copy()]

# Execução do Adam
for t in range(1, max_iteracoes + 1):
    gradiente = gradiente_simplificada(posicao_atual[0], posicao_atual[1])
    m = beta1 * m + (1 - beta1) * gradiente  # Atualização do momento
    v = beta2 * v + (1 - beta2) * (gradiente**2)  # Atualização do RMS
    m_corrigido = m / (1 - beta1**t)  # Correção do momento
    v_corrigido = v / (1 - beta2**t)  # Correção do RMS
    nova_posicao = posicao_atual - taxa_aprendizado * m_corrigido / (np.sqrt(v_corrigido) + epsilon)
    if np.linalg.norm(nova_posicao - posicao_atual) < tolerancia:
        break
    posicao_atual = nova_posicao
    caminho.append(posicao_atual.copy())

# Conversão do caminho em array para visualização
caminho = np.array(caminho)
z_caminho = funcao_simplificada(caminho[:, 0], caminho[:, 1])

# Criação do gráfico 3D
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Superfície da função
surf = ax.plot_surface(X, Y, Z, cmap='winter', alpha=0.8, edgecolor='none')

# Trajeto do gradiente descendente (Adam)
ax.plot(caminho[:, 0], caminho[:, 1], z_caminho, color='gold', marker='o', markersize=5, linewidth=3, label='Trajeto do Gradiente (Adam)', zorder=10)

# Destacar o ponto inicial
ax.scatter(caminho[0, 0], caminho[0, 1], z_caminho[0], color='red', s=100, label='Ponto Inicial', zorder=11)
ax.text(caminho[0, 0], caminho[0, 1], z_caminho[0] + 0.5, 'Início', color='red', zorder=11)

# Destacar o ponto final
ax.scatter(caminho[-1, 0], caminho[-1, 1], z_caminho[-1], color='yellow', s=100, label='Ponto Final', zorder=11)
ax.text(caminho[-1, 0], caminho[-1, 1], z_caminho[-1] + 0.5, 'Final', color='yellow', zorder=11)

# Personalização do gráfico
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_zlabel('f(x, y)', fontsize=12)
ax.set_title('Descida de gradiente', fontsize=15, pad=20)

# Limites do gráfico
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(np.min(Z), np.max(Z))

# Barra de cores
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='f(x, y)')

# Legenda
ax.legend()

# Ajustar layout
plt.tight_layout()

# Exibir o gráfico
plt.show()

# Exibir informações do resultado
print(f"Ponto final do gradiente: {posicao_atual}")
print(f"Valor da função no ponto final: {funcao_simplificada(posicao_atual[0], posicao_atual[1])}")
