import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Definir estilo claro e simples
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['figure.dpi'] = 100

# Criar figura
fig = plt.figure(figsize=(12, 8))
ax = plt.subplot(111)

# Criar grafo direcionado
G = nx.DiGraph()

# Número de nós em cada camada
n_input = 4
n_output = 3

# Definir posições para nós de entrada e saída de forma simétrica
# Ajuste: centralizar nós ao redor de y=0
# Por exemplo, se n_input=4, posições: 1.5, 0.5, -0.5, -1.5
input_y_positions = np.linspace((n_input-1)/2, -(n_input-1)/2, n_input)
output_y_positions = np.linspace((n_output-1)/2, -(n_output-1)/2, n_output)

# Adicionar nós de entrada
input_nodes = [f'x{i+1}' for i in range(n_input)]
for i, node in enumerate(input_nodes):
    G.add_node(node, layer=0, pos=(0, input_y_positions[i]))

# Adicionar nós de saída
output_nodes = [f'y{i+1}' for i in range(n_output)]
for i, node in enumerate(output_nodes):
    G.add_node(node, layer=1, pos=(2, output_y_positions[i]))

# Gerar pesos aleatórios
weights = np.random.randn(n_input, n_output)

# Adicionar conexões com atributos de peso
edges = []
for i, input_node in enumerate(input_nodes):
    for j, output_node in enumerate(output_nodes):
        weight = weights[i, j]
        edges.append((input_node, output_node, {'weight': weight}))
G.add_edges_from(edges)

# Obter posições dos nós
pos = nx.get_node_attributes(G, 'pos')

# Extrair pesos e normalizar para o colormap
edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
vmin, vmax = min(edge_weights), max(edge_weights)
if vmin == vmax:
    # Caso raro se todos os pesos forem iguais
    vmin, vmax = vmin-1e-9, vmax+1e-9 
cmap = plt.cm.coolwarm

# Desenhar arestas manualmente para controlar a cor e a espessura
for (u, v, d) in G.edges(data=True):
    start = pos[u]
    end = pos[v]
    w = d['weight']
    # Normalizar peso para [0, 1] no colormap
    norm_w = (w - vmin) / (vmax - vmin)
    color = cmap(norm_w)
    # Ajustar largura baseada na magnitude do peso
    linewidth = 2 + (abs(w) / max(abs(vmin), abs(vmax))) * 3
    plt.plot([start[0], end[0]], [start[1], end[1]], 
             color=color, linewidth=linewidth, alpha=0.9)

# Desenhar nós de entrada
nx.draw_networkx_nodes(G, pos,
                       nodelist=input_nodes,
                       node_color='lightblue',
                       node_size=1800,
                       node_shape='o',
                       edgecolors='k',
                       linewidths=2)

# Desenhar nós de saída
nx.draw_networkx_nodes(G, pos,
                       nodelist=output_nodes,
                       node_color='lightcoral',
                       node_size=1800,
                       node_shape='o',
                       edgecolors='k',
                       linewidths=2)

# Adicionar labels nos nós
nx.draw_networkx_labels(G, pos, font_size=16, font_weight='bold')

# Título
plt.title('Camada Totalmente Conectada (Fully Connected Layer)', pad=20)

# Legendas das camadas
plt.text(-0.25, max(input_y_positions)+0.8, 'Camada de Entrada',
         fontsize=14, ha='center', fontweight='bold', color='darkblue')
plt.text(2.25, max(output_y_positions)+0.8, 'Camada de Saída',
         fontsize=14, ha='center', fontweight='bold', color='darkred')

ax.axis('off')

# Ajuste da posição dos eixos para a barra de cor
plt.subplots_adjust(right=0.85)

# Adicionar barra de cores
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
cbar_ax = fig.add_axes([0.88, 0.25, 0.03, 0.5])
cbar = plt.colorbar(sm, cax=cbar_ax)
cbar.set_label('Peso da Conexão (w)', fontsize=14)
cbar.ax.tick_params(labelsize=12)

# Equação
eq_text = r'$y_j = max(0, \sum_{i=1}^{4} w_{ij}x_i + b_j)$'
plt.figtext(0.5, 0.05, eq_text, ha='center', fontsize=18)

plt.savefig("fcl_relu")
