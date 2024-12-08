import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['figure.dpi'] = 100

fig = plt.figure(figsize=(14, 8))
ax = plt.subplot(111)

G = nx.DiGraph()

# Definir as camadas do autoencoder
# Exemplo: Input (6 nós) -> Encoder hidden (4 nós) -> Latent (2 nós) -> Decoder hidden (4 nós) -> Output (6 nós)

n_input = 6
n_enc_hidden = 4
n_latent = 2
n_dec_hidden = 4
n_output = 6

# Posições horizontais de cada camada
x_input = 0
x_enc_hidden = 2
x_latent = 4
x_dec_hidden = 6
x_output = 8

# Função auxiliar para posicionar camadas verticalmente centralizadas
def layer_positions(num_nodes, x_center, y_span=5):
    # centralizar nós em torno de zero no eixo y
    offsets = np.linspace((num_nodes-1)/2, -(num_nodes-1)/2, num_nodes)
    return [(x_center, o) for o in offsets]

# Criar nós
input_nodes = [f'in_{i+1}' for i in range(n_input)]
enc_hidden_nodes = [f'enc_{i+1}' for i in range(n_enc_hidden)]
latent_nodes = [f'lat_{i+1}' for i in range(n_latent)]
dec_hidden_nodes = [f'dec_{i+1}' for i in range(n_dec_hidden)]
output_nodes = [f'out_{i+1}' for i in range(n_output)]

# Atribuir posições
pos_input = layer_positions(n_input, x_input)
pos_enc_hidden = layer_positions(n_enc_hidden, x_enc_hidden)
pos_latent = layer_positions(n_latent, x_latent)
pos_dec_hidden = layer_positions(n_dec_hidden, x_dec_hidden)
pos_output = layer_positions(n_output, x_output)

for node, p in zip(input_nodes, pos_input):
    G.add_node(node, pos=p)
for node, p in zip(enc_hidden_nodes, pos_enc_hidden):
    G.add_node(node, pos=p)
for node, p in zip(latent_nodes, pos_latent):
    G.add_node(node, pos=p)
for node, p in zip(dec_hidden_nodes, pos_dec_hidden):
    G.add_node(node, pos=p)
for node, p in zip(output_nodes, pos_output):
    G.add_node(node, pos=p)

# Criar pesos aleatórios para as conexões
# Encoder conexões: input->enc_hidden e enc_hidden->latent
# Decoder conexões: latent->dec_hidden e dec_hidden->output
np.random.seed(42)
def create_edges(from_nodes, to_nodes):
    edges = []
    weights = np.random.randn(len(from_nodes), len(to_nodes))
    for i, fn in enumerate(from_nodes):
        for j, tn in enumerate(to_nodes):
            w = weights[i, j]
            edges.append((fn, tn, {'weight': w}))
    return edges

G.add_edges_from(create_edges(input_nodes, enc_hidden_nodes))
G.add_edges_from(create_edges(enc_hidden_nodes, latent_nodes))
G.add_edges_from(create_edges(latent_nodes, dec_hidden_nodes))
G.add_edges_from(create_edges(dec_hidden_nodes, output_nodes))

pos = nx.get_node_attributes(G, 'pos')
edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
vmin, vmax = min(edge_weights), max(edge_weights)
if vmin == vmax:
    vmin, vmax = vmin - 1e-9, vmax + 1e-9
cmap = plt.cm.coolwarm

# Desenhar arestas
for (u, v, d) in G.edges(data=True):
    start = pos[u]
    end = pos[v]
    w = d['weight']
    norm_w = (w - vmin) / (vmax - vmin)
    color = cmap(norm_w)
    linewidth = 1.5 + (abs(w)/max(abs(vmin), abs(vmax))) * 2.5
    plt.plot([start[0], end[0]], [start[1], end[1]],
             color=color, linewidth=linewidth, alpha=0.9)

# Desenhar nós
def draw_nodes(nodes, positions, color):
    nx.draw_networkx_nodes(G, positions,
                           nodelist=nodes,
                           node_color=color,
                           node_size=1500,
                           node_shape='o',
                           edgecolors='k',
                           linewidths=2)

draw_nodes(input_nodes, pos, 'lightblue')
draw_nodes(enc_hidden_nodes, pos, 'lightgreen')
draw_nodes(latent_nodes, pos, 'yellow')
draw_nodes(dec_hidden_nodes, pos, 'lightgreen')
draw_nodes(output_nodes, pos, 'lightcoral')

# Labels nos nós
labels = {}
for n in input_nodes:
    labels[n] = n.replace('in_', 'x')
for n in enc_hidden_nodes:
    labels[n] = n.replace('enc_', 'h_e')
for n in latent_nodes:
    labels[n] = n.replace('lat_', 'z')
for n in dec_hidden_nodes:
    labels[n] = n.replace('dec_', 'h_d')
for n in output_nodes:
    labels[n] = n.replace('out_', 'x̂')

nx.draw_networkx_labels(G, pos, labels=labels, font_size=14, font_weight='bold')

ax.axis('off')


# Legendas das camadas
plt.text(x_input, max(p[1] for p in pos_input)+1.2, 'Entrada', fontsize=14, ha='center', fontweight='bold', color='black')
plt.text(x_enc_hidden, max(p[1] for p in pos_enc_hidden)+1.2, 'Codificador\n(Encoder)', fontsize=14, ha='center', fontweight='bold', color='black')
plt.text(x_latent, max(p[1] for p in pos_latent)+1.2, 'Espaço latente\n(Latent space)', fontsize=14, ha='center', fontweight='bold', color='black')
plt.text(x_dec_hidden, max(p[1] for p in pos_dec_hidden)+1.2, 'Decodificador\n(Decoder)', fontsize=14, ha='center', fontweight='bold', color='black')
plt.text(x_output, max(p[1] for p in pos_output)+1.2, 'Saída\n(Reconstrução)', fontsize=14, ha='center', fontweight='bold', color='black')

# Barra de cores
plt.subplots_adjust(right=0.85)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))

# Equação do objetivo (reconstrução)
eq_text = r'$\min_{w,b}\ \sum_{i} \| x^{(i)} - \hat{x}^{(i)} \|^{2}$'
plt.figtext(0.5, 0.05, eq_text, ha='center', fontsize=27)

plt.savefig('autoencoder.png')