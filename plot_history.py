import os
import numpy as np
import matplotlib.pyplot as plt

# Carregar dados do histórico completo de treinamento
file_path = 'models/MATD3/full_training_history.npy' 
data = np.load(file_path)

# Definir tamanho dos dados
pop_size = data.shape[1]
num_iter = data.shape[0]


# Plotar e salvar a evolução das pontuações para cada agente
# Definições de estilo
colors = ["#d1534a", "#f29a1e", "#4f9d74", "#2e5d9f"]
line_styles = ["solid", "--", (5,(10,3)), "dashdot"]

# Plotar evolução das pontuações médias para cada agente
plt.figure(figsize=(12, 6))
for i in range(pop_size):
    plt.plot(data[:,i].mean(axis=1), linewidth=2, label=f"Agent {i+1}", c=colors[i], ls=line_styles[i])
    plt.title('Evolução das Pontuações de Cada Agente Durante o Treinamento', fontsize=14)
    plt.xlabel('Iterações de Evolução', fontsize=12)
    plt.ylabel('Pontuação Média de Cada Agente', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.legend()

# Salvar figura
path = "./models/MATD3"
plot_path_1 = os.path.join(path, "training_scores_evo_agents.png")
plt.savefig(plot_path_1, dpi=300, bbox_inches='tight')

# Plotar evolução das pontuações para cada agente e cada iteração de evolução
fig, ax = plt.subplots(nrows=pop_size, ncols=num_iter, figsize=(12,6), sharex=True, sharey=True)
for i in range(pop_size*num_iter):
    # Plotar dados
    ax[i//num_iter, i%num_iter].plot(data[i%num_iter,i//num_iter,:],linewidth=1,c=colors[-1])
    ax[i//num_iter, i%num_iter].grid(True, alpha=0.3)
    ax[i//num_iter, i%num_iter].yaxis.tick_right()
    ax[i//num_iter, i%num_iter].yaxis.set_label_position("right")

    # Exibir rótulo dos eixos e subtítulos
    if i//num_iter == 0:
        ax[0, i].set_title(f'Iteração de evolução {i+1}', size=12)
    
    if i%3 == 0:
        ax[i//num_iter, 0].text(-1500, -500, f'Agente {i//num_iter + 1}', rotation=0, size=12)
        ax[i//num_iter, 0].tick_params(labelright=False, right=True)
    
    if i%3 == 2:
        ax[i//num_iter, 2].set_ylabel(f'Pontuação', rotation=0, size=9)
        ax[i//num_iter, 2].yaxis.set_label_coords(1.15, 1.12)
        ax[i//num_iter, 2].tick_params(labelright=True, right=True)

    if i//3 == 3:
        ax[num_iter, i%num_iter].set_xlabel("Iterações", size=9)

fig.suptitle("Evolução das Pontuações de Cada Agente da População\n", size=14)

# Salvar figura
plot_path_2 = os.path.join(path, "training_scores_complete_data.png")
plt.savefig(plot_path_2, dpi=300)
