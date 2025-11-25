import os
import numpy as np
import matplotlib.pyplot as plt

# Specify the path to your .npy file
file_path = 'models/MATD3/full_training_history.npy' 

# Load the .npy file
data = np.load(file_path)


# Plotar e salvar a evolução das pontuações
colors = ["black", "red", "blue", "green"]
# line_styles = ["solid", "--", "dotted", "dashdot"]
plt.figure(figsize=(12, 6))
for i in range(4):
    # plt.fill_between([0.0, 1.0], data[:,i].mean(axis=1) - data[:,i].mean(axis=1), data[:,i].mean(axis=1) + data[:,i].mean(axis=1), alpha=0.3, color=colors[i])
    plt.plot(data[:,i].mean(axis=1), linewidth=2, label=f"Agent {i+1}", c=colors[i])
    print(data[:,i].mean(axis=1))
    print(data[:,i].std(axis=1))
    plt.title('Evolução das Pontuações de Cada Agente Durante o Treinamento', fontsize=14)
    plt.xlabel('Iterações de Evolução', fontsize=12)
    plt.ylabel('Pontuação Média da População', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.legend()

path = "./models/MATD3"
plot_path = os.path.join(path, "training_scores_evo_agents.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')