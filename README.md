# Relatório do projeto final

**Amanda de Mendonça Perez e Lucas Westfal**

*Ambos os integrantes do grupo contribuíram igualmente, majoritariamente por meio programação por pares.*

## Introdução

Este trabalho tem como finalidade treinar um algoritmo de *reinforcement learning* para resolver um problema multi-agente em um ambiente _speaker-listener_, a partir de um código base já fornecido (que implementa o algoritmo MATD3).
O objetivo é obter uma melhora na performance do código, seja pela implementação de outro algoritmo ou pelo ajuste das configurações do MATD3. Neste relatório, discutimos as mudanças exploradas e os resultados obtidos.

## Alterações feitas no código

Como foram dadas duas opções para a modificação do código, optamos por primeiramente explorar variações de configuração do MATD3, a fim de avaliar a necessidade de implementar outro algoritmo. Em nossos experimentos, verificamos que alguns ajustes nos hiperparâmetros já forneceram uma melhora no desempenho do algoritmo, de forma que optamos por manter o algoritmo inicial e apenas explorar diferentes configurações. 

Além dos ajustes nos hiperparâmetros, buscamos otimizar o treinamento vetorizando algumas operações. Também incluímos código para gerar algumas figuras extras, a fim de avaliar em mais detalhes o desempenho do modelo para diferentes configurações de hiperparâmetros. Nesta seção, discutimos em maior detalhe essas modificações.

### Hiperparâmetros

Para melhorar o modelo, testamos variar os seguintes hiperparâmetros:

- Configuração da rede:
    - `hidden_size`
- Inicialização dos hiperparâmetros:
    - `POPULATION_SIZE`
    - `BATCH_SIZE`
    - `EXPL_NOISE`
    - `LR_ACTOR`
    - `LR_CRITIC`
    - `GAMMA`
    - `MEMORY_SIZE`
    - `LEARN_STEP`
    - `TAU`
    - `POLICY_FREQ`
- Probabilidades de mutação:
    - `architecture`
    - `parameters`
    - `activation`
    - `rl_hp`
- Parâmetros do loop de treinamento:
    - `max_steps`
    - `learning_delay`
    - `evo_steps`
    - `eval_steps`
    - `eval_loop`

Para alguns desses, o desempenho resultante foi piorado ou o tempo de treinamento ficou insustentavelmente alto. Como consequência, optamos por retornar aos valores padrão.

Para os hiperparâmetros que permaneceram alterados na versão final do código, a tabela a seguir sumariza essas alterações e as justificativas para mantê-las.

| Hiperparâmetro | Valor padrão | Valor modificado | Justificativa |
| -------- | ----- | ----------- | - |
| `hidden_size` | [64]    | [128, 128]    | Melhorar capacidade de generalização.
| `BATCH_SIZE` | 128   |    4096     | Mais acurácia na estimação dos gradientes; acelerar convergência.
| `EXPL_NOISE` | 0.1   |   0.3       | Aumentar exploração.
| `MEMORY_SIZE` | 100,000  |   3,000,000   | Evitar overfitting.
| `max_steps`   | 2,000,000  | 400,000     | Limitações computacionais, pelo aumento do número de passos de evolução.
| `learning_delay` | 0 | 10,000      | Aumentar estabilidade do treinamento, quebrando correlação entre os passos.
| `evo_steps`   | 10,000 | 100,000     | Mais iterações a cada passo de evolução, aprendendo mais antes de julgar os parâmetros.
| `eval_steps`  | None | 1,000       | Diminuir tempo computacional e acelera evolução.
| `eval_loop`   | 1 | 10          | Diminuir a variância das pontuações.

Em resumo, as adaptações consistiram principalmente em aumentar o tamanho da rede, a memória, o tamanho dos batches e o número de passos a cada iteração de evolução, a fim de acelerar o aprendizado e melhorar o desempenho geral do modelo; ao mesmo tempo, alguns hiperparâmetros foram "enfraquecidos" a fim de permitir o treinamento com o poder computacional disponível. 

### Vetorização de operações

A fim de melhorar a eficiência do código e permitir que o modelo fosse treinado em menos tempo, foram feitas algumas modificações no arquivo [`main.py`](/main.py). A principal modificação foi a vetorização de algumas operações para permitir o uso de GPU e acelerar o treinamento. Além disso, a linha `agent_keys = env.agents`, que estava dentro do loop de treinamento, foi movida para fora, para evitar redundâncias e perda de eficiência. O código alterado está detalhado a seguir.

Versão inicial:
```
scores += np.sum(np.array(list(reward.values())).transpose(), axis=-1)

term_array = np.array(list(termination.values())).transpose()
trunc_array = np.array(list(truncation.values())).transpose()
```

Versão modificada:
```
stacked_rewards = np.stack([reward[agent] for agent in agent_keys])
scores += np.sum(stacked_rewards, axis=0)

term_array = np.stack([termination[agent] for agent in agent_keys]).T
trunc_array = np.stack([truncation[agent] for agent in agent_keys]).T
```

### Novos plots

Com a diminuição no número máximo de passos, o gráfico gerado pelo código inicial passou a ser menos esclarecedor quanto à evolução da pontuação. Por esse motivo, optamos por implementar outros dois plots que exibissem esses dados com maior granularidade. Para isso, no código de treinamento (em [`main.py`](/main.py)), foi inserida uma etapa de armazenamento das pontuações em um arquivo `full_training_history.npy`. Diferentemente do arquivo `training_scores_history.npy`, que armazena apenas as pontuações médias a cada iteração de evolução, este arquivo armazena os dados completos, para cada agente e cada iteração dentro das iterações de evolução. 

Utilizando esses dados, o script presente no arquivo [`plot_history.py`](/plot_history.py) gera duas novas imagens: `training_scores_evo_agents.png` exibe a evolução das pontuações médias de cada agente da população (neste caso, 4) ao longo das iterações de evolução; já `training_scores_complete_data.png` exibe a totalidade dos dados, mostrando como a pontuação evoluiu para cada agente dentro cada iteração de evolução. Esses gráficos são apresentados e melhor discutidos na seção de resultados.

## Resultados obtidos

A versão final do modelo demorou aproximadamente 56min17s para treinar e alcançou uma pontuação média acima de aproximadamente -33.43 ao final do treinamento, bem acima da pontuação média obtida com a configuração inicial, de cerca de -60. O gráfico a seguir exibe essa evolução.

![Evolução das pontuações por agente para cada iteração de evolução](https://github.com/LucasWestfal/Reinforcement-Learning-Final-Project/blob/main/models/MATD3/training_scores_evolution.png)
*Fig. 1: Evolução das pontuações médias ao longo das iterações de evolução.*

</br>

Essa evolução foi consistente entre os agentes utilizados pelo algoritmo evolutivo. O plot exibido a seguir mostra a evolução das pontuações para cada um dos agentes. Note que todos os quatro alcançaram pontuações acima de -60 nas duas últimas iterações de evolução.

![Evoluação das pontuações médias por agente](https://github.com/LucasWestfal/Reinforcement-Learning-Final-Project/blob/main/models/MATD3/training_scores_evo_agents.png)
*Fig. 2: Evoluação das pontuações médias de cada agente.*

</br>

Por fim, buscamos exibir ainda a evolueção para agente ao longo de cada uma das iterações de evolução, já que optamos por aumentar o número de passos em cada uma delas.

![Evolução das pontuações por agente para cada iteração de evolução](https://github.com/LucasWestfal/Reinforcement-Learning-Final-Project/blob/main/models/MATD3/training_scores_complete_data.png)
*Fig. 3: Evolução das pontuações de cada agente, para cada iteração de evolução.*

</br>

Nota-se acima que as pontuações obtidas vão se concentrando em pontuações menores com o passar das iterações de evolução, indicando o aprendizado das agentes.

Por fim, ao testar o modelo e gerar o gif, podemos notar alguns padrões no comportamento do listener. 

![GIF do jogo com o modelo treinado](https://github.com/LucasWestfal/Reinforcement-Learning-Final-Project/blob/main/videos/speaker_listener.gif)

*Fig. 4: Animação mostrando o comportamento do modelo treinado em 10 episódios.*

</br>

Na animação acima, o listener parece consistentemente buscar primeiro o centroide das posições das *landkarks*, para em seguida se mover em direção ao objetivo. Apesar de não alcançar consistentemente uma _landmark_, o agente consegue, ao menos, se aproximar de alguma delas. Esse política pode ter sido aprendida pelo uso de `eval_steps` diferente de `None`, pois isso faz com que apenas as primeiras iterações sejam consideradas para avaliar o modelo. Dessa forma, é possível que o comportamento ótimo seja, antes de tudo, procurar o centroide das _landmarks_.

Além desse resultado, uma versão anterior do modelo exibiu resultados ainda melhores, com pontuação média chegando a mais de -30 ao final do treinamento. Entretanto, ao tentar treinar o modelo novamente com a mesma configuração, o desempenho caiu, indicando que a melhora pode ter decorrido da estocasticidade de treinamento/evolução dos hiperparâmetros. O modelo, assim como os parâmetros obtidos, está disponível [nessa versão do repositório](https://github.com/LucasWestfal/Reinforcement-Learning-Final-Project/blob/8a28edb69556c761270603a78c5d3a40e7742f0d/). As imagens a seguir apresentam os resultados obtidos nesse modelo.

![Resultado anterior](https://github.com/LucasWestfal/Reinforcement-Learning-Final-Project/blob/8a28edb69556c761270603a78c5d3a40e7742f0d/models/MATD3/training_scores_evolution.png)
*Fig. 5: Evolução das pontuações médias ao longo das iterações de evolução para modelo anterior.*

</br>

![Resultado anterior](https://github.com/LucasWestfal/Reinforcement-Learning-Final-Project/blob/8a28edb69556c761270603a78c5d3a40e7742f0d/models/MATD3/training_scores_evo_agents.png)
*Fig. 6: Evoluação das pontuações médias de cada agente para modelo anterior.*

</br>

![Resultado anterior](https://github.com/LucasWestfal/Reinforcement-Learning-Final-Project/blob/8a28edb69556c761270603a78c5d3a40e7742f0d/models/MATD3/training_scores_complete_data.png)
*Fig. 7:  Evolução das pontuações de cada agente, para cada iteração de evolução para o modelo anterior.*

</br>


![Resultado anterior](https://github.com/LucasWestfal/Reinforcement-Learning-Final-Project/blob/8a28edb69556c761270603a78c5d3a40e7742f0d/videos/speaker_listener.gif)

*Fig. 8: Animação mostrando o comportamento do modelo treinado em 10 episódios para modelo anterior.*

</br>

---


até o momento: 
corrigimos os imports pra tirar modelo deprecado
corrigimos o create population, estava passando argumentos no lugar errado

testando hiperparametros:
    eval_loop = 1 -> 3
    eval_steps = None -> 100000
    "hidden_size": [64] -> 32
    max_steps = 2_000_000 -> 300_000

nao prestou
nova tentativa

    max_steps = 2_000_000 -> 300_000
    (voltar presets de NET_CONFIG)
    "MEMORY_SIZE": 100000 -> 1000000
    "BATCH_SIZE": 64 -> 512
    eval_steps = 100000
    evo_steps = 100_000
    learning_delay = 10_000
    "EXPL_NOISE": 0.3

Parece ter começado a aprender, vou deixar rodando por 2000000 steps. também fiz         "MEMORY_SIZE": 2000000,  # Max memory buffer size
demorou 9h4min pra rodar, resultados:
- loss melhorou bastante
- agora o agente ta buscando a média das bolas consistentemente
possiveis formas de melhorar:
- refinar learn_step
- aumentar tamanho do modelo para capturar comportamentos mais rebuscados


agora:
    "LEARN_STEP": 100,
    max_steps = 2_000_000 -> 300_000
mudei pra
    "LEARN_STEP": 10,

Mudei
    evo_steps = 100_000 -> 50_000 
    eval_steps = 100_000 -> 50_000

Incluí um novo esquema pra armazenar os scores e plotei a evolução de cada
agente separadamente, além disso, voltei para 100_000 em evo_steps e eval_steps, porque
fica muito melhor, e diminuí max_steps para 200_000 pro computador conseguir rodar

Mudei novamente para:
    evo_steps = 100_000
    eval_steps = 100_000
    max_steps = 2_000_000
mas ainda não rodei.

a fazer em seguida

    hidden_size: [64, 64]
    max_steps = 500_000
    evo_steps = 20_000
    "BATCH_SIZE": 4096
    e colocar condicional pra armazenar melhor iteração?
    

coisas que to otimizando no codigo:

ANTIGO:
scores += np.sum(np.array(list(reward.values())).transpose(), axis=-1)

term_array = np.array(list(termination.values())).transpose()
trunc_array = np.array(list(truncation.values())).transpose()

=> virou
agent_keys = env.agents (fora do training loop)

stacked_rewards = np.stack([reward[agent] for agent in agent_keys])
scores += np.sum(stacked_rewards, axis=0)

term_array = np.stack([termination[agent] for agent in agent_keys]).T
trunc_array = np.stack([truncation[agent] for agent in agent_keys]).T

Mudei `activation` de 0 para 0.2:
```activation=0.2,  # Probability of activation function mutation```
Descobri que não faz diferença pq o modelo MATD3 não suporta esse tipo de mutação, então voltei pra 0

mexi no batch size de novo pra 1024 e depois pra 2048 e de volta a 4096
mudei "POLICY_FREQ": 3 (era 2) /// voltei pra 2
mudei population_size para 5 /// voltei pra 4 pq o pc não tankou :(
também mudei as probabilidades de mutação (todas essas eram 0.2):

    architecture=0.1
    parameters=0.3
    rl_hp=0.3


Vou tentar mudar esses parâmetros para esses valores:
    "LR_ACTOR": 3e-4,
    "LR_CRITIC": 1e-3,
    "TAU": 0.005
    "GAMMA": 0.97
(antes estava:
    "LR_ACTOR": 0.0001,
    "LR_CRITIC": 0.001,
    "TAU": 0.01
    "GAMMA": 0.95
) /// esquece, voltei para o que estava antes

botei eval_steps = None (estava 10_000) /// voltei pra 10_000 e depois pra 1_000
