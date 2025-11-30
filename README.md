# Relatório do projeto final

**Amanda de Mendonça Perez e Lucas Westfal**

*Ambos os integrantes do grupo contribuíram igualmente, majoritariamente por meio de programação por pares.*

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
| `eval_steps`  | None | 1,000       | Diminuir tempo computacional e acelerar evolução.
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

Utilizando esses dados, o script presente no arquivo [`plot_history.py`](/plot_history.py) gera duas novas imagens: `training_scores_evo_agents.png` exibe a evolução das pontuações médias de cada agente da população (neste caso, 4) ao longo das iterações de evolução; já `training_scores_complete_data.png` exibe a totalidade dos dados, mostrando como a pontuação evoluiu para cada agente dentro de cada iteração de evolução. Esses gráficos são apresentados e melhor discutidos na seção de resultados.

## Resultados obtidos

A versão final do modelo demorou aproximadamente 56min17s para treinar e alcançou uma pontuação média acima de aproximadamente -33.43 ao final do treinamento, bem acima da pontuação média obtida com a configuração inicial, de cerca de -60. O gráfico a seguir exibe essa evolução.

![Evolução das pontuações por agente para cada iteração de evolução](https://github.com/LucasWestfal/Reinforcement-Learning-Final-Project/blob/main/models/MATD3/training_scores_evolution.png)
*Fig. 1: Evolução das pontuações médias ao longo das iterações de evolução.*

</br>

Essa evolução foi consistente entre os agentes utilizados pelo algoritmo evolutivo. O plot exibido a seguir mostra a evolução das pontuações para cada um dos agentes. Note que todos os quatro alcançaram pontuações acima de -60 nas duas últimas iterações de evolução.

![Evoluação das pontuações médias por agente](https://github.com/LucasWestfal/Reinforcement-Learning-Final-Project/blob/main/models/MATD3/training_scores_evo_agents.png)
*Fig. 2: Evolução das pontuações médias de cada agente.*

</br>

Por fim, buscamos exibir ainda a evolução para cada agente ao longo de cada uma das iterações de evolução, já que optamos por aumentar o número de passos em cada uma delas.

![Evolução das pontuações por agente para cada iteração de evolução](https://github.com/LucasWestfal/Reinforcement-Learning-Final-Project/blob/main/models/MATD3/training_scores_complete_data.png)
*Fig. 3: Evolução das pontuações de cada agente, para cada iteração de evolução.*

</br>

Nota-se acima que as pontuações obtidas vão se concentrando em pontuações menores com o passar das iterações de evolução, indicando o aprendizado das agentes.

Por fim, ao testar o modelo e gerar o gif, podemos notar alguns padrões no comportamento do listener. 

![GIF do jogo com o modelo treinado](https://github.com/LucasWestfal/Reinforcement-Learning-Final-Project/blob/main/videos/speaker_listener.gif)

*Fig. 4: Animação mostrando o comportamento do modelo treinado em 10 episódios.*

</br>

Na animação acima, o listener parece consistentemente buscar primeiro o centroide das posições das *landmarks*, para em seguida se mover em direção ao objetivo. Apesar de não alcançar consistentemente uma _landmark_, o agente consegue, ao menos, se aproximar de alguma delas. Esse política pode ter sido aprendida pelo uso de `eval_steps` diferente de `None`, pois isso faz com que apenas as primeiras iterações sejam consideradas para avaliar o modelo. Dessa forma, é possível que o comportamento ótimo seja, antes de tudo, procurar o centroide das _landmarks_.

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
