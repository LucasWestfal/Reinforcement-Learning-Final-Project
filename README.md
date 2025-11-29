# Relatório do projeto final

**Amanda de Mendonça Perez e Lucas Westfal**

*Ambos os integrantes do grupo contribuíram igualmente, majoritariamente por meio programação por pares.*

## Introdução

Lorem

## Alterações feitas no código

Lorem

### Vetorização de operações

Lorem ipsum

### Hiperparâmetros

| Hiperparâmetro | Valor anterior | Valor final | Justificativa |
| -------- | ----- | ----------- | - |
| `eval_steps`      | -    | -     | Lorem
| B        | 2     |      a       |


### Novos plots

### Outras alterações exploradas, mas descartadas

## Resultados obtidos

[Inserir figuras aqui]


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

botei eval_steps = None (estava 10_000) /// voltei pra 10_000