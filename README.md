# Relatório do projeto final

**Amanda de Mendonça Perez e Lucas Westfal**

*Ambos os integrantes do grupo contribuíram igualmente, majoritariamente por meio programação por pares.*

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
- agora o agente ta buscan
# line_styles = ["solid", "--", "dotted", "dashdot"]do a média das bolas consistentemente
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

a fazer em seguida



    "latent_dim": 128
    "hidden_size": [128]
    max_steps = 2_000_000
