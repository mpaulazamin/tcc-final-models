## Multi agent - RL como ferramenta de otimização em tempo real

Modelo com malha de inventário para o nível do tanque e com controle liga-desliga do boiler. Com malha cascata, sem split-range.

![image](https://github.com/mpaulazamin/tcc-final-models/blob/multi_agent_camada_rto_iqb_sem_split_range/imagens/chuveiro_controle_t4a_sem_split.jpg)

### Espaço de ações

- SPTs: 30 a 40 - contínuo
- SPTq: 30 a 70 - contínuo
- xs: 0.01 a 0.99 - contínuo
- Sr: 0.01 a 0.99 - contínuo (devido a normalização)

### Espaço de estados

- Ts: 0 a 100
- Tq: 0 a 100
- Tt: 0 a 100
- h: 0 a 10000
- Fs: 0 a 100
- xf: 0 a 1
- xq: 0 a 1
- iqb: 0 a 1
- Tinf: 10 a 35

### Variáveis fixas

- Fd: 0
- Td: 25
- Tf: 25
- custo_gas: 3
- custo_agua: 4

### Temperatura ambiente e tarifa da energia elétrica

- Foram definidos concepts de acordo com a temperatura do dia (fria, amena, quente) e a tarifa da energia elétrica (dia, noite).
- Tinf: 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30
- custo_eletrico_kwh: R$1,00, R$1,25, R$1,50, R$1,75, R$2,00, R$2,25

```bash
if concept == "banho_dia_frio":
    Tinf_list = [15, 16, 17, 18, 19]
    custo_eletrico_kwh_list = [1, 1.25, 1.5]

if concept == "banho_noite_fria":
    Tinf_list = [15, 16, 17, 18, 19]
    custo_eletrico_kwh_list = [1.75, 2, 2.25]

if concept == "banho_dia_ameno":
    Tinf_list = [20, 21, 22, 23, 24]
    custo_eletrico_kwh_list = [1, 1.25, 1.5]

if concept == "banho_noite_amena":
    Tinf_list = [20, 21, 22, 23, 24]
    custo_eletrico_kwh_list = [1.75, 2, 2.25]

if concept == "banho_dia_quente":
    Tinf_list = [25, 26, 27, 28, 29, 30]
    custo_eletrico_kwh_list = [1, 1.25, 1.5]

if concept == "banho_noite_quente":
    Tinf_list = [25, 26, 27, 28, 29, 30]
    custo_eletrico_kwh_list = [1.75, 2, 2.25]

if concept == "seleciona_banho":
    Tinf_list = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    custo_eletrico_kwh_list = [1, 1.25, 1.5, 1.75, 2, 2.25]
```

### Episódios

- Tempo de cada iteração: 2 minutos
- Tempo total de cada episódio: 14 minutos
- 7 ações em cada episódio
- PPO: 100 steps no PPO, totalizando 400000 episódios

### Parâmetros

- PPO: Parâmetros default 

### Recompensa

Definida como:

```bash
recompensa = iqb
```

### Resultados

- Havia rodado o selector com 6 concepts, mas nesse caso não faz sentido pois o custo da energia elétrica não é otimizado. Logo, estou rodando novamente o concept `seleciona_banho_v2` com somente 3 concepts (`banho_dia_frio`, `banho_dia_ameno`, `banho_dia_quente`).
- Talvez eu precise renomear os concepts para algo como `banho_temperatura_fria_energia_barata`.
