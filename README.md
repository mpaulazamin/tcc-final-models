## Single concept - RL como ferramenta da camada RTO com diversos cenários

Modelo com malha de inventário para o nível do tanque e com controle liga-desliga do boiler. Com malha cascata, sem split-range.

![image](https://github.com/mpaulazamin/tcc-final-models/blob/single_concept_camada_rto_sem_split_range_multiple_scenarios/imagens_iniciais/chuveiro_controle_t4a_sem_split.jpg)

### Espaço de ações

- SPTs: 30 a 40 - contínuo
- SPTq: 30 a 70 - contínuo
- xs: 0.01 a 0.99 - contínuo
- Sr: 0 a 1 - contínuo

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
- custo_eletrico_kwh: 0 a 3
- custo_eletrico: 0 a 1 
- custo_gas: 0 a 1 
- custo_agua: 0 a 1

### Variáveis fixas

- Fd: 0
- Td: 25
- Tf: 25
- custo_gas: 3
- custo_agua: 4

### Temperatura ambiente e tarifa da energia elétrica

- Foram selecionados aleatoriamente no reset de cada episódio.
- Tinf: 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30
- custo_eletrico_kwh: R$1,00, R$1,25, R$1,50, R$1,75, R$2,00, R$2,25

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
recompensa = 5 * iqb - 2 * custo_eletrico - custo_gas - custo_agua
```

### Resultados

- A pasta models_v2 contém o agente treinado com 150 steps