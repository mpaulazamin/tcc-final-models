## Single concept - RL como ferramenta da camada RTO

Modelo com malha de inventário para o nível do tanque e com controle liga-desliga do boiler. Com malha cascata, sem split-range.

![image](https://github.com/mpaulazamin/tcc-final-models/blob/single_concept_camada_rto_iqb_com_split_range/imagens/chuveiro_controle_t4a.jpg)

### Espaço de ações

- SPTs: 30 a 40 - contínuo
- SPTq: 30 a 70 - contínuo
- xs: 0.01 a 0.99 - contínuo
- split_range: 0 a 1 - discreto para PPO e contínuo para SAC

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
- Td: 15, 20, 25 e 30
- Tf: 15, 20, 25 e 30
- Tinf: 15, 20, 25 e 30
- custo_eletrico: 2
- custo_gas: 3
- custo_agua: 4

### Episódios

- Tempo de cada iteração: 2 minutos
- Tempo total de cada episódio: 14 minutos
- 7 ações em cada episódio
- PPO: 100 steps no PPO, totalizando 400000 episódios
- SAC: 1000 steps, totalizando 100000 episódios

### Parâmetros

- PPO: Parâmetros default 
- SAC: Parâmetros default 

### Recompensa

Definida como:

```bash
recompensa = iqb
```