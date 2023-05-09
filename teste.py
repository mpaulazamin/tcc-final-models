import ray
import ray.rllib.algorithms.ppo as ppo
import ray.rllib.algorithms.sac as sac
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.policy.policy import Policy

import random
import os
import glob
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import accumulate

from controle_temperatura_saida import simulacao_malha_temperatura
from controle_temperatura_saida import modelagem_sistema
from controle_temperatura_saida import modelo_valvula_saida
from controle_temperatura_saida import calculo_iqb
from controle_temperatura_saida import custo_eletrico_banho
from controle_temperatura_saida import custo_gas_banho
from controle_temperatura_saida import custo_agua_banho


print(Policy.from_checkpoint(glob.glob("C:\\Users\\maria\\OneDrive\\TCC\\RLLib_códigos\\tcc-final-models\\tcc-final-models\\models\\results_proximal_policy_optimization_concept_banho_dia_frio"+"/*")[-1]).get("default_policy"))
print(Policy.from_checkpoint(glob.glob("C:\\Users\\maria\\OneDrive\\TCC\\RLLib_códigos\\tcc-final-models\\tcc-final-models\\models\\results_proximal_policy_optimization_concept_banho_noite_fria"+"/*")[-1]).get("default_policy"))
print(Policy.from_checkpoint(glob.glob("C:\\Users\\maria\\OneDrive\\TCC\\RLLib_códigos\\tcc-final-models\\tcc-final-models\\models\\results_proximal_policy_optimization_concept_banho_dia_ameno"+"/*")[-1]).get("default_policy"))
print(Policy.from_checkpoint(glob.glob("C:\\Users\\maria\\OneDrive\\TCC\\RLLib_códigos\\tcc-final-models\\tcc-final-models\\models\\results_proximal_policy_optimization_concept_banho_noite_amena"+"/*")[-1]).get("default_policy"))
print(Policy.from_checkpoint(glob.glob("C:\\Users\\maria\\OneDrive\\TCC\\RLLib_códigos\\tcc-final-models\\tcc-final-models\\models\\results_proximal_policy_optimization_concept_banho_dia_quente"+"/*")[-1]).get("default_policy"))
print(Policy.from_checkpoint(glob.glob("C:\\Users\\maria\\OneDrive\\TCC\\RLLib_códigos\\tcc-final-models\\tcc-final-models\\models\\results_proximal_policy_optimization_concept_banho_noite_quente"+"/*")[-1]).get("default_policy"))

print(glob.glob("C:\\Users\\maria\\OneDrive\\TCC\\RLLib_códigos\\tcc-final-models\\tcc-final-models\\models\\results_proximal_policy_optimization_concept_seleciona_banho" +"/*")[-1])

print(Algorithm.from_checkpoint("C:\\Users\\maria\\OneDrive\\TCC\\RLLib_códigos\\tcc-final-models\\tcc-final-models\\models\\results_proximal_policy_optimization_concept_seleciona_banho\\checkpoint_000100"))




