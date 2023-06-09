# Modelos de RL treinados com os algoritmos PPO e SAC

## Instalação

Siga estas instruções: https://github.com/ray-project/ray/tree/master/rllib#installation-and-setup.

```bash
conda create -n rllib python=3.8
conda activate rllib
pip install "ray[rllib]" tensorflow 
```

Depois, instale o pacote `tensorflow_probability`: https://github.com/tensorflow/probability/releases.

```bash
pip install tensorflow-probability==0.19.0
```

## Integração com Tensorboard

Siga estas instruções: https://stackoverflow.com/questions/45095820/tensorboard-command-not-found.

Execute o seguinte comando:

```bash
pip show tensorflow
```

Entre no local onde o `tensorflow` está instalado:

```bash
cd C:\users\maria\appdata\roaming\python\python38\site-packages
```

Entre no folder do `tensorboard`:

```bash
cd tensorboard
```

Execute o seguinte comando:

```bash
python main.py --logdir "C:\users\maria\ray_results\folder_experiment"
```

## Sanity check

Treinando o sistema com o ambiente customizado [neste script](https://github.com/mpaulazamin/tcc-models-rllib/blob/main/sanity_check.py), obtém-se o resultado abaixo, 
que reproduz o resultado encontrado no [notebook](https://github.com/mpaulazamin/tcc-models-rllib/blob/main/chuveiro_turbinado.ipynb) do professor.

![check](https://github.com/mpaulazamin/tcc-models-rllib/blob/main/imagens/custom_env.jpg)
