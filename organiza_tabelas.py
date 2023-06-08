import glob
import pandas as pd

def organiza_tabelas(path):

    csv_files = glob.glob(path + "/*.csv")
    df_list = (pd.read_csv(file) for file in csv_files)
    df = pd.concat(df_list, ignore_index=True)
    df.to_csv(path + "/tabela_final.csv", index=False)

path1 = "./resultados_tabela_selector"
organiza_tabelas(path1)

path2 = "./resultados_concepts_selector"
organiza_tabelas(path2)