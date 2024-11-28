import pandas as pd

def calcular_volumetria(df, group_by_columns):
    """
    Calcula a volumetria baseada em colunas de agrupamento.

    Parâmetros:
    - df (pd.DataFrame): DataFrame contendo os dados.
    - group_by_columns (list): Lista de colunas para agrupar.

    Retorno:
    - pd.DataFrame: DataFrame contendo a volumetria calculada.
    """
    volumetria = df.groupby(group_by_columns).size().unstack(fill_value=0)
    volumetria['Total'] = volumetria.sum(axis=1)
    volumetria['%Promotores'] = (volumetria.get('Promotor', 0) / volumetria['Total']) * 100
    volumetria['%Neutros'] = (volumetria.get('Neutro', 0) / volumetria['Total']) * 100
    volumetria['%Detratores'] = (volumetria.get('Detrator', 0) / volumetria['Total']) * 100
    return volumetria.round(2)

def calcular_volumetria_todas_safras(df):
    """
    Calcula a volumetria para todas as safras e adiciona uma linha total.

    Parâmetros:
    - df (pd.DataFrame): DataFrame contendo os dados.

    Retorno:
    - pd.DataFrame: DataFrame contendo a volumetria com as safras.
    """
    # Calcula a volumetria por safra
    volumetria = calcular_volumetria(df, ['safra', 'target'])

    # Calcula os totais
    volumetria_total = volumetria.sum()
    volumetria_total['%Promotores'] = (volumetria_total['Promotor'] / volumetria_total['Total']) * 100
    volumetria_total['%Neutros'] = (volumetria_total['Neutro'] / volumetria_total['Total']) * 100
    volumetria_total['%Detratores'] = (volumetria_total['Detrator'] / volumetria_total['Total']) * 100

    # Adiciona a linha total
    volumetria.loc['Total'] = volumetria_total

    return volumetria
