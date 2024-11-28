import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import numpy as np

def calcular_e_plotar_correlacao(df, csat_columns, safra_name):
    """
    Calcula a correlação de Spearman entre 'nota' e as colunas CSAT, gera gráficos e salva os resultados.
    Apenas correlações positivas são consideradas.
    """
    correlations = []
    for col in csat_columns:
        # Calcular a correlação de Spearman entre 'nota' e a coluna CSAT
        valid_data = df[['nota', col]].dropna()  # Remove linhas com NaN
        if valid_data.shape[0] >= 3:  # Garantir que há pelo menos 3 valores válidos
            corr, _ = spearmanr(valid_data['nota'], valid_data[col])  # Calcula a correlação de Spearman
            if corr > 0:  # Considerar apenas correlações positivas
                correlations.append((col, corr))
        else:
            correlations.append((col, np.nan))

    # Criar DataFrame de correlações e ordená-lo
    correlation_df = pd.DataFrame(correlations, columns=['Coluna', 'Correlação'])
    correlation_df = correlation_df.dropna(subset=['Correlação'])  # Remover correlações nulas
    correlation_df = correlation_df.sort_values(by='Correlação', ascending=False)

    if correlation_df.empty:
        print(f"Nenhuma correlação positiva válida encontrada para {safra_name}.")
        return

    # Função para definir as cores com base na força da correlação
    def get_color(corr):
        if abs(corr) > 0.7:
            return 'red'  # Forte
        elif abs(corr) > 0.4:
            return 'orange'  # Média
        else:
            return 'blue'  # Fraca

    # Adicionar coluna de cores ao DataFrame
    correlation_df['Cor'] = correlation_df['Correlação'].apply(get_color)

    # Plotar as correlações
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Correlação', y='Coluna', data=correlation_df, palette=correlation_df['Cor'].tolist())
    plt.title(f'Correlação de Spearman entre "nota" e Variáveis CSAT ({safra_name})')
    plt.xlabel('Correlação de Spearman')
    plt.ylabel('Colunas CSAT')
    plt.tight_layout()
    plt.savefig(f'correlacao_spearman_{safra_name}.png')  # Salvar gráfico como arquivo PNG
    plt.show()

    # Exportar as correlações para Excel
    correlation_df.to_excel(f'correlacoes_spearman_{safra_name}.xlsx', index=False)

def filtrar_colunas_csat(df, numerical_columns):
    """
    Filtra colunas CSAT numéricas e não vazias.
    """
    return [
        col for col in numerical_columns
        if "(csat)" in col.lower() and df[col].notna().sum() > 0
    ]
