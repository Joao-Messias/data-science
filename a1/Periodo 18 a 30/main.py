import pandas as pd

from a1.funções.volumetria import calcular_volumetria_todas_safras
from a1.funções.correlacao import calcular_e_plotar_correlacao, filtrar_colunas_csat
from a1.funções.modelos import criar_targets_binarios, treinar_modelo_random_forest, salvar_importancias

# Carregar a base de dados
file_path = '../../Lista NPS Positivo_V4.xlsx'
df = pd.read_excel(file_path)

# Função para calcular o target
def calcular_target(nota):
    if nota >= 9:
        return 'Promotor'
    elif nota >= 7:
        return 'Neutro'
    else:
        return 'Detrator'

# Criar a coluna target
df['target'] = df['nota'].apply(calcular_target)

# Filtrar apenas os dados do mercado Brasil
df = df.loc[df['mercado'].str.lower() == 'brasil']

# Filtrar grupos 9 e 10
df = df[df['Grupo de Produto'].astype(str).str.contains('9|10', na=False, case=False)]

# Garantir que a coluna de data está no formato datetime
df['data_resposta'] = pd.to_datetime(df['data_resposta'], errors='coerce', dayfirst=True)

# Criar a coluna 'safra' com o ano da resposta
df['safra'] = df['data_resposta'].dt.year

# Filtrar por período de pesquisa "18 a 30 M"
df = df[df['Periodo de Pesquisa'] == '18 a 30 M']

# Calcular volumetria para todas as safras
volumetria_safra = calcular_volumetria_todas_safras(df)

# Exportar volumetria para Excel
volumetria_safra.to_excel('volumetria_por_safra_18a30M.xlsx', sheet_name='Safra', index=True)

# Identificar colunas numéricas
numerical_columns = df.select_dtypes(include=['number']).columns

# Filtrar colunas CSAT válidas
csat_columns = filtrar_colunas_csat(df, numerical_columns)

# Gerar gráficos e arquivos para cada safra
for safra in df['safra'].unique():
    df_safra = df[df['safra'] == safra]
    calcular_e_plotar_correlacao(df_safra, csat_columns, f"safra_{safra}")

# Gerar gráfico e arquivo para todas as safras combinadas
calcular_e_plotar_correlacao(df, csat_columns, "todas_safras")

# Modelos
# Criar os targets binários
df = criar_targets_binarios(df)

# Preparar espaço de características (X)
X = df[csat_columns].fillna(-99999)

# Modelo para Detratores
print("Treinando Modelo para Detratores...")
model_d = treinar_modelo_random_forest(X, df['y_target_d'])
salvar_importancias(model_d, csat_columns, "importancias_detratores")

# Modelo para Neutros
print("Treinando Modelo para Neutros...")
model_n = treinar_modelo_random_forest(X, df['y_target_n'])
salvar_importancias(model_n, csat_columns, "importancias_neutros")