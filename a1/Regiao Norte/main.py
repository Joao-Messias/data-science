import pandas as pd

from a1.funções.volumetria import calcular_volumetria_todas_safras
from a1.funções.correlacao import calcular_e_plotar_correlacao, filtrar_colunas_csat
from a1.funções.modelos import criar_targets_binarios, treinar_modelo_random_forest, salvar_importancias

# Carregar a base de dados
file_path = '../../Lista NPS Positivo_V4.xlsx'
df = pd.read_excel(file_path)

def calcular_target(nota):
    if nota >= 9:
        return 'Promotor'
    elif nota >= 7:
        return 'Neutro'
    else:
        return 'Detrator'

df['target'] = df['nota'].apply(calcular_target)

df = df.loc[df['mercado'].str.lower() == 'brasil']

df = df[df['Grupo de Produto'].astype(str).str.contains('9|10', na=False, case=False)]

df['data_resposta'] = pd.to_datetime(df['data_resposta'], errors='coerce', dayfirst=True)

df['safra'] = df['data_resposta'].dt.year

def definir_regiao(estado):
    regioes = {
        'Sul': ['PR', 'SC', 'RS'],
        'Sudeste': ['SP', 'RJ', 'MG', 'ES'],
        'Centro-Oeste': ['GO', 'MT', 'MS', 'DF'],
        'Nordeste': ['BA', 'PE', 'CE', 'RN', 'PB', 'AL', 'SE', 'PI', 'MA'],
        'Norte': ['PA', 'AM', 'RO', 'AC', 'RR', 'AP', 'TO']
    }
    for regiao, estados in regioes.items():
        if estado in estados:
            return regiao
    return 'Outros'

df['região'] = df['estado'].apply(definir_regiao)

df = df[df['região'] == 'Nordeste']

# Calcular volumetria para todas as safras
volumetria_safra = calcular_volumetria_todas_safras(df)
volumetria_safra.to_excel('volumetria_por_safra_norte.xlsx', sheet_name='Safra', index=True)

# Identificar colunas numéricas e CSAT válidas
numerical_columns = df.select_dtypes(include=['number']).columns
csat_columns = filtrar_colunas_csat(df, numerical_columns)

# Gerar gráficos de correlação de spearman e arquivos para cada safra
for safra in df['safra'].unique():
    df_safra = df[df['safra'] == safra]
    calcular_e_plotar_correlacao(df_safra, csat_columns, f"safra_{safra}")
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

print("\nValidação do y_target_d para cada target original:")
print(df.groupby(['target', 'y_target_d']).size())
