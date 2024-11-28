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

# Filtro por região
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

# Criar a nova coluna 'região'
df['região'] = df['estado'].apply(definir_regiao)

# Filtrar pela região 'Sudeste'
df = df[df['região'] == 'Sul']
# df = df[df['Periodo de Pesquisa'] == '6 a 12 M']

# Calcular volumetria para todas as safras
volumetria_safra = calcular_volumetria_todas_safras(df)

# Exportar volumetria para Excel
volumetria_safra.to_excel('volumetria_por_safra_sudeste.xlsx', sheet_name='Safra', index=True)

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

print("Modelos treinados e resultados salvos.")
