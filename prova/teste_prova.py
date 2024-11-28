import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, \
    precision_recall_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# Carrega a base de dados define as colunas como text (texto das avaliações) e target (rótulo de classe).
file_path = '../amazon_cells_labelled.txt'
df = pd.read_csv(file_path, sep='\t', header=None, names=['text', 'target'])

# Calcular o balanceamento das classes (0: avaliações negativas e 1: avaliações positivas) em porcentagem.
class_balance = df['target'].value_counts(normalize=True) * 100

# 2. Pré-processamento dos dados
# A base já tem as classes como 0 (negativo) e 1 (positivo), então não é necessário fazer transformações
# df['target'] = df['target'].apply(lambda x: 1 if x == 'spam' else 0)

# X é o texto das avaliações e y é o rótulo de classe
X = df['text']
y = df['target']
# 3. Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 4. Transformar o texto das avaliações em uma matriz TF-IDF
vectorizer = TfidfVectorizer()
# Fit -> identifica as palavras (atributos) e cada palavrarecebe um indice unico e é feito o calculo do peso.
X_train_tfidf = vectorizer.fit_transform(X_train)
# Transform, converte todos os textos em vetores com base nas palavras e pesos ajustados (treinados anteriormente)
X_test_tfidf = vectorizer.transform(X_test)

# 5. Criar os classificadores: RandomForest e Regressão Logística
model_rf = RandomForestClassifier(random_state=42)
model_lr = LogisticRegression()

# 6. Treinar os classificadores
# É passado as mensagens treinadas, e seus respectivos pesos, e o ytrain é o target que o modelo deve prever. Basicamente ele contém os valores se é 0 ou 1.
# Ele treina, baseado nas caracteristicas de cara mensagem, e verifica se é spam ou não, pra gerar um padrão.
model_rf.fit(X_train_tfidf, y_train)
model_lr.fit(X_train_tfidf, y_train)

# 7. Fazer as previsões
# Prever se é spam ou não, baseado no modelo treinado - usando predict apenas diz se é 0 ou 1 - com base no corte de 50%
y_pred_rf = model_rf.predict(X_test_tfidf)
y_pred_lr = model_lr.predict(X_test_tfidf)

# Probabilidades previstas
y_proba_rf = model_rf.predict_proba(X_test_tfidf)[:, 1]
y_proba_lr = model_lr.predict_proba(X_test_tfidf)[:, 1]

# 8. Avaliar os modelos com as métricas solicitadas
# Acurácia
accuracy_rf = accuracy_score(y_test, y_pred_rf)
accuracy_lr = accuracy_score(y_test, y_pred_lr)

# Precisão
precision_rf = precision_score(y_test, y_pred_rf)
precision_lr = precision_score(y_test, y_pred_lr)

# Recall
recall_rf = recall_score(y_test, y_pred_rf)
recall_lr = recall_score(y_test, y_pred_lr)

# F1-Score
f1_rf = f1_score(y_test, y_pred_rf)
f1_lr = f1_score(y_test, y_pred_lr)

# Área sob a curva ROC
roc_auc_rf = roc_auc_score(y_test, y_proba_rf)
roc_auc_lr = roc_auc_score(y_test, y_proba_lr)
print(roc_auc_lr)
# 9. Imprimir os resultados
print(f'Acurácia Random Forest: {accuracy_rf}')
print(f'Acurácia Regressão Logística: {accuracy_lr}')
print(f'Precisão Random Forest: {precision_rf}')
print(f'Precisão Regressão Logística: {precision_lr}')
print(f'Recall Random Forest: {recall_rf}')
print(f'Recall Regressão Logística: {recall_lr}')
print(f'F1-Score Random Forest: {f1_rf}')
print(f'F1-Score Regressão Logística: {f1_lr}')
print(f'Área sob a curva ROC Random Forest: {roc_auc_rf}')
print(f'Área sob a curva ROC Regressão Logística: {roc_auc_lr}')

# 10. Plotar as curvas ROC para ambos os modelos

# Calcula as taxas de FPR, TPR e os thresholds para Random Forest
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_proba_rf)
# Calcula a distância euclidiana de cada ponto até o canto superior esquerdo (0,1)
distances_rf = np.sqrt((fpr_rf)**2 + (1 - tpr_rf)**2)
# Encontra o índice da menor distância
index_rf = np.argmin(distances_rf)
# Obtém o threshold ótimo correspondente
optimal_threshold_rf = thresholds_rf[index_rf]
optimal_fpr_rf = fpr_rf[index_rf]
optimal_tpr_rf = tpr_rf[index_rf]

# Faz o mesmo para Regressão Logística
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, y_proba_lr)
distances_lr = np.sqrt((fpr_lr)**2 + (1 - tpr_lr)**2)
index_lr = np.argmin(distances_lr)
optimal_threshold_lr = thresholds_lr[index_lr]
optimal_fpr_lr = fpr_lr[index_lr]
optimal_tpr_lr = tpr_lr[index_lr]

# Plota as curvas ROC e destaca os pontos ótimos
plt.figure()
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.2f})', color='darkorange')
plt.scatter(optimal_fpr_rf, optimal_tpr_rf, marker='o', color='red', label=f'Threshold Ótimo RF = {optimal_threshold_rf:.2f}')
plt.plot(fpr_lr, tpr_lr, label=f'Regressão Logística (AUC = {roc_auc_lr:.2f})', color='green')
plt.scatter(optimal_fpr_lr, optimal_tpr_lr, marker='o', color='blue', label=f'Threshold Ótimo LR = {optimal_threshold_lr:.2f}')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC')
plt.legend()
plt.show()

# 11. Plotar as curvas Precision-Recall
precision_rf, recall_rf, thresholds_pr_rf = precision_recall_curve(y_test, y_proba_rf)
precision_lr, recall_lr, thresholds_pr_lr = precision_recall_curve(y_test, y_proba_lr)

plt.figure()
plt.plot(recall_rf, precision_rf, label='Random Forest', color='darkorange')
plt.plot(recall_lr, precision_lr, label='Regressão Logística', color='green')
plt.xlabel('Recall')
plt.ylabel('Precisão')
plt.title('Curva Precision-Recall')
plt.legend()
plt.show()

# Vamos criar um dataframe com as probabilidades previstas e os valores verdadeiros
df_probs_rf = pd.DataFrame({'probabilidade': y_proba_rf, 'verdadeiro': y_test.values})
df_probs_lr = pd.DataFrame({'probabilidade': y_proba_lr, 'verdadeiro': y_test.values})

# Definir os pontos de corte
ponto_corte_negativo_rf = 0.3  # Definir que probabilidades menores que 30% serão classificadas como "não spam"
ponto_corte_positivo_rf = 0.7  # Definir que probabilidades maiores que 70% serão classificadas como "spam"

ponto_corte_negativo_lr = 0.3  # Mesmos valores para comparação
ponto_corte_positivo_lr = 0.7

# Classificar com base nos pontos de corte para Random Forest
df_probs_rf['classificacao'] = 'analise_manual'
df_probs_rf.loc[df_probs_rf['probabilidade'] <= ponto_corte_negativo_rf, 'classificacao'] = 'negativo'
df_probs_rf.loc[df_probs_rf['probabilidade'] > ponto_corte_positivo_rf, 'classificacao'] = 'positivo'

# Classificar com base nos pontos de corte para Regressão Logística
df_probs_lr['classificacao'] = 'analise_manual'
df_probs_lr.loc[df_probs_lr['probabilidade'] <= ponto_corte_negativo_lr, 'classificacao'] = 'negativo'
df_probs_lr.loc[df_probs_lr['probabilidade'] > ponto_corte_positivo_lr, 'classificacao'] = 'positivo'

# Contagem de classificações
contagem_classificacao_rf = df_probs_rf['classificacao'].value_counts()
contagem_classificacao_lr = df_probs_lr['classificacao'].value_counts()

# print("Distribuição para Random Forest:")
# print(contagem_classificacao_rf)

# print("\nDistribuição para Regressão Logística:")
# print(contagem_classificacao_lr)

# Analisar a distribuição de probabilidades na população geral e população positiva para Random Forest
populacao_geral_rf = df_probs_rf['probabilidade']
populacao_positiva_rf = df_probs_rf[df_probs_rf['verdadeiro'] == 1]['probabilidade']

# Analisar a distribuição de probabilidades na população geral e população positiva para Regressão Logística
populacao_geral_lr = df_probs_lr['probabilidade']
populacao_positiva_lr = df_probs_lr[df_probs_lr['verdadeiro'] == 1]['probabilidade']

# Criar histogramas para Random Forest
step = 0.1
bins = np.arange(0, 1.1, step)

total_populacao_geral_rf = len(populacao_geral_rf)
total_populacao_positiva_rf = len(populacao_positiva_rf)

altura_geral_rf = np.histogram(populacao_geral_rf, bins=bins, density=False)[0] / total_populacao_geral_rf * 100
altura_positiva_rf = np.histogram(populacao_positiva_rf, bins=bins, density=False)[
                         0] / total_populacao_positiva_rf * 100

plt.figure(figsize=(12, 7))

# Gráfico de barras da população geral
plt.bar(bins[:-1] * 100, altura_geral_rf, width=step * 100 / 2, label='População Geral', color='cyan',
        edgecolor='black', align='edge')

# Gráfico de barras da população positiva
plt.bar(bins[:-1] * 100 + step * 100 / 2, altura_positiva_rf, width=step * 100 / 2, label='População Positiva',
        color='yellow', edgecolor='black', align='edge')

# Adicionar linhas verticais para os pontos de corte
plt.axvline(x=ponto_corte_negativo_rf * 100, color='green', linestyle='--', linewidth=2,
            label='Ponto de Corte Negativo (30%)')
plt.axvline(x=ponto_corte_positivo_rf * 100, color='purple', linestyle='--', linewidth=2,
            label='Ponto de Corte Positivo (70%)')

# Formatação do gráfico
plt.title('Distribuição de Probabilidades para Ponto de Corte - Random Forest')
plt.xlabel('Probabilidade Positiva (%)')
plt.ylabel('Porcentagem da População')
plt.legend(loc='upper right')
# plt.show()

# Criar histogramas para Regressão Logística
total_populacao_geral_lr = len(populacao_geral_lr)
total_populacao_positiva_lr = len(populacao_positiva_lr)

altura_geral_lr = np.histogram(populacao_geral_lr, bins=bins, density=False)[0] / total_populacao_geral_lr * 100
altura_positiva_lr = np.histogram(populacao_positiva_lr, bins=bins, density=False)[
                         0] / total_populacao_positiva_lr * 100

plt.figure(figsize=(12, 7))

# Gráfico de barras da população geral
plt.bar(bins[:-1] * 100, altura_geral_lr, width=step * 100 / 2, label='População Geral', color='cyan',
        edgecolor='black', align='edge')

# Gráfico de barras da população positiva
plt.bar(bins[:-1] * 100 + step * 100 / 2, altura_positiva_lr, width=step * 100 / 2, label='População Positiva',
        color='yellow', edgecolor='black', align='edge')

# Adicionar linhas verticais para os pontos de corte
plt.axvline(x=ponto_corte_negativo_lr * 100, color='green', linestyle='--', linewidth=2,
            label='Ponto de Corte Negativo (30%)')
plt.axvline(x=ponto_corte_positivo_lr * 100, color='purple', linestyle='--', linewidth=2,
            label='Ponto de Corte Positivo (70%)')

# Formatação do gráfico
plt.title('Distribuição de Probabilidades para Ponto de Corte - Regressão Logística')
plt.xlabel('Probabilidade Positiva (%)')
plt.ylabel('Porcentagem da População')
plt.legend(loc='upper right')
# plt.show()

# Justificativa dos pontos de corte
# print(f"Justificativa dos pontos de corte para Random Forest e Regressão Logística:")
# print(f"1. O ponto de corte negativo foi estabelecido em 30% para garantir que as mensagens abaixo desse valor têm uma baixa probabilidade de serem spam, reduzindo falsos positivos.")
# print(f"2. O ponto de corte positivo foi estabelecido em 70% para assegurar que mensagens com probabilidade maior que 70% são classificadas como spam com alta confiança, minimizando falsos negativos.")
# print(f"3. Mensagens entre 30% e 70% foram designadas para análise manual, pois o modelo não tem certeza suficiente, permitindo uma segunda análise para evitar erros.")
