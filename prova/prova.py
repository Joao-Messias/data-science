import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn import metrics

file_path = '../SMSSpamCollection'
df = pd.read_csv(file_path, sep='\t', header=None, names=['target', 'text'])

class_balance = df['target'].value_counts(normalize=True) * 100

df['target'] = df['target'].apply(lambda x: 1 if x == 'spam' else 0)

# X representação do feature space (espaço de características)
X = df['text']
# y representação do target (a verdade - a base é supervisionada)
y= df['target']

# 4. Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Pegando os índices de treimaneto e teste
X_train_indices = X_train.index  # Índices sorteados para treinamentoa
X_test_indices = X_test.index    # Índices sorteados para teste

X_train_text = df['text'].iloc[X_train_indices] # Extraindo o texto correspondente aos índices presentes no treinamento
X_test_text = df['text'].iloc[X_test_indices]   # Extraindo o texto correspondente aos índices presentes no teste

## Transformar o texto das mensagens em uma matriz TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_test_tfidf = vectorizer.transform(X_test_text)

#Criar um classificador RandomForest e Regressão Logística
model_rf = RandomForestClassifier(random_state=42)
model_lr = LogisticRegression()

# Treinar os classificadores
model_rf.fit(X_train_tfidf, y_train)
model_lr.fit(X_train_tfidf, y_train)

# 2. Previsão com ambos os modelos (Random Forest e Regressão Logística)
y_pred_rf = model_rf.predict(X_test_tfidf)
y_pred_lr = model_lr.predict(X_test_tfidf)

y_proba_rf = model_rf.predict_proba(X_test_tfidf)
y_proba_lr = model_lr.predict_proba(X_test_tfidf)

y_proba_rf_1 = y_proba_rf[:,1]
y_proba_lr_1 = y_proba_lr[:,1]

# print(y_proba) - predict_proba
# [0.99 0.01] -> 99% de não ser spam e 1% de ser spam
#  predict: [0, 1] -> 0 é não spam e 1 é spam

# 3. Avaliação dos modelos
# 3.1. Acurácia
accuracy_rf = accuracy_score(y_test, y_pred_rf)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f'Acurácia Random Forest: {accuracy_rf}')
print(f'Acurácia Regressão Logística: {accuracy_lr}')
# 3.2. Precisão
precision_rf = precision_score(y_test, y_pred_rf)
precision_lr = precision_score(y_test, y_pred_lr)
print(f'Precisão Random Forest: {precision_rf}')
print(f'Precisão Regressão Logística: {precision_lr}')
# 3.3. Recall
recall_rf = recall_score(y_test, y_pred_rf)
recall_lr = recall_score(y_test, y_pred_lr)
print(f'Recall Random Forest: {recall_rf}')
print(f'Recall Regressão Logística: {recall_lr}')
# 3.4. F1-Score
# F1-Score = 2 * (Precisão * Recall) / (Precisão + Recall)
f1_rf = 2 * (precision_rf * recall_rf) / (precision_rf + recall_rf)
f1_lr = 2 * (precision_lr * recall_lr) / (precision_lr + recall_lr)
print(f'F1-Score Random Forest: {f1_rf}')
print(f'F1-Score Regressão Logística: {f1_lr}')
# 3.5. Curva ROC (Receiver Operating Characteristic)
# Cálculo da área abaixo da curva ROC, o métdo roc_auc_score leva os seguintes parâmetros
#(i) y_test -> é a verdade anotada na base supervisionada (target) para os dados de teste
#(i) y_proba1 -> é probabilidade do acerto na classe postiva, extraída pelo modelo ao escorar os espaços de característica de teste, ou seja, é a probabilidade de ser spam
area_rf = metrics.roc_auc_score(y_test,y_proba_rf_1)
area_lr = metrics.roc_auc_score(y_test,y_proba_lr_1)
print(f'Área sob a curva ROC Random Forest: {area_rf}')
print(f'Área sob a curva ROC Regressão Logística: {area_lr}')

# fpr -> false positive rate
# tpr -> true positive rate
# thresholds -> limiares de probabilidade
fpr_rf, tpr_rf, thresholds_rf = metrics.roc_curve(y_test, y_proba_rf_1)
print(f'Falso Positivo Random Forest: {fpr_rf}')
print(f'Verdadeiro positivo Random Forest: {tpr_rf}')
print(f'Limiares Random Forest: {thresholds_rf}')

fpr_lr, tpr_lr, thresholds_lr = metrics.roc_curve(y_test, y_proba_lr_1)

# plotar a curva ROC
import matplotlib.pyplot as plt
import numpy as np

# plotar a curva ROC para Random Forest
roc_auc_rf = metrics.auc(fpr_rf, tpr_rf)
plt.figure()

# Plotar a curva ROC
plt.plot(fpr_rf, tpr_rf, color='darkorange', lw=2, label='Random Forest (AUC = %0.2f)' % roc_auc_rf)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

# Marcar o ponto de equilíbrio na curva ROC
optimal_idx = np.argmax(tpr_rf - fpr_rf)  # Ponto onde a TPR é alta e FPR é baixa
optimal_threshold = thresholds_rf[optimal_idx]
plt.scatter(fpr_rf[optimal_idx], tpr_rf[optimal_idx], marker='o', color='red', label='Ponto Ideal (Threshold = %.2f)' % optimal_threshold)

# Limites do gráfico
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

# Rótulos e título
plt.xlabel('Falso Positivo Rate')
plt.ylabel('Verdadeiro Positivo Rate')
plt.title(f'Curva ROC - Random Forest (AUC = {roc_auc_rf:.2f})')

# Adicionar legenda
plt.legend(loc="lower right")

# Mostrar o gráfico
plt.show()

# plotar a curva ROC para Regressão Logística
roc_auc_lr = metrics.auc(fpr_lr, tpr_lr)
plt.figure()

# Plotar a curva ROC
plt.plot(fpr_lr, tpr_lr, color='green', lw=2, label='Regressão Logística (AUC = %0.2f)' % roc_auc_lr)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

# Marcar o ponto de equilíbrio na curva ROC
optimal_idx_lr = np.argmax(tpr_lr - fpr_lr)  # Ponto onde a TPR é alta e FPR é baixa
optimal_threshold_lr = thresholds_lr[optimal_idx_lr]
plt.scatter(fpr_lr[optimal_idx_lr], tpr_lr[optimal_idx_lr], marker='o', color='red', label='Ponto Ideal (Threshold = %.2f)' % optimal_threshold_lr)

# Limites do gráfico
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

# Rótulos e título
plt.xlabel('Falso Positivo Rate')
plt.ylabel('Verdadeiro Positivo Rate')
plt.title(f'Curva ROC - Regressão Logística (AUC = {roc_auc_lr:.2f})')

# Adicionar legenda
plt.legend(loc="lower right")

# Mostrar o gráfico
plt.show()


# 3.6. Curva PR (Precision-Recall)
# Precision-Recall curve para Random Forest
precision_rf, recall_rf, thresholds_rf_pr = metrics.precision_recall_curve(y_test, y_proba_rf_1)

# Precision-Recall curve para Regressão Logística
precision_lr, recall_lr, thresholds_lr_pr = metrics.precision_recall_curve(y_test, y_proba_lr_1)

# Plotar a Curva Precision-Recall para Random Forest
plt.figure()
plt.plot(recall_rf, precision_rf, color='blue', lw=2, label='Random Forest (AUC-PR)')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Curva Precision-Recall - Random Forest')
plt.legend(loc="lower left")
plt.show()

# Plotar a Curva Precision-Recall para Regressão Logística
plt.figure()
plt.plot(recall_lr, precision_lr, color='green', lw=2, label='Regressão Logística (AUC-PR)')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Curva Precision-Recall - Regressão Logística')
plt.legend(loc="lower left")
plt.show()

# Área sob a curva Precision-Recall
auc_pr_rf = metrics.auc(recall_rf, precision_rf)
auc_pr_lr = metrics.auc(recall_lr, precision_lr)

print(f'AUC-PR do Random Forest: {auc_pr_rf}')
print(f'AUC-PR da Regressão Logística: {auc_pr_lr}')

# Justificativa das métricas mais adequadas para avaliar cada modelo:
# Explicação de como as curvas ROC e PR são plotadas:

# Criar um dataframe com as probabilidades previstas e os valores verdadeiros
df_probs = pd.DataFrame({'probabilidade': y_proba_rf_1, 'verdadeiro': y_test.values})

# Definir os pontos de corte
ponto_corte_negativo = 0.2  # Definir que probabilidades menores que 20% serão classificadas como "não spam"
ponto_corte_positivo = 0.7  # Definir que probabilidades maiores que 70% serão classificadas como "spam"

# Classificar com base nos pontos de corte
df_probs['classificacao'] = 'analise_manual'
df_probs.loc[df_probs['probabilidade'] <= ponto_corte_negativo, 'classificacao'] = 'negativo'
df_probs.loc[df_probs['probabilidade'] > ponto_corte_positivo, 'classificacao'] = 'positivo'

# Contagem de classificações
contagem_classificacao = df_probs['classificacao'].value_counts()
print(contagem_classificacao)

# Analisar a distribuição de probabilidades na população geral e população positiva
populacao_geral = df_probs['probabilidade']
populacao_positiva = df_probs[df_probs['verdadeiro'] == 1]['probabilidade']

step = 0.1
bins = np.arange(0, 1.1, step)

total_populacao_geral = len(populacao_geral)
total_populacao_positiva = len(populacao_positiva)

altura_geral = np.histogram(populacao_geral, bins=bins, density=False)[0] / total_populacao_geral * 100
altura_positiva = np.histogram(populacao_positiva, bins=bins, density=False)[0] / total_populacao_positiva * 100

# Criar o gráfico
plt.figure(figsize=(12, 7))

# Gráfico de barras da população geral
bars1 = plt.bar(bins[:-1] * 100, altura_geral,
                width=step * 100 / 2, label='População Geral', color='cyan', edgecolor='black', align='edge')

# Gráfico de barras da população positiva (spam)
bars2 = plt.bar(bins[:-1] * 100 + step * 100 / 2, altura_positiva,
                width=step * 100 / 2, label='População Positiva', color='yellow', edgecolor='black', align='edge')

# Anotação nas barras
for bar in bars1:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, f'{yval:.2f}%', ha='center', va='bottom', fontsize=10)

for bar in bars2:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, f'{yval:.2f}%', ha='center', va='bottom', fontsize=10)

# Adicionar linhas verticais para os pontos de corte
plt.axvline(x=ponto_corte_negativo * 100, color='green', linestyle='--', linewidth=2, label='Ponto de Corte Negativo (20%)')
plt.axvline(x=ponto_corte_positivo * 100, color='purple', linestyle='--', linewidth=2, label='Ponto de Corte Positivo (70%)')

# Formatação do gráfico
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
plt.ylim(0, 100)  # Limitar o eixo Y de 0 a 100%

plt.title('Distribuição de Probabilidades para Ponto de Corte')
plt.xlabel('Probabilidade Positiva (%)')
plt.ylabel('Porcentagem da População')
plt.xticks(np.arange(0, 110, 10))

plt.legend(loc='upper right')
plt.show()

# Justificativa da escolha dos pontos de corte
print(f"Justificativa:")
print(f"O ponto de corte negativo foi estabelecido em 20% para garantir que, abaixo desse valor, as mensagens têm uma probabilidade muito baixa de serem spam, reduzindo falsos positivos.")
print(f"O ponto de corte positivo foi estabelecido em 70% para assegurar que as mensagens acima desse valor são classificadas como spam com alta confiança, minimizando falsos negativos.")
print(f"As mensagens entre 20% e 70% foram designadas para análise manual, pois o modelo não tem certeza suficiente, permitindo que erros sejam evitados.")
