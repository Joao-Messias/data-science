import pandas as pd
import numpy as np
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import re
import os


def criar_targets_binarios(df):
    """
    Cria colunas binárias para os targets Detrator e Neutro.
    """
    df['y_target_d'] = (df['target'] == 'Detrator').astype(int)
    df['y_target_n'] = (df['target'] == 'Neutro').astype(int)
    return df


def treinar_modelo_xgboost(X, y):
    """
    Treina um modelo XGBoost usando os dados fornecidos.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBClassifier(random_state=42, n_estimators=100, eval_metric='logloss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Relatório de Classificação:")
    print(classification_report(y_test, y_pred))
    return model


def clean_feature_names(feature_names):
    """
    Remove caracteres proibidos nos nomes das features.
    """
    cleaned_feature_names = []
    for name in feature_names:
        # Convert to string
        name = str(name)
        # Remove prohibited characters
        name = re.sub(r'[\[\]<]', '', name)
        cleaned_feature_names.append(name)
    return cleaned_feature_names


def salvar_importancias(model, feature_names, output_name):
    """
    Salva as importâncias das features do modelo em Excel e gera um gráfico de barras.
    """
    importances = model.feature_importances_
    feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importances = feature_importances.sort_values(by='Importance', ascending=True)

    # Salvar em Excel
    feature_importances.to_excel(f'{output_name}.xlsx', index=False)

    # Criar gráfico
    plt.figure(figsize=(40, 10))
    plt.barh(feature_importances['Feature'], feature_importances['Importance'], color='skyblue')
    plt.xlabel('Importância')
    plt.ylabel('Variáveis')
    plt.title(f'Importância das Features - {output_name}')
    plt.tight_layout()
    plt.savefig(f'{output_name}.png')
    plt.show()

    # Retornar as importâncias das features
    return feature_importances


def gerar_pdp(model, X, features, output_prefix):
    for feature in features:
        try:
            # Generate a safe, shortened file name
            safe_feature_name = feature[:50].replace(" ", "_").replace(">", "").replace("<", "").replace("/", "").replace("?", "")
            output_file = f'PDP_{output_prefix}_{safe_feature_name}.png'

            # Plot and save PDP
            plt.figure(figsize=(8, 6))
            PartialDependenceDisplay.from_estimator(model, X, [feature], kind='average')
            plt.title(f'PDP - {output_prefix} - {feature}')
            plt.tight_layout()
            plt.savefig(output_file)
            plt.show()
        except Exception as e:
            print(f"Error generating PDP for feature '{feature}': {e}")