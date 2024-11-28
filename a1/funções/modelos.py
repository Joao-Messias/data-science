import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


def criar_targets_binarios(df):
    df['y_target_d'] = (df['target'] == 'Detrator').astype(int)
    df['y_target_n'] = (df['target'] == 'Neutro').astype(int)
    return df


def treinar_modelo_random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Relatório de Classificação:")
    print(classification_report(y_test, y_pred))
    return model


def salvar_importancias(model, feature_names, output_name):
    importances = model.feature_importances_
    feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

    # Salvar em Excel
    feature_importances.to_excel(f'{output_name}.xlsx', index=False)

    # Criar gráfico
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importances['Feature'], feature_importances['Importance'], color='skyblue')
    plt.xlabel('Importância')
    plt.ylabel('Variáveis')
    plt.title(f'Importância das Features - {output_name}')
    plt.tight_layout()
    plt.savefig(f'{output_name}.png')
    plt.show()
