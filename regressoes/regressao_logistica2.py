import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

train = pd.read_csv('./arq_csv/titanic_train.csv')

print(train.head())

plt.figure(figsize=(12,6))
plt.show(sns.boxplot(x='Pclass', y='Age', data=train))

# preencher dados faltantes da idade com a média
def inputar_idade(cols):
    Idade = cols[0]
    Classe = cols[1]

    if pd.isnull(Idade):
        if Classe == 1:
            return 37
        elif Classe == 2:
            return 29
        else:
            return 24
    else:
        return Idade

train['Age'] = train[['Age', 'Pclass']].apply(inputar_idade, axis=1)

plt.figure(figsize=(12,6))
plt.show(sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis'))

# deletar coluna
del train['Cabin']
# train.drop('Cabin', inplace=True)
plt.figure(figsize=(12,6))
plt.show(sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis'))

# apagar dados falsos
train.dropna(inplace=True)
plt.figure(figsize=(12,6))
plt.show(sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis'))

# tranforma string ( masc/fem) em número (0/1)
#print(pd.get_dummies(train['Sex']))
sex = pd.get_dummies(train['Sex'], drop_first=True)
embark = pd.get_dummies(train['Embarked'], drop_first=True)

# contar
print(train['Embarked'].value_counts())

# apagar colunas
train.drop(['Sex', 'PassengerId', 'Name', 'Ticket', 'Embarked'], axis=1, inplace=True)
print(train.head())

# concatenar nas colunas
train = pd.concat([train, sex, embark], axis=1)
print(train.head(20))

# dividindo entre dados de teste e treino

X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived', axis=1), train['Survived'], test_size=0.3)

# criar instancia
logmodel = LogisticRegression()
# calibrar o modelo
logmodel.fit(X_train, y_train)

# utilizar o modelo para fazer predições
predictions = logmodel.predict(X_test)

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))