import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('./arq_csv/titanic_train.csv')
print(train.head())
print(train.info())

# verificar dados faltantes
    # aumenta o tamanho do plot
plt.figure(figsize=(12,6))
plt.show(sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis'))

# sobreviventes, segregar por sexo: hue='sex'
sns.set_style('whitegrid')
plt.show(sns.countplot(x='Survived', data=train, hue='Sex', palette='RdBu_r'))

# sobreviventes, segregar por classe: hue='Pclas'
sns.set_style('whitegrid')
plt.show(sns.countplot(x='Survived', data=train, hue='Pclass', palette='rainbow'))

# histograma com idade
plt.show(train['Age'].hist(bins=30, color='darkred', alpha=0.4))

# número de acompanhantes
plt.show(sns.countplot(x='SibSp', data=train))

print(train['Fare'].count())
print(train.count())
