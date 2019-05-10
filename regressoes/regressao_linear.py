import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


USAhousing = pd.read_csv('./arq_csv/USA_Housing.csv')

print(USAhousing.columns)
#plt.show(sns.pairplot(USAhousing))
#plt.show(sns.heatmap(USAhousing.corr()))

X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms','Avg. Area Number of Bedrooms', 'Area Population']]
y = USAhousing['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
#print(X_train)
#print(X_train.shape[0])

lm = LinearRegression()
lm.fit(X_train, y_train)
#print(lm.intercept_)
#print(lm.coef_)

coefs = pd.DataFrame(lm.coef_, X.columns, columns=['Coefs'])
print(coefs)

predict = lm.predict(X_test)
print(predict)

plt.show(plt.scatter(y_test, predict))
plt.show(sns.distplot(y_test - predict))

# erro absoluto médio
print('MAE', metrics.mean_absolute_error(y_test, predict))

# média do quadrado do erro
print('MSE', metrics.mean_squared_error(y_test, predict))

# raiz quadrada da média do quadrado do erro
print('RMSE', np.sqrt(metrics.mean_squared_error(y_test, predict)))