import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the data
dataset = pd.read_csv('Salary.csv');
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/5, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

# shows training datasets graph
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.show()

# shows data test set
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.show()
