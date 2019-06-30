#!/usr/bin/env python
# coding: utf-8
# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values    #   X here is matrix and Y is a vector
Y = dataset.iloc[:,2].values

# Splitting the dataset into the Training set and Test set
# from sklearn.model_selection import train_test_split 
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)
#no testing set for this problem because its very small dataset and we need a effecient solution

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# ## Fitting The Regression to dataset


# ## Predicting a new result with Regression Model
y_pred = regressor.predict([[val,]])

# ## Visaulising the Polynomial Model
plt.scatter(X, Y,color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Truth or Bluff(Regression Model)')
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

# ## Visaulising the Regression Model(for higher resolution and smoother curve)
X_grid = np.arrange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, Y,color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or Bluff(Regression Model)')
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()






