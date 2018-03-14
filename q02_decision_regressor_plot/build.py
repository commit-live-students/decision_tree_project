
# default imports
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('data/house_pricing.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)

depth_list = [2, 8, 10, 15, 20, 25, 30, 35, 45, 50, 65, 80]

# Write your solution here :

def decision_regressor_plot(X_train, X_test, y_train, y_test, depths):
    mean_test_scores = []
    mean_train_scores = []
    
    for depth in depths:
        dt_regressor = DecisionTreeRegressor(max_depth=depth)
        dt_regressor.fit(X_train, y_train)
        mse_train = mean_squared_error(y_train, dt_regressor.predict(X_train))
        mse_test = mean_squared_error(y_test, dt_regressor.predict(X_test))
        mean_test_scores.append(mse_test)
        mean_train_scores.append(mse_train)
    
    plt.figure(figsize=(10, 6))
    plt.plot(depths, mean_train_scores, c='b', label='Train set')
    plt.plot(depths, mean_test_scores, c='g', label='Test set')
    plt.legend(loc='upper left')
    plt.xlabel('depths')
    plt.ylabel('mean square error')
    plt.show()


