# %load q02_decision_regressor_plot/build.py
# default imports
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.switch_backend('agg')

data = pd.read_csv('./data/house_pricing.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)

depth_list = [2, 8, 10, 15, 20, 25, 30, 35, 45, 50, 80]

def decision_regressor_plot(X_train, X_test, y_train, y_test, depths):
    mse_train, mse_test = [], []
    for i in depths:
        model = DecisionTreeRegressor(max_depth=i)
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        mse_train.append(mean_squared_error(y_train, y_pred_train))
        mse_test.append(mean_squared_error(y_test,y_pred_test))
    plt.figure(figsize=(10,6))
    plt.plot(depths, mse_train, 'b-', label='Train Set')
    plt.plot(depths, mse_test, 'g.-', label='Test Set')
    plt.xlabel('depths')
    plt.ylabel('mean square error')
    plt.legend()



