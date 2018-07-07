# %load q02_decision_regressor_plot/build.py
# default imports
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('./data/house_pricing.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)

depth_list = [2, 8, 10, 15, 20, 25, 30, 35, 45, 50, 80]

def decision_regressor_plot(X_train, X_test, y_train, y_test, depths):
    train_scores = []
    test_scores = []
    for depth in depths:
        model = DecisionTreeRegressor(criterion='mse', max_depth=depth, random_state=9)
        model = model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        mse_train = mean_squared_error(y_pred_train, y_train)
        train_scores.append(mse_train)
        y_pred_test = model.predict(X_test)
        mse_test = mean_squared_error(y_pred_test, y_test)
        test_scores.append(mse_test)
    plt.plot(depth_list, train_scores)
    plt.plot(depth_list, test_scores)
    plt.show()
        

