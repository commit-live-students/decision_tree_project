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

# Write your solution here :
def decision_regressor_plot(X_train, X_test, y_train, y_test, depth_list):
    
    train_error = []
    test_error = []
    for d in depth_list:
        reg_tree = DecisionTreeRegressor(random_state=9, max_depth=d)
        reg_tree.fit(X_train,y_train)
        y_pred = reg_tree.predict(X_test)
        test_error.append( mean_squared_error(y_test, y_pred))
        y_pred1 = reg_tree.predict(X_train)
        train_error.append(mean_squared_error(y_train, y_pred1))
    
    plt.plot(depth_list, train_error, label = 'train set')
    plt.plot(depth_list, test_error, label = 'test set')
    plt.xlabel('depths')
    plt.ylabel('mean square error')
    plt.legend()
    plt.show()

decision_regressor_plot(X_train, X_test, y_train, y_test, depth_list)




