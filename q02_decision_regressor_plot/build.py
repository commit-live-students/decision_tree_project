# default imports
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("./data/house_pricing.csv")
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)

depth_list = [2, 8, 10, 15, 20, 25, 30, 35, 45, 50, 80]

# Write your solution here :
def r2_score(X_train, X_test, y_train, y_test, depth):
    depths = DecisionTreeRegressor(random_state=9, max_features=25, max_depth=depth)
    depths.fit(X_train, y_train)
    y_pred_test = depths.predict(X_test)
    y_pred_train = depths.predict(X_train)
    #mean_squared_error_test = mean_squared_error(y_test, y_pred_test)
    #mean_squared_error_train = mean_squared_error(y_train, y_pred_train)
    return [depth, mean_squared_error(y_test, y_pred_test), mean_squared_error(y_train, y_pred_train)]

def decision_regressor_plot(X_train, X_test, y_train, y_test, depth):
    # Plot the variation between the depth and mean_squared_error
    # Plot the test_score vs max_depth and train_score vs max_depth
    result = map(lambda depth : r2_score(X_train, X_test, y_train, y_test, depth), depth)
    results = pd.DataFrame(result)
    plt.plot(results[:][0], results[:][1])
    plt.plot(results[:][0], results[:][2])
    plt.show()


print decision_regressor_plot(X_train, X_test, y_train, y_test, depth_list)
