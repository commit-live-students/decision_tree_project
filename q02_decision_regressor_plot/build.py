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
def decision_regressor_plot(X_train, X_test, y_train, y_test, depth_list):
    trainscore_lst = list()
    testscore_lst = list()
    for val in depth_list:
        dtree = DecisionTreeRegressor(random_state=9, max_depth=val)
        dtree.fit(X_train, y_train)
        # train score
        y_pred_train = dtree.predict(X_train)
        train_score = mean_squared_error(y_true=y_train, y_pred=y_pred_train)
        trainscore_lst.append(train_score)
        # test score
        y_pred_test = dtree.predict(X_test)
        test_score = mean_squared_error(y_true=y_test, y_pred=y_pred_test)
        testscore_lst.append(test_score)

    train_mse = plt.plot(depth_list, trainscore_lst, color='b',label='Train_MSE')
    test_mse = plt.plot(depth_list, testscore_lst, color='g',label='Test_MSE')
    plt.xlabel('Tree Depth')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.show()
