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
def decision_regressor_plot(X_train,X_test,y_train,y_test,depth_list):
    test_score = []
    train_score = []
    for x in depth_list :
        model = DecisionTreeRegressor(max_depth=x)
        model.fit(X_train,y_train)
        y_train_pred = model.predict(X_train)
        train_score.append(mean_squared_error(y_train, y_train_pred))
        y_test_pred = model.predict(X_test)
        test_score.append(mean_squared_error(y_test, y_test_pred))
#     for x in depth_list :
#         model = DecisionTreeRegressor(max_depth=x)
#         model.fit(X_test,y_test)
#         y_test_pred = model.predict(X_test)
#         test_score.append(mean_squared_error(y_test, y_test_pred))
    #plt.hold(True)
    depth_list = np.array(depth_list)
    train_score = np.array(train_score)
    test_score = np.array(test_score)
    plt.plot(depth_list, train_score, "r", label="Training scores")
    plt.plot(depth_list, test_score, "b", label="Testing score")
    #plt.show()
