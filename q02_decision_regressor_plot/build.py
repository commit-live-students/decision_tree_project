# %load q02_decision_regressor_plot/build.py
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
def decision_regressor_plot(X_train, X_test, y_train, y_test,depth_list):
    test_score=[]
    train_score=[]
# Write your solution here :
    for i in range(len(depth_list)):
        model =DecisionTreeRegressor(max_depth=depth_list[i])
        model.fit(X_train,y_train)
        y_pred=model.predict(X_train)
        y_pred_test=model.predict(X_test)
        train_score.insert(i,mean_squared_error(y_train,y_pred))
        test_score.insert(i,mean_squared_error(y_test,y_pred_test))
    plt.plot(depth_list, train_score, 'b', label='train set')
    plt.plot(depth_list, test_score, 'g', label='test set')
    plt.legend(loc='best')
    plt.xlabel('depth')
    plt.ylabel('mean_square_error')
    plt.show()
