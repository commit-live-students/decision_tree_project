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
def decision_regressor_plot(X_train,X_test,y_train,y_test,depths):

    mses_train = []
    mses_test = []
    for i in depths:
        clf = DecisionTreeRegressor(random_state=9,max_depth=i)
        clf.fit(X_train,y_train)
        y_pred1 = clf.predict(X_test)
        y_pred2 = clf.predict(X_train)
        mse1 = mean_squared_error(y_test,y_pred1)
        mse2 = mean_squared_error(y_train,y_pred2)
        mses_train.append(mse2)
        mses_test.append(mse1)
        plt.plot(depths,mses_test)
        plt.plot(depths,mses_train)
        plt.legend(['Test set', 'Train set'], loc='upper left')
        plt.show()
