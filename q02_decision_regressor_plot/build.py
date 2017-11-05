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

def decision_regressor_plot(X_train,X_test,y_train,y_test,depths):
    a=[]
    b=[]
    for i in depths:
        dsc = DecisionTreeRegressor(random_state=9,max_depth=i)
        dsc.fit(X_train,y_train)
        y_pred_test = dsc.predict(X_test)
        y_pred_train = dsc.predict(X_train)
        err1 = mean_squared_error(y_test,y_pred_test)
        err2 = mean_squared_error(y_train,y_pred_train)
        a.append(err1)
        b.append(err2)
    plt.plot(depth_list,a)
    plt.plot(depth_list,b)
    plt.show()
