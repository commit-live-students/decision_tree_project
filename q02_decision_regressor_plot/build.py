# default imports
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.pyplot.switch_backend('agg')
import numpy as np

data = pd.read_csv("./data/house_pricing.csv")
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)

depth_list = [2, 8, 10, 15, 20, 25, 30, 35, 45, 50, 80]

# Write your solution here :
def decision_regressor_plot(X_train, X_test, y_train, y_test,depth_list):
    dlist = []
    mse_error1 = []
    mse_error2 = []
    for d in depth_list:
        dt = DecisionTreeRegressor(max_depth=d)
        dt.fit(X_train,y_train)
        ypred = dt.predict(X_test)
        ypred2 = dt.predict(X_train)
        mse1 = mean_squared_error(ypred,y_test)
        mse2 = mean_squared_error(y_train,ypred2)
        dlist.append(d)
        mse_error1.append(mse1)
        mse_error2.append(mse2)

    return dlist,mse_error1,mse_error2

dlist,mse_error1,mse_error2 = decision_regressor_plot(X_train, X_test, y_train, y_test,depth_list)

#plt.figure(figsize=(10,10))
#plt.plot(dlist,mse_error1,label='Test Set')
#plt.plot(dlist,mse_error2,label='Train Set')
#plt.legend()
#plt.show()
