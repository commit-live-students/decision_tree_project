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
    l=[]
    m=[]
    for i in depth_list:
        model=DecisionTreeRegressor(random_state=9,max_depth=i)
        c=model.fit(X_train,y_train)
        ypred1=c.predict(X_train)
        l.append(mean_squared_error(y_train,ypred1))
        #d=model.fit(X_test,y_test)
        ypred2=c.predict(X_test)
        m.append(mean_squared_error(y_test,ypred2))
    plt.plot(depth_list,l)
    plt.plot(depth_list,m)
    plt.show()
#decision_regressor_plot(X_train, X_test, y_train, y_test,depth_list)
