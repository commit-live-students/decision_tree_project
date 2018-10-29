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

depths= [2, 8, 10, 15, 20, 25, 30, 35, 45, 50, 80]
def decision_regressor_plot(X_train,X_test,y_train,y_test,depths):
    mse_train = []
    mse_test = []
    
    for i in depths:
        dtr = DecisionTreeRegressor(max_depth=i)
        model = dtr.fit(X_train,y_train)
        y_pred1 = model.predict(X_train)
        e = mean_squared_error(y_train,y_pred1)
        mse_train.append(e)
        
        y_pred2 = model.predict(X_test)
        t = mean_squared_error(y_test,y_pred2)
        mse_test.append(t)
    
    plt.plot(depths,mse_train)
    plt.plot(depths,mse_test)
c=decision_regressor_plot(X_train,X_test,y_train,y_test,depths)


