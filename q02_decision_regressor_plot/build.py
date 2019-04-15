# %load q02_decision_regressor_plot/build.py
# default imports
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('./data/house_pricing.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)

depth_list = [2, 8, 10, 15, 20, 25, 30, 35, 45, 50, 80]
mse_train=[]
mse_test=[]
# Write your solution here :
def decision_regressor_plot(X_train,X_test,y_train,y_test,depths):
    fig=plt.figure()
    for d in depths:
        dtr=DecisionTreeRegressor(max_depth=d)
    
        dtr.fit(X_train,y_train)
        
        predicted=dtr.predict(X_train)
        mean=mean_squared_error(y_train,predicted)
        mse_train.append(mean)
    
        predicted=dtr.predict(X_test)
        mean=mean_squared_error(y_test,predicted)
        mse_test.append(mean)
        
       
    plt.plot(depth_list,mse_train)
    plt.plot(depth_list,mse_test)
    return fig
 
    

