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

depth_list = [2, 8, 10, 15, 20, 25, 30, 35, 45, 50, 80]

# Write your solution here :
def decision_regressor_plot(X_train,X_test,y_train,y_test,depths):
    train_results=[]
    test_results=[]
    depth_list = [2, 8, 10, 15, 20, 25, 30, 35, 45, 50, 80]
    for depth_list in depth_list:    
        model=DecisionTreeRegressor(random_state=9,max_depth=depth_list)
        model.fit(X_train,y_train)
        train_pred=model.predict(X_train)
        mse_train=mean_squared_error(y_train,train_pred)
        train_results.append(mse_train)

        test_pred=model.predict(X_test)
        mse_test=mean_squared_error(y_test,test_pred)
        test_results.append(mse_test)
       
    plt.plot([2, 8, 10, 15, 20, 25, 30, 35, 45, 50, 80],train_results,'b-',label='Train set')
    plt.plot([2, 8, 10, 15, 20, 25, 30, 35, 45, 50, 80],test_results,'g-',label='Test set')
    plt.legend(loc='best')
    plt.xlabel('depths')
    plt.ylabel('mean squared error')
    plt.show()
    


