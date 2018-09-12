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
def decision_regressor_plot(X_train,X_test,y_train,y_test,depth_list):
    mean_test_score = []
    mean_train_score = []
    fig = plt.figure()
    for i in depth_list:
        dt_model = DecisionTreeRegressor(max_depth=i)
        dt_model.fit(X_train,y_train)
 
        y_pred = dt_model.predict(X_test)
        mean_test = mean_squared_error(y_test,y_pred)
        mean_test_score.append(mean_test)
        
        y_pred = dt_model.predict(X_train)
        mean_train = mean_squared_error(y_train,y_pred)
        mean_train_score.append(mean_train)
    plt.plot(depth_list,mean_train_score)   
    plt.plot(depth_list,mean_test_score)
    return fig        


