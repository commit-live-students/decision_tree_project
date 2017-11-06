
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

# Write your solution here :
def decision_regressor_plot(X_train, X_test, y_train, y_test,depth_list):
    #regressor = DecisionTreeRegressor()
    train_error_arr=[]
    test_error_arr=[]
    for i in depth_list:
        #print(i)
        regressor = DecisionTreeRegressor(max_depth=i,random_state=9)
        regressor.fit(X_train,y_train)
        y_train_predict=regressor.predict(X_train)
        y_test_predict=regressor.predict(X_test)
        train_error=(y_train,y_train_predict)
        train_Score=regressor.score(X_train,y_train)
        test_Score=regressor.score(X_test,y_test)#
        #test_error=(y_test,y_test)
        #print(np.mean(test_error))
        #print(np.test_error)
        train_error_arr.append(train_Score)
        test_error_arr.append(test_Score)
    print(test_error_arr)
    plt.plot(depth_list,test_error_arr, c='r', label='Test set')

    plt.plot(depth_list,train_error_arr, c='r', label='Train set')
#     plt.xticks(x,param_values)
#     plt.plot(x,mean_test_scores,c='g', label='Test set')
#     plt.xlabel(grid_obj.param_grid.keys()[0])
#     plt.ylabel('mean scores')
#     plt.legend()
    plt.show()

    #print(train_error_arr)

#decision_regressor_plot(X_train, X_test, y_train, y_test,depth_list)
