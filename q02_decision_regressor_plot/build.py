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
def decision_regressor_plot(X_train, X_test, y_train, y_test, depth_list):

    lst  = []
    for depth in depth_list:
        tree_reg = DecisionTreeRegressor(max_depth=depth, max_features=25,random_state=9)
        tree_reg.fit(X_train, y_train)
        y_pred_test=tree_reg.predict(X_test)
        y_pred_train=tree_reg.predict(X_train)
        lst.append((depth,mean_squared_error(y_train,y_pred_train),mean_squared_error(y_test,y_pred_test)))
    df =  pd.DataFrame(lst)
    plt.plot(df.iloc[:,0],df.iloc[:,1],c='r',label = 'Train')
    plt.plot(df.iloc[:,0],df.iloc[:,2],c='g',label = 'Test')
    plt.legend()
    #plt.show()
