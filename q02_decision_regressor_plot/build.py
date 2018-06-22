# %load q02_decision_regressor_plot/build.py
# default imports
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib
matplotlib.use('agg')
import pylab as plt
import numpy as np

data = pd.read_csv('./data/house_pricing.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)

depth_list = [2, 8, 10, 15, 20, 25, 30, 35, 45, 50, 80]

# Write your solution here :
def decision_regressor_plot(X_train, X_test, y_train, y_test,depths):
    m1=[]
    m2=[]
    for i in range(len(depths)):
        clf=DecisionTreeRegressor(random_state=9,max_depth=depths[i])
        clf.fit(X_train,y_train)
        mse1=mean_squared_error(y_test,clf.predict(X_test))
        clf=DecisionTreeRegressor(random_state=9,max_depth=depths[i])
        clf.fit(X_test,y_test)
        mse2=mean_squared_error(y_train,clf.predict(X_train))
        m1.append(mse1)
        m2.append(mse2)
    plt.plot(depths,m1,depths,m2)
   
    plt.show()


decision_regressor_plot(X_train, X_test, y_train, y_test,depth_list)



