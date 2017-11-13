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

depths = [2, 8, 10, 15, 20, 25, 30, 35, 45, 50, 80]

# Write your solution here :
def decision_regressor_plot (X_train, X_test, y_train, y_test,depths):
    trl=[]
    tsl=[]
    for i in depths:

        ds=DecisionTreeRegressor(max_depth=i)
        ds.fit(X_train,y_train)
        trainres=ds.predict(X_train)

        msetrain=mean_squared_error(y_train,trainres)
        trl.append(msetrain)
        testres=ds.predict(X_test)
        msetest=mean_squared_error(y_test,testres)
        tsl.append(msetest)

    plt.plot(depths,tsl,c='r')
    plt.plot(depths,trl,c='y')

#decision_regressor_plot (X_train, X_test, y_train, y_test,depths)
