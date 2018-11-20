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

def decision_regressor_plot(X_train, X_test, y_train, y_test ,depths):
    np.random.seed(9)
    test_scores =[]
    train_scores =[]
    etest_scores =[]
    etrain_scores =[]
    for i in depths:
        dtm = DecisionTreeRegressor(max_depth=i )
        dtm.fit(X_train,y_train)
        train_scores.append(dtm.score(X_train,y_train))
        test_scores.append( dtm.score(X_test,y_test))
        y_tpred = dtm.predict(X_train)
        y_pred = dtm.predict(X_test)
        etrain_scores.append(mean_squared_error(y_train, y_tpred))
        etest_scores.append(mean_squared_error(y_test, y_pred))
    plt.plot(depths,train_scores)
    plt.plot(depths,test_scores )
    plt.xlabel('Max Depth')
    plt.ylabel('Mean Square Error')
    plt.show()




