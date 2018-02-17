# default imports
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("./data/house_pricing.csv")
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)

depth_list = [2, 8, 10, 15, 20, 25, 30, 35, 45, 50, 80]

# Write your solution here :
def get_r2_score(X_train, y_train, X_test,y_test, depth):
    model =DecisionTreeRegressor (random_state=9, max_depth=depth, max_features=25)
    model.fit(X=X_train,y=y_train)
    y_pred_test=model.predict(X_test)
    y_pred_train=model.predict(X_train)
    return [depth, mean_squared_error(y_test,y_pred_test), mean_squared_error(y_train, y_pred_train)]

def decision_regressor_plot (X_train, X_test, y_train, y_test,depths):
    #Plots the variation between depth and mean square error.
    #Plots test_scores vs max_depth and train_scores vs max_depth (in the same plot).
    results = map(lambda depth :  get_r2_score(X_train, y_train, X_test,y_test, depth), depth_list)
    results = pd.DataFrame(results)
    plt.plot(results[:][0],results[:][1])
    plt.plot(results[:][0],results[:][2])
    plt.show()
    return;

decision_regressor_plot (X_train, X_test, y_train, y_test,depth_list)
