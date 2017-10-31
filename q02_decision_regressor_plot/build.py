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
y_test_frame= []
x_test_frame = []
y_training_frame = []

def decision_regressor_plot (X_train,X_test,y_train,y_test,depths):
    for i in depths:
        dr = DecisionTreeRegressor(max_depth = 5)
        dr.fit(X_train,y_train)
        y_test_pred = dr.predict(X_test)
      #  y_test_pred_list = np.ndarray.tolist(y_test_pred)
        y = mean_squared_error(y_test,y_test_pred)
        y_test_frame.append(y)
        x_test_frame.append(i)
        y_train_pred = dr.predict(X_train)
        y_train_f = mean_squared_error(y_train,y_train_pred)
        y_training_frame.append(y_train_f)
    plt.plot(x_test_frame,y_test_frame)
    plt.plot(x_test_frame,y_training_frame)

    plt.show()
