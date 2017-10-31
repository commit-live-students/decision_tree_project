# default imports
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv("./data/loan_prediction.csv")
np.random.seed(9)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)

depth_list = [8, 10, 15, 20, 50, 100, 120, 150, 175, 200]


# Write your solution here :
y_test_frame= []
x_test_frame = []
y_training_frame = []

def decision_classifier_plot (X_train,X_test,y_train,y_test,depths):
    for i in depths:
        dr = DecisionTreeClassifier(max_depth = 5)
        dr.fit(X_train,y_train)
        y_test_pred = dr.predict(X_test)
      #  y_test_pred_list = np.ndarray.tolist(y_test_pred)
        y = accuracy_score(y_test,y_test_pred)
        y_test_frame.append(y)
        x_test_frame.append(i)
        y_train_pred = dr.predict(X_train)
        y_train_f = accuracy_score(y_train,y_train_pred)
        y_training_frame.append(y_train_f)
    plt.plot(x_test_frame,y_test_frame)
    plt.plot(x_test_frame,y_training_frame)

    plt.show()
