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


def decision_classifier_plot(X_train, X_test, y_train, y_test, depths):
    def accr_cal(depth):
        model = DecisionTreeClassifier(max_depth=depth)
        model.fit(X_train,y_train)

        y_pred_test = model.predict(X_test)
        accr_test = accuracy_score(y_test,y_pred_test)

        y_pred_train = model.predict(X_train)
        accr_train = accuracy_score(y_train, y_pred_train)

        return depth, accr_test, accr_train

    results = map(lambda depth :  accr_cal(depth), depth_list)
    results = pd.DataFrame(results)

    plt.plot(results[:][0],results[:][1])
    plt.plot(results[:][0],results[:][2])
    plt.show()
    return
