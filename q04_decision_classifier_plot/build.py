# %load q04_decision_classifier_plot/build.py
# default imports
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
plt.switch_backend('agg')

data = pd.read_csv('./data/loan_prediction.csv')
np.random.seed(9)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)

depth_list = [8, 10, 15, 20, 50, 100, 120, 150, 175, 200]


def decision_classifier_plot(X_train, X_test, y_train, y_test, depths):
    acc_train, acc_test = [], []
    for i in depths:
        model = DecisionTreeClassifier(max_depth=i)
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        acc_train.append(accuracy_score(y_train, y_pred_train))
        acc_test.append(accuracy_score(y_test,y_pred_test))
    plt.figure(figsize=(10,6))
    plt.plot(depths, acc_train, 'b-', label='Train Set')
    plt.plot(depths, acc_test, 'g.-', label='Test Set')
    plt.xlabel('depths')
    plt.ylabel('accuracy score')
    plt.legend()



