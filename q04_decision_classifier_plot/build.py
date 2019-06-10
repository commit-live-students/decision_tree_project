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
    acc_train = []
    acc_test = []

    for d in depths:
        clf = DecisionTreeClassifier(random_state = 9, max_depth = d)
        clf.fit(X_train, y_train)

        acc_train.append(accuracy_score(y_train, clf.predict(X_train)))
        acc_test.append(accuracy_score(y_test, clf.predict(X_test)))

    fig, ax = plt.subplots()
    ax.plot(depth_list, acc_train, label = 'Train Set')
    ax.plot(depth_list, acc_test, label = 'Test Set')
    ax.legend()
    plt.xlabel('Depth')
    plt.ylabel('Accuracy Score')
    plt.show()


