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
def decision_classifier_plot(X_train,X_test,y_train,y_test,depths):
    accur1 = []
    accur2 = []
    for i in depths:
        clf = DecisionTreeClassifier(random_state=9,max_depth=i)
        clf.fit(X_train,y_train)
        y_pred1 = clf.predict(X_test)
        y_pred2 = clf.predict(X_train)
        acc1 = accuracy_score(y_test,y_pred1)
        acc2 = accuracy_score(y_train,y_pred2)
        accur1.append(acc1)
        accur2.append(acc2)
    plt.plot(depths,accur1)
    plt.plot(depths,accur2)
    plt.legend('Test set','Train set')
    plt.show()
