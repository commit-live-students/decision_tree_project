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


# Write your solution here :
def decision_classifier_plot(X_train,X_test,y_train,y_test,depths):
    train_scores=[]
    test_scores=[]
    for depth in depths:
        tree_clf = DecisionTreeClassifier(max_depth=depth)
        tree_clf.fit(X_train,y_train)
        train_scores.append(tree_clf.score(X_train,y_train))
        test_scores.append(tree_clf.score(X_test,y_test))
    return plt.plot(depths,train_scores),plt.plot(depths,test_scores)
decision_classifier_plot(X_train,X_test,y_train,y_test,depth_list)


