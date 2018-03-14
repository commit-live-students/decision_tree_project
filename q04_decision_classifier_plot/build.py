
# default imports
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv('./data/loan_prediction.csv')
np.random.seed(9)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)

depth_list = [8, 10, 15, 20, 50, 100, 120, 150, 175, 200]

# Write your solution here :

def decision_classifier_plot(X_train, X_test, y_train, y_test, depths):
    mean_test_scores = []
    mean_train_scores = []
    
    for depth in depths:
        dt_classifier = DecisionTreeClassifier(max_depth=depth)
        dt_classifier.fit(X_train, y_train)
        acc_train = accuracy_score(y_train, dt_classifier.predict(X_train))
        acc_test = accuracy_score(y_test, dt_classifier.predict(X_test))
        mean_test_scores.append(acc_test)
        mean_train_scores.append(acc_train)
    
    plt.figure(figsize=(10, 6))
    plt.plot(depths, mean_train_scores, c='b', label='Train set')
    plt.plot(depths, mean_test_scores, c='g', label='Test set')
    plt.legend(loc='upper left')
    plt.xlabel('depths')
    plt.ylabel('mean square error')
    plt.show()

