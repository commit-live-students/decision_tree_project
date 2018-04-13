# %load q04_decision_classifier_plot/build.py
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
def get_accuracy_score(X_train, y_train, X_test,y_test, depth):
    model =DecisionTreeClassifier (random_state=9, max_depth=depth)
    model.fit(X=X_train,y=y_train)
    y_pred_test=model.predict(X_test)
    y_pred_train=model.predict(X_train)
    return [depth, accuracy_score(y_test,y_pred_test), accuracy_score(y_train, y_pred_train)]
 
def decision_classifier_plot (X_train, X_test, y_train, y_test,depths):
    
    results = map(lambda depth :  get_accuracy_score(X_train, y_train, X_test,y_test, depth), depth_list)
    results = pd.DataFrame(results)
    plt.plot(results[:][0],results[:][1])
    plt.plot(results[:][0],results[:][2])
    plt.show()
    return;


