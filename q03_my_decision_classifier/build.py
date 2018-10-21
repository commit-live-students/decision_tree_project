# %load q03_my_decision_classifier/build.py
# default imports
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

data = pd.read_csv('./data/loan_prediction.csv')
np.random.seed(9)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)

param_grid = {'max_depth': [8, 10, 15, 20],
              'max_leaf_nodes': [2, 5, 9, 15, 20],
              'max_features': [1, 2, 3, 5]}


# Write your solution here :

def my_decision_classifier (X_train, X_test, y_train, y_test,param_grid,n_iter_search=10):
    
    dt = DecisionTreeClassifier(random_state=9)
    ran_search = RandomizedSearchCV(estimator = dt, param_distributions=param_grid,n_iter=n_iter_search)
    model = ran_search.fit(X_train,y_train)
    
    y_pred = model.predict(X_test)
    best_params=model.best_params_
    accuracy = accuracy_score(y_test,y_pred)
    
    return accuracy, best_params
# my_decision_classifier (X_train, X_test, y_train, y_test,param_grid,n_iter_search=10)


