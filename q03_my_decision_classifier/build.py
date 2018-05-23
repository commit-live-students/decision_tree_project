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

def my_decision_classifier(X_train1,X_test1,y_train1,y_test1,params,n=10):
    X_train1,X_test1,y_train1,y_test1 = X_train, X_test, y_train, y_test
    params = param_grid    
    dt = DecisionTreeClassifier(random_state=9)
    dt_rnd = RandomizedSearchCV(dt,param_distributions=params,n_iter=n)
    dt_rnd.fit(X_train1,y_train1)
    y_pred=dt_rnd.predict(X_test1)
    score=accuracy_score(y_test1,y_pred)
    return score,dt_rnd.best_params_

my_decision_classifier(X_train,X_test,y_train,y_test,param_grid,n=10)


