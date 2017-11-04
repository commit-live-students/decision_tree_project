# %load q01_my_decision_regressor/build.py
# default imports
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv("./data/house_pricing.csv")
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)

param_grid = {"max_depth": [2, 3, 5, 6, 8, 10, 15, 20, 30, 50],
              "max_leaf_nodes": [2, 3, 4, 5, 10, 15, 20],
              "max_features": [4, 8, 20, 25]}
def my_decision_regressor(X_train, X_test, y_train, y_test,param_grid):
    model=DecisionTreeRegressor(random_state=9)
# Write your solution here :
    tree=GridSearchCV(model,param_grid,cv=5)
    tree.fit(X_train,y_train)
    y_pred=tree.predict(X_test)
    r=r2_score(y_test,y_pred)
    return r,tree.best_params_
