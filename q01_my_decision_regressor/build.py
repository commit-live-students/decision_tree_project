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

# Write your solution here :
def my_decision_regressor(X_train, X_test, y_train, y_test, param_grid):
    dtree_regressor = DecisionTreeRegressor(random_state=9)
    dtree_gridcv = GridSearchCV(estimator=dtree_regressor, param_grid=param_grid, scoring='r2', n_jobs=4, cv=5)
    dtree_gridcv.fit(X_train, y_train)
    # Returns r2_score (since we have passed scoring as 'r2') to GridSearchCV
    rsquare_score = dtree_gridcv.score(X_test, y_test)
    # Returns the best params from the passed param grid to GridSearchCV
    best_params = dtree_gridcv.best_params_
    return rsquare_score, best_params
