# default imports
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

data = pd.read_csv('./data/house_pricing.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)

parameters = {'max_depth': [2, 3, 5, 6, 8, 10, 15, 20, 30, 50],
              'max_leaf_nodes': range(2,21),
              'max_features':range(2,26)
             }

def my_decision_regressor(X_train1,X_test1,y_train1,y_test1,params):
    X_train1,X_test1,y_train1,y_test1 = X_train, X_test, y_train, y_test
    params = parameters 
    dt = DecisionTreeRegressor(random_state=9)
    dt_grid = GridSearchCV(dt,params, cv=5)
    dt_grid.fit(X_train1,y_train1)
    y_pred=dt_grid.predict(X_test1)
    r2score=r2_score(y_test1,y_pred)
    return r2score,dt_grid.best_params_



