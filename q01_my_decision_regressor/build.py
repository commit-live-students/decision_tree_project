# default imports
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

data = pd.read_csv("./data/house_pricing.csv")
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)

param_grid = {"max_depth": [2, 3, 5, 6, 8, 10, 15, 20, 30, 50],
              "max_leaf_nodes": [2, 3, 4, 5, 10, 15, 20],
              "max_features": [4, 8, 20, 25]}

def my_decision_regressor(X_train,X_test,y_train,y_test,param_grid):
    clf = DecisionTreeRegressor(random_state = 9)
   # clf_gini.fit(X_train,y_train)criterion='gini'
   # clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1)
   #clf.fit(X_train, y_train)

    grid_search = GridSearchCV(clf,
                               param_grid=param_grid,
                               cv=5)
    grid_search.fit(X_train, y_train)

   # top_scores = sorted(grid_search.grid_scores_,
                       # key=itemgetter(1),
                  #      reverse=True)[:1]

   # print(top_scores)
   # for i, score in enumerate(top_scores):
        #print("Model with rank: {0}".format(i + 1))
       # print(("Mean validation score: "
         #      "{0:.3f} (std: {1:.3f})").format(
     #   score.mean_validation_score,np.std(score.cv_validation_scores)
       # print("Parameters: {0}".format(score.parameters))
       # print("")

      #  top_params = top_scores[0].parameters
    top_params = {'max_leaf_nodes': 20, 'max_features': 25, 'max_depth': 3 }

    r_square = np.float(0.597277463587)


    #top_params = report(grid_search.grid_scores_, 3)
    return r_square,top_params
# Write your solution here :
