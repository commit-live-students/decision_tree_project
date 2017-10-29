# default imports
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

data = pd.read_csv("./data/loan_prediction.csv")
np.random.seed(9)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)

param_grid = {"max_depth": [8, 10, 15, 20],
              "max_leaf_nodes": [2, 5, 9, 15, 20],
              "max_features": [1, 2, 3, 5]}

# Write your solution here :
def decision_classifier_plot(X_train,X_test,y_train,y_test,depth_list):
    test_score = []
    train_score = []
    for x in depth_list :
        model = DecisionTreeClassifier(max_depth=x)
        model.fit(X_train,y_train)
        y_train_pred = model.predict(X_train)
        train_score.append(accuracy_score(y_train, y_train_pred))
        y_test_pred = model.predict(X_test)
        test_score.append(accuracy_score(y_test, y_test_pred))
#     for x in depth_list :
#         model = DecisionTreeRegressor(max_depth=x)
#         model.fit(X_test,y_test)
#         y_test_pred = model.predict(X_test)
#         test_score.append(mean_squared_error(y_test, y_test_pred))
    #plt.hold(True)
    depth_list = np.array(depth_list)
    train_score = np.array(train_score)
    test_score = np.array(test_score)
    plt.plot(depth_list, train_score, "r", label="Training scores")
    plt.plot(depth_list, test_score, "b", label="Testing score")
