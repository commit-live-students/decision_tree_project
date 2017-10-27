# %load q04_decision_classifier_plot/build.py
# default imports
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv("./data/loan_prediction.csv")
np.random.seed(9)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)

depth_list = [8, 10, 15, 20, 50, 100, 120, 150, 175, 200]


# Write your solution here :
def decision_classifier_plot(X_train, X_test, y_train, y_test,depth_list):
    est = DecisionTreeClassifier(random_state=9)

    rand_search = RandomizedSearchCV(estimator = est, param_distributions={"max_depth":depth_list},n_iter=10)
    rand_search.fit(X_train,y_train)
    y_prediction = rand_search.predict(X_test)

    mse_test = mean_squared_error(y_test,y_prediction)

    fig = plt.figure(figsize=(10, 7))
    plt.plot(depth_list, rand_search.cv_results_['mean_test_score'], label="Train set")
    plt.plot(depth_list, rand_search.cv_results_['mean_train_score'], label="Test Set")

    plt.xlabel('Depth')
    plt.ylabel('mean square  error')
    plt.show()
