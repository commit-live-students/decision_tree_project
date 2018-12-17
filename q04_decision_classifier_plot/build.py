# %load q04_decision_classifier_plot/build.py
# default imports
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
plt.switch_backend('agg')

data = pd.read_csv('./data/loan_prediction.csv')
np.random.seed(9)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)

depth_list = [8, 10, 15, 20, 50, 100, 120, 150, 175, 200]


# Write your solution here :
def decision_classifier_plot(X_train,X_test,y_train,y_test,depth_list):
    error_train = []
    error_test = []
    for i in range(len(depth_list)):
        dt_classifier = DecisionTreeClassifier(max_depth=depth_list[i],random_state=9)
        dt_classifier.fit(X_train,y_train)
        predict_train = dt_classifier.predict(X_train)
        predict_test = dt_classifier.predict(X_test)

        error_train.append(accuracy_score(y_train,predict_train))

        error_test.append(accuracy_score(y_test,predict_test))


    plt.plot(depth_list,error_test)
    plt.plot(depth_list,error_train)
    plt.show()


