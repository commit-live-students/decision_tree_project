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
def decision_classifier_plot(X_train,X_test,y_train,y_test,depths = depth_list):
    clf = DecisionTreeClassifier(random_state=9)
    cv = RandomizedSearchCV(clf,param_distribution=depth_list,n_iter=10)
    cv.fit(X_train,y_train)
    y_pred = cv.predict(X_test)
    score = accuracy_score(y_test,y_pred).mean()
    score_train = accuracy_score(y_train,y_pred)
    plt.plot(depth_list,score)
    plt.plot(score,depth_list)
    plt.plot(score_train,depth_list)
    plt.show()


