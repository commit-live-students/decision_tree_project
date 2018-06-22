# %load q04_decision_classifier_plot/build.py
# default imports
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('agg')
import pylab as plt
import pandas as pd
import numpy as np

data = pd.read_csv('./data/loan_prediction.csv')
np.random.seed(9)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)

depth_list = [8, 10, 15, 20, 50, 100, 120, 150, 175, 200]


# Write your solution here :
def decision_classifier_plot(X_train, X_test, y_train, y_test,depths):
    m1=[]
    m2=[]
    for i in range(len(depths)):
        clf=DecisionTreeClassifier(random_state=9,max_depth=depths[i])
        clf.fit(X_train,y_train)
        mse1=accuracy_score(y_test,clf.predict(X_test))
        clf=DecisionTreeClassifier(random_state=9,max_depth=depths[i])
        clf.fit(X_test,y_test)
        mse2=accuracy_score(y_train,clf.predict(X_train))
        m1.append(mse1)
        m2.append(mse2)
    plt.plot(depths,m1,depths,m2)
    plt.show()

decision_classifier_plot(X_train, X_test, y_train, y_test,depth_list)

