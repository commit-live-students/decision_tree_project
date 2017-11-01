# %load q04_decision_classifier_plot/build.py
# default imports
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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
    l=[]
    m=[]
    for i in depth_list:
        model=DecisionTreeClassifier(random_state=9,max_depth=i)
        c=model.fit(X_train,y_train)
        ypred1=c.predict(X_train)
        l.append(accuracy_score(y_train,ypred1))
        #d=model.fit(X_test,y_test)
        ypred2=c.predict(X_test)
        m.append(accuracy_score(y_test,ypred2))
    plt.plot(depth_list,l)
    plt.plot(depth_list,m)
    plt.show()

#decision_classifier_plot(X_train, X_test, y_train, y_test,depth_list)
