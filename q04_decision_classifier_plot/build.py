# %load q04_decision_classifier_plot/build.py
# default imports
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv('./data/loan_prediction.csv')
np.random.seed(9)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)

depth_list = [8, 10, 15, 20, 50, 100, 120, 150, 175, 200]


# Write your solution here :
# Write your solution here :
def decision_classifier_plot(X_train, X_test, y_train, y_test,depth):
    train_as_list=[]
    test_as_list=[]
    for i in depth_list:
        dt=DecisionTreeClassifier(max_depth=i,random_state=9)
        dt=dt.fit(X_train,y_train)
        y_train_pred= dt.predict(X_train)
        train_as_list.append(accuracy_score(y_train,y_train_pred))

        y_test_pred = dt.predict(X_test)
        test_as_list.append(accuracy_score(y_test,y_test_pred))

    plt.figure(figsize=(10,6))
    plt.plot(depth_list,train_as_list, c='r',label='Train set')
    plt.xlabel('Depth')
    plt.plot(depth_list,test_as_list,c='g',label='Test set')
    plt.ylabel('Accuracy Score')
    plt.legend()
    plt.show()

