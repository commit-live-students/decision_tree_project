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

def decision_classifier_plot(X_train,X_test,y_train,y_test,depth_list):
    acc_score_test=[]
    acc_score_train=[]

    for d in depth_list:
        clf=DecisionTreeClassifier(random_state=9,max_depth=d)
        clf.fit(X_train,y_train)

        y_pred_test=clf.predict(X_test)
        y_pred_train=clf.predict(X_train)

        acc_score_test.append(accuracy_score(y_test,y_pred_test))
        acc_score_train.append(accuracy_score(y_train,y_pred_train))


    plt.plot(depth_list,acc_score_test,label='Test Set')
    plt.plot(depth_list,acc_score_train,label='train Set')

    plt.xlabel('Depths')
    plt.ylabel('Mean Accuracy Score')
    plt.legend()
    plt.show()
