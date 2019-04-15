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
def decision_classifier_plot(X_train,X_test,y_train,y_test,depths):
        accuracy_fit=[]
        accuracy_predict=[]
        for a in depths:
            dtr=DecisionTreeClassifier(max_depth=a)
            dtr.fit(X_train,y_train)
            predicted=dtr.predict(X_train)
            accuracy_fit.append(accuracy_score(y_train,predicted))            
            
            predicted=dtr.predict(X_test)
            accuracy_predict.append(accuracy_score(y_test,predicted))
        plt.plot(depths,accuracy_fit)
        plt.plot(depths,accuracy_predict)
        plt.show()
            





