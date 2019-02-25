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
def decision_classifier_plot(X_train,X_test,y_train,y_test,depths):
    train_results=[]
    test_results=[]
    depth_list = [8, 10, 15, 20, 50, 100, 120, 150, 175, 200]
    for depth_list in depth_list:    
        model=DecisionTreeClassifier(random_state=9,max_depth=depth_list)
        model.fit(X_train,y_train)
        train_pred=model.predict(X_train)
        accuracy=accuracy_score(y_train,train_pred)
        train_results.append(accuracy)

        test_pred=model.predict(X_test)
        accuracy1=accuracy_score(y_test,test_pred)
        test_results.append(accuracy1)
       
    plt.plot([8, 10, 15, 20, 50, 100, 120, 150, 175, 200],train_results,'b-',label='Train set')
    plt.plot([8, 10, 15, 20, 50, 100, 120, 150, 175, 200],test_results,'g-',label='Test set')
    plt.legend(loc='best')
    plt.xlabel('depths')
    plt.ylabel('mean accuracy score')
    plt.show()
    


