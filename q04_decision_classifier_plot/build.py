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
def decision_classifier_plot(X_train, X_test, y_train, y_test,depths):
    mean_train_score=[]
    mean_test_score=[]
    for depth in depths:
        
        clf=DecisionTreeClassifier(max_depth=depth)
        clf.fit(X_train,y_train)
        y_train_predictions=clf.predict(X_train)
        y_test_predictions=clf.predict(X_test)
        mean_train_score.append(accuracy_score(y_train,y_train_predictions))
        mean_test_score.append(accuracy_score(y_test,y_test_predictions))
    plt.plot(depths,mean_train_score)
    plt.plot(depths,mean_test_score)
    plt.show()

#decision_classifier_plot(X_train,X_test,y_train,y_test,depth_list)
    
        
        


