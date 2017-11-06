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
def decision_classifier_plot(X_train, X_test, y_train, y_test,depths):
    train_list=[]
    test_list=[]
    for i in depths:
        ds=DecisionTreeClassifier(max_depth=i)
        ds.fit(X_train,y_train)
        trainres=ds.predict(X_train)
        msetrain=accuracy_score(y_train,trainres)
        train_list.append(msetrain)
        testres=ds.predict(X_test)
        msetest=accuracy_score(y_test,testres)
        test_list.append(msetest)

    #print test_list
    #print train_list

    plt.plot(depths,test_list)
    plt.plot(depths,train_list)
    #plt.show()
