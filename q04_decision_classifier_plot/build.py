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
    lst  = []
    for depth in depth_list:
        tree_reg = DecisionTreeClassifier(max_depth=depth,random_state=9)
        tree_reg.fit(X_train, y_train)
        y_pred_test=tree_reg.predict(X_test)
        y_pred_train=tree_reg.predict(X_train)
        lst.append((depth,accuracy_score(y_train,y_pred_train),accuracy_score(y_test,y_pred_test)))
    df =  pd.DataFrame(lst)
    plt.plot(df.iloc[:,0],df.iloc[:,1],c='r',label = 'Train')
    plt.plot(df.iloc[:,0],df.iloc[:,2],c='g',label = 'Test')
    plt.legend()
    return
