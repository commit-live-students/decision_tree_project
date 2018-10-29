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

param_grid = {'max_depth': [8, 10, 15, 20],
              'max_leaf_nodes': [2, 5, 9, 15, 20],
              'max_features': [1, 2, 3, 5]}



depth_list = [8, 10, 15, 20, 50, 100, 120, 150, 175, 200]
res = []
acc_list=[]
def decision_classifier_plot(X_train,X_test,y_train,y_test,depth_list):
    for i in depth_list:
        dtc = DecisionTreeClassifier(max_depth=i,random_state=9)
        rscv = RandomizedSearchCV(dtc,param_grid,n_iter=10)
        model = rscv.fit(X_train,y_train)
        y_pred1 = model.predict(X_train)
        acc = accuracy_score(y_train,y_pred1)
        res.append(acc)


        y_pred2 = model.predict(X_test)
        ac = accuracy_score(y_test,y_pred2)
        acc_list.append(ac)
    plt.plot(depth_list,res)
    plt.plot(depth_list,acc_list)


c = decision_classifier_plot(X_train,X_test,y_train,y_test,depth_list)


