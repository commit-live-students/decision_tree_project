# default imports
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.pyplot.switch_backend('agg')

data = pd.read_csv("./data/loan_prediction.csv")
np.random.seed(9)
X = data.iloc[:, :-1]

y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)

depth_list = [8, 10, 15, 20, 50, 100, 120, 150, 175, 200]


# Write your solution here :
def decision_classifier_plot(X_train, X_test, y_train, y_test,depth_list):
    dlist = []
    mse_error1 = []
    mse_error2 = []
    for d in depth_list:
        dt = DecisionTreeClassifier(max_depth=d)
        dt.fit(X_train,y_train)
        ypred1 = dt.predict(X_test)
        ypred2 = dt.predict(X_train)
        mse1 = accuracy_score(y_test,ypred1)
        mse2 = accuracy_score(y_train,ypred2)
        dlist.append(d)
        mse_error1.append(mse1)
        mse_error2.append(mse2)

    return dlist,mse_error1,mse_error2

dlist,mse_error1,mse_error2 = decision_classifier_plot(X_train, X_test, y_train, y_test,depth_list)

#print dlist,error,error1


plt.figure(figsize=(10,10))
plt.plot(dlist,mse_error1,label='Test Set')
plt.plot(dlist,mse_error2,label='Train Set')
plt.legend()
plt.show()
