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

def decision_classifier_plot(X_train,X_test,y_train,y_test,depths):
    mse_train=[]
    mse_test = []
    for i in depths:
        model = DecisionTreeClassifier(random_state=9)
        model.fit(X_train,y_train)
        y_pred_train = accuracy_score(y_train,model.predict(X_train))
        y_pred_test = accuracy_score(y_test,model.predict(X_test))
        mse_train.append(mse_train)
        mse_test.append(mse_test)
        print(i)


    plt.plot(depth_list,mse_train)
    plt.title('mse_train')

    plt.plot(depth_list,mse_test)


    plt.show()



    return
# Write your solution here :
