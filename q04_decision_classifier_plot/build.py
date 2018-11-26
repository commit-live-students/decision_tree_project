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

def decision_classifier_plot (X_train, X_test, y_train, y_test ,depths):
    np.random.seed(9)
    test_scores =[]
    train_scores =[]
    etest_scores =[]
    etrain_scores =[]
    for i in depths:
        dtm = DecisionTreeClassifier(max_depth=i )
        dtm.fit(X_train,y_train)
        train_scores.append(dtm.score(X_train,y_train))
        test_scores.append( dtm.score(X_test,y_test))
        y_tpred = dtm.predict(X_train)
        y_pred = dtm.predict(X_test)
        etrain_scores.append(accuracy_score(y_train, y_tpred))
        etest_scores.append(accuracy_score(y_test, y_pred))
    plt.plot(depths,etrain_scores)
    plt.plot(depths,etest_scores )
    plt.xlabel('Max Depth')
    plt.ylabel('Mean Square Error')
    plt.show()



