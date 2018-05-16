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
def decision_classifier_plot(X_train, X_test, y_train, y_test, depths):
    dtree = DecisionTreeClassifier(random_state=9)
    param_grid = { "max_depth" : depths }
    random_cv = RandomizedSearchCV(estimator=dtree, param_distributions=param_grid, n_iter=10, scoring='accuracy')
    random_cv.fit(X_train, y_train)
    y_pred_test = random_cv.predict(X_test)
    df_results = pd.DataFrame(data=random_cv.cv_results_)
    # display(df_results)
    train_mse = plt.plot(df_results.param_max_depth, df_results.split0_train_score, color='b',label='Train_Mean_Accuracy')
    test_mse = plt.plot(depth_list, df_results.split0_test_score, color='g',label='Test_Mean_Accuracy')
    plt.xlabel('Tree Depth')
    plt.ylabel('Mean Accuracy_Score')
    plt.legend()
    plt.show()
