from unittest import TestCase
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from inspect import getfullargspec
from ..build import my_decision_classifier

data = pd.read_csv("./data/loan_prediction.csv")
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)

param_grid = {"max_depth": [8, 10, 15, 20],
              "max_leaf_nodes": [2, 5, 9, 15, 20],
              "max_features": [1, 2, 3, 5]}


class TestMy_decision_classifier(TestCase):
    def test_my_decision_classifier(self):

        # Input parameters tests
        args = getfullargspec(my_decision_classifier).args
        args_default = getfullargspec(my_decision_classifier).defaults
        self.assertEqual(len(args), 6, "Expected arguments %d, Given %d" % (6, len(args)))
        self.assertEqual(args_default, (10,), "Expected default values do not match given default values")

        # Return data types
        np.random.seed(9)
        accuracy, best_params = my_decision_classifier(X_train, X_test, y_train, y_test, param_grid, 10)

        self.assertIsInstance(accuracy, float,
                              "Expected data type for return value is `Float`, you are returning %s" % (
                                  type(accuracy)))
        self.assertIsInstance(best_params, dict,
                              "Expected data type for return value is `Dictionary`, you are returning %s" % (
                                  type(best_params)))

        # Return value tests

        self.assertAlmostEqual(accuracy, 0.751351351351, 5, "Return value does not match expected value")
        self.assertEqual(dict(best_params), {'max_leaf_nodes': 5, 'max_features': 3, 'max_depth': 10},
                         "Return best parameters does not match expected best parameters")