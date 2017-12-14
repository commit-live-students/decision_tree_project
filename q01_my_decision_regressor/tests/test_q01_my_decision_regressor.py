from unittest import TestCase
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ..build import my_decision_regressor
from inspect import getargspec

data = pd.read_csv("./data/house_pricing.csv")
np.random.seed(9)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)

param_grid = {"max_depth": [2, 3, 5, 6, 8, 10, 15, 20, 30, 50],
              "max_leaf_nodes": [2, 3, 4, 5, 10, 15, 20],
              "max_features": [4, 8, 20, 25]}
r_square, best_params = my_decision_regressor(X_train, X_test, y_train, y_test, param_grid)

class TestMy_decision_regressor(TestCase):
    def test_my_decision_regressor_arguments(self):

        # Input parameters tests
        args = getargspec(my_decision_regressor)
        self.assertEqual(len(args[0]), 5, "Expected arguments %d, Given %d" % (5, len(args[0])))
    def test_my_decision_regressor_defaults(self):
        args=getargspec(my_decision_regressor)
        self.assertEqual(args[3], None, "Expected default values do not match given default values")

        # Return data types
    def test_my_decision_regressor_return_r2_score_type(self):   

        self.assertIsInstance(r_square, float,
                              "Expected data type for return value is `Float`, you are returning %s" % (
                                  type(r_square)))

    def test_my_decision_regressor_best_params_type(self):
        self.assertIsInstance(best_params, dict,
                              "Expected data type for return value is `Float`, you are returning %s" % (
                                  type(best_params)))

        # Return value tests
    def test_my_decision_regressor_return_r2_score_value(self):
        self.assertAlmostEqual(r_square, 0.597277463587, 5, "Return value does not match expected value")
        
    def test_my_decision_regressor_return_best_parms_values(self):
        self.assertEqual(dict(best_params), {'max_leaf_nodes': 20, 'max_features': 25, 'max_depth': 3},
                         "Return value does not match expected value")
