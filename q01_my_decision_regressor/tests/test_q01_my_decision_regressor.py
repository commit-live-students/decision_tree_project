from unittest import TestCase
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ..build import my_decision_regressor
from inspect import getfullargspec

data = pd.read_csv("./data/house_pricing.csv")
np.random.seed(9)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)

param_grid = {"max_depth": [2, 3, 5, 6, 8, 10, 15, 20, 30, 50],
              "max_leaf_nodes": [2, 3, 4, 5, 10, 15, 20],
              "max_features": [4, 8, 20, 25]}


class TestMy_decision_regressor(TestCase):
    def test_my_decision_regressor(self):

        # Input parameters tests
        args = getfullargspec(my_decision_regressor).args
        args_default = getfullargspec(my_decision_regressor).defaults
        self.assertEqual(len(args), 5, "Expected arguments %d, Given %d" % (5, len(args)))
        self.assertEqual(args_default, None, "Expected default values do not match given default values")

        # Return data types
        r_square, best_params = my_decision_regressor(X_train, X_test, y_train, y_test, param_grid)

        self.assertIsInstance(r_square, float,
                              "Expected data type for return value is `Float`, you are returning %s" % (
                                  type(r_square)))
        self.assertIsInstance(best_params, dict,
                              "Expected data type for return value is `Float`, you are returning %s" % (
                                  type(best_params)))

        # Return value tests

        self.assertAlmostEqual(r_square, 0.597277463587, 5, "Return value does not match expected value")
        self.assertEqual(dict(best_params), {'max_leaf_nodes': 20, 'max_features': 25, 'max_depth': 3},
                         "Return value does not match expected value")
