from unittest import TestCase
from inspect import getfullargspec
from ..build import decision_regressor_plot


class TestDecision_regressor_plot(TestCase):
    def test_decision_regressor_plot(self):

        # Input parameters tests
        args = getfullargspec(decision_regressor_plot).args
        args_default = getfullargspec(decision_regressor_plot).defaults
        self.assertEqual(len(args), 5, "Expected arguments %d, Given %d" % (5, len(args)))
        self.assertEqual(args_default, None, "Expected default values do not match given default values")

        # Return type tests
        # Nothing to check here

        # Return value tests
        # Nothing to check here
