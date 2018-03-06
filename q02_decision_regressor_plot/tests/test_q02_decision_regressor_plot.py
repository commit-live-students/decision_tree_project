from unittest import TestCase
from inspect import getfullargspec
from ..build import decision_regressor_plot


class TestDecision_regressor_plot(TestCase):
    def test_decision_regressor_plot_arguments(self):

    # Input parameters tests
        args = getfullargspec(decision_regressor_plot)
        self.assertEqual(len(args[0]), 5, "Expected arguments %d, Given %d" % (5, len(args[0])))

    def test_decision_regressor_plot_defaults(self):
        args = getfullargspec(decision_regressor_plot)
        self.assertEqual(args[3], None, "Expected default values do not match given default values")

    # Return type tests
    # Nothing to check here

    # Return value tests
    # Nothing to check here
