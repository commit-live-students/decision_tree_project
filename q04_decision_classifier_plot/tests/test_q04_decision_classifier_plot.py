from unittest import TestCase
from inspect import getargspec
from ..build import decision_classifier_plot


class TestDecision_classifier_plot(TestCase):
    def test_decision_classifier_plot_arguments(self):

    # Input parameters tests
        args = getargspec(decision_classifier_plot)
        self.assertEqual(len(args[0]), 5, "Expected arguments %d, Given %d" % (5, len(args[0])))
    def test_decision_classifier_plot_defaults(self):
        args= getargspec(decision_classifier_plot)
        self.assertEqual(args[3], None, "Expected default values do not match given default values")

        # Return type tests
        # Nothing to check here

        # Return value tests
        # Nothing to check here
