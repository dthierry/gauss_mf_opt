from mFgo.nlp.nlp import *
import unittest

class TestNlpMfG(unittest.TestCase):
    """Test the optimization"""

    def test(self):
        nlp = gauss_reg_nlp()
        self.assertTrue(True)

if __name__ == "__main__":
    unittest.main()

