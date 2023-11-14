#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from mFgo.nlp.nlp import *
import unittest

class SimpleTest(unittest.TestCase):
    """Test the gaussian regression"""
    def test(self):
        mod = mfG()
        self.assertTrue(True)

if __name__ == "__main__":
    unittest.main()

