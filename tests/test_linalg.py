# -*- coding: utf-8 -*-
"""
"""
import unittest
import numpy as np
import unittest_numpy as utn
import sl_py_tools.numpy_tricks.linalg as la

errstate = utn.errstate(invalid='raise')
# =============================================================================

# =============================================================================
# %% Test BLAS ufuncs
# =============================================================================


class TestShape(utn.TestCaseNumpy):
    """Testing norm, matmul and rmatmul"""
    def setUp(self):
        super().setUp()
        self.sctype.append('i')
        self.x = {}
        for sctype in self.sctype:
            self.x[sctype] = utn.randn_asa((2, 3, 5), sctype)

    def test_shape(self):
        """Check that norm returns arrays with the expected shape
        """
        # shape


# =============================================================================
if __name__ == '__main__':
    unittest.main(verbosity=2)
#    ut.main(testRunner=ut.TextTestRunner(resultclass=utn.TestResultNumpy))
