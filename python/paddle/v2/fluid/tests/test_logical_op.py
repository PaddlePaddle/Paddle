import op_test
import unittest
import numpy as np


def create_test_class(op_type, callback, binary_op=True):
    class Cls(op_test.OpTest):
        def setUp(self):
            a = np.random.choice(a=[True, False], size=(10, 7)).astype(bool)
            if binary_op:
                b = np.random.choice(a=[True, False], size=(10, 7)).astype(bool)
                c = callback(a, b)
            else:
                c = callback(a)
            self.outputs = {'Out': c}
            self.op_type = op_type
            if binary_op:
                self.inputs = {'X': a, 'Y': b}
            else:
                self.inputs = {'X': a}

        def test_output(self):
            self.check_output()

    Cls.__name__ = op_type
    globals()[op_type] = Cls


create_test_class('logical_and', lambda _a, _b: np.logical_and(_a, _b))
create_test_class('logical_or', lambda _a, _b: np.logical_or(_a, _b))
create_test_class('logical_not', lambda _a: np.logical_not(_a), False)
create_test_class('logical_xor', lambda _a, _b: np.logical_xor(_a, _b))

if __name__ == '__main__':
    unittest.main()
