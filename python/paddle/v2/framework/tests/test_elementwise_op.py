import unittest
import numpy as np
from op_test import OpTest


def generator():
    l = []

    x = np.random.uniform(0.1, 1, [13, 17]).astype("float32")
    y = np.random.uniform(0.1, 1, [13, 17]).astype("float32")
    d = {
        'op_type': "elementwise_mul",
        'inputs': {
            'X': x,
            'Y': y
        },
        'outputs': {
            'Out': np.multiply(x, y)
        }
    }

    l.append(d)

    for t in l:
        yield t


class DataProvider(type):
    def __new__(cls, name, bases, metaattrs):
        for t in generator():
            metaattrs['op_type'] = t['op_type']
            print metaattrs['op_type']
            metaattrs['inputs'] = t['inputs']
            metaattrs['outputs'] = t['outputs']
            if 'attr' in t:
                metaattrs['attr'] = t['attr']

            return type.__new__(cls, name, bases, metaattrs)


class ElementwiseOp(OpTest, object):
    __metaclass__ = DataProvider
    '''
    def setUp(self):
        self.op_type = "elementwise"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [13, 17]).astype("float32"),
            'Y': np.random.uniform(0.1, 1, [13, 17]).astype("float32")
        }
        self.outputs = {'Out': np.multiply(self.inputs['X'], self.inputs['Y'])}
    '''

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out', max_relative_error=0.1)

    def test_check_grad_ingore_x(self):
        self.check_grad(
            ['Y'], 'Out', max_relative_error=0.1, no_grad_set=set("X"))

    def test_check_grad_ingore_y(self):
        self.check_grad(
            ['X'], 'Out', max_relative_error=0.1, no_grad_set=set('Y'))


if __name__ == '__main__':
    unittest.main()
