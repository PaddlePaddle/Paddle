import unittest
import numpy as np
from op_test import OpTest


def generator():
    l = []

    #matrix * matrix
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

    #vector * vector
    x = np.random.random((32, )).astype("float32")
    y = np.random.random((32, )).astype("float32")
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

    #broad cast axis 0
    x = np.random.rand(2, 3, 4).astype(np.float32)
    y = np.random.rand(2).astype(np.float32)
    d = {
        'op_type': "elementwise_mul",
        'inputs': {
            'X': x,
            'Y': y
        },
        'outputs': {
            'Out': np.multiply(x, y.reshape(2, 1, 1))
        },
        'attrs': {
            'axis': 0
        }
    }
    l.append(d)

    #broad cast axis 1
    x = np.random.rand(2, 3, 4).astype(np.float32)
    y = np.random.rand(3).astype(np.float32)
    d = {
        'op_type': "elementwise_mul",
        'inputs': {
            'X': x,
            'Y': y
        },
        'outputs': {
            'Out': np.multiply(x, y.reshape(1, 3, 1))
        },
        'attrs': {
            'axis': 1
        }
    }
    l.append(d)

    #broad cast axis 2
    x = np.random.rand(2, 3, 4).astype(np.float32)
    y = np.random.rand(4).astype(np.float32)
    d1 = {
        'op_type': "elementwise_mu",
        'inputs': {
            'X': x,
            'Y': y
        },
        'outputs': {
            'Out': np.multiply(x, y.reshape(1, 1, 4))
        },
        'attrs': {
            'axis': 2
        }
    }
    l.append(d1)

    #broad cast with axis 2
    x = np.random.rand(2, 3, 4, 5).astype(np.float32)
    y = np.random.rand(3, 4).astype(np.float32)
    d = {
        'op_type': "elementwise_mul",
        'inputs': {
            'X': x,
            'Y': y
        },
        'outputs': {
            'Out': np.multiply(x, y.reshape(1, 3, 4, 1))
        },
        'attrs': {
            'axis': 1
        }
    }
    l.append(d)

    for t in l:
        yield t


def test_check_output(self):
    self.check_output()


def test_check_grad_normal(self):
    self.check_grad(['X', 'Y'], 'Out', max_relative_error=0.1)


def test_check_grad_ingore_x(self):
    self.check_grad(['Y'], 'Out', max_relative_error=0.1, no_grad_set=set("X"))


def test_check_grad_ingore_y(self):
    self.check_grad(['X'], 'Out', max_relative_error=0.1, no_grad_set=set('Y'))


c = []
for t in generator():
    c.append(
        type('Test', (OpTest, ), {
            'test_check_output': test_check_output,
            'test_check_grad_normal': test_check_grad_normal,
            'test_check_grad_ingore_x': test_check_grad_ingore_x,
            'test_check_grad_ingore_y': test_check_grad_ingore_y,
            'op_type': t['op_type'],
            'inputs': t['inputs'],
            'attrs': t['attrs'] if 'attrs' in t else dict(),
            'outputs': t['outputs']
        }))
"""
for t in c:
    print t.op_type
"""

if __name__ == '__main__':
    unittest.main()
