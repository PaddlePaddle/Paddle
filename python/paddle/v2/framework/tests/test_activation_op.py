import unittest
import numpy as np
from op_test import OpTest


class TestExp(OpTest):
    def setUp(self):
        self.op_type = "exp"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [11, 17]).astype("float32")
        }
        self.outputs = {'Y': np.exp(self.inputs['X'])}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Y', max_relative_error=0.007)


class TestSigmoid(OpTest):
    def setUp(self):
        self.op_type = "sigmoid"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [11, 17]).astype("float32")
        }
        self.outputs = {'Y': 1 / (1 + np.exp(-self.inputs['X']))}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Y', max_relative_error=0.008)


class TestTanh(OpTest):
    def setUp(self):
        self.op_type = "tanh"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [11, 17]).astype("float32")
        }
        self.outputs = {'Y': np.tanh(self.inputs['X'])}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Y', max_relative_error=0.007)


class TestTanhShrink(OpTest):
    def setUp(self):
        self.op_type = "tanh_shrink"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [10, 17]).astype("float32")
        }
        self.outputs = {'Y': self.inputs['X'] - np.tanh(self.inputs['X'])}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Y', max_relative_error=0.008)


class TestSqrt(OpTest):
    def setUp(self):
        self.op_type = "sqrt"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [11, 17]).astype("float32")
        }
        self.outputs = {'Y': np.sqrt(self.inputs['X'])}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Y', max_relative_error=0.007)


class TestAbs(OpTest):
    def setUp(self):
        self.op_type = "abs"
        x = np.random.uniform(-1, 1, [4, 4]).astype("float32")
        # Because we set delta = 0.005 in caculating numeric gradient,
        # if x is too small, such as 0.002, x_neg will be -0.003
        # x_pos will be 0.007, so the numeric gradient is unaccurate.
        # we should avoid this
        x[np.abs(x) < 0.005] = 0.02
        self.inputs = {'X': x}
        self.outputs = {'Y': np.abs(self.inputs['X'])}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Y', max_relative_error=0.007)


class TestRelu(OpTest):
    def setUp(self):
        self.op_type = "relu"
        x = np.random.uniform(-1, 1, [11, 17]).astype("float32")
        # The same reason with TestAbs
        x[np.abs(x) < 0.005] = 0.02
        self.inputs = {'X': x}
        self.outputs = {'Y': np.maximum(self.inputs['X'], 0)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Y', max_relative_error=0.007)


class TestBRelu(OpTest):
    def setUp(self):
        self.op_type = "brelu"
        x = np.random.uniform(-1, 1, [4, 4]).astype("float32")
        t_min = 1
        t_max = 4
        # The same with TestAbs
        x[np.abs(x - t_min) < 0.005] = t_min + 0.02
        x[np.abs(x - t_max) < 0.005] = t_max + 0.02

        self.inputs = {'X': x}
        self.attrs = {'t_min': t_min, 't_max': t_max}
        t = np.copy(x)
        t[t < t_min] = t_min
        t[t > t_max] = t_max
        self.outputs = {'Y': t}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Y', max_relative_error=0.02)


class TestRelu6(OpTest):
    def setUp(self):
        self.op_type = "relu6"
        x = np.random.uniform(-1, 1, [4, 10]).astype("float32")
        threshold = 6.0
        # The same with TestAbs
        x[np.abs(x) < 0.005] = 0.02
        x[np.abs(x - threshold) < 0.005] = threshold + 0.02

        self.inputs = {'X': x}
        self.attrs = {'threshold': threshold}
        self.outputs = {
            'Y': np.minimum(np.maximum(self.inputs['X'], 0), threshold)
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Y', max_relative_error=0.02)


class TestSoftRelu(OpTest):
    def setUp(self):
        self.op_type = "soft_relu"
        x = np.random.uniform(-3, 3, [4, 4]).astype("float32")
        threshold = 2
        # The same reason with TestAbs
        x[np.abs(x - threshold) < 0.005] = threshold + 0.02
        x[np.abs(x + threshold) < 0.005] = -threshold + 0.02
        self.inputs = {'X': x}
        self.attrs = {'threshold': threshold}
        t = np.copy(x)
        t[t < -threshold] = -threshold
        t[t > threshold] = threshold
        self.outputs = {'Y': np.log((np.exp(t) + 1))}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Y', max_relative_error=0.02)


class TestReciprocal(OpTest):
    def setUp(self):
        self.op_type = "reciprocal"
        self.inputs = {'X': np.random.uniform(1, 2, [11, 17]).astype("float32")}
        self.outputs = {'Y': np.reciprocal(self.inputs['X'])}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Y', max_relative_error=0.01)


class TestLog(OpTest):
    def setUp(self):
        self.op_type = "log"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [11, 17]).astype("float32")
        }
        self.outputs = {'Y': np.log(self.inputs['X'])}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Y', max_relative_error=0.007)


class TestSquare(OpTest):
    def setUp(self):
        self.op_type = "square"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [11, 17]).astype("float32")
        }
        self.outputs = {'Y': np.square(self.inputs['X'])}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Y', max_relative_error=0.007)


class TestPow(OpTest):
    def setUp(self):
        self.op_type = "pow"
        self.inputs = {'X': np.random.uniform(1, 2, [11, 17]).astype("float32")}
        self.attrs = {'factor': 3}
        self.outputs = {'Y': np.power(self.inputs['X'], 3)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Y', max_relative_error=0.02)


class TestSTanh(OpTest):
    def setUp(self):
        self.op_type = "stanh"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [11, 17]).astype("float32")
        }
        scale_a = 2.0 / 3.0
        scale_b = 1.7159
        self.attrs = {'scale_a': scale_a, 'scale_b': scale_b}
        self.outputs = {'Y': scale_b * np.tanh(self.inputs['X'] * scale_a)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Y', max_relative_error=0.007)


class TestSoftsign(OpTest):
    def setUp(self):
        self.op_type = "softsign"
        self.inputs = {
            'X': np.random.uniform(-1, 1, [11, 17]).astype("float32")
        }
        self.outputs = {
            'Y': np.divide(self.inputs['X'], 1 + np.abs(self.inputs['X']))
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Y', max_relative_error=0.007)


if __name__ == "__main__":
    unittest.main()
