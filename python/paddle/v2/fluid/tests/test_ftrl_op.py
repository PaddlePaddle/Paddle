import unittest
import numpy as np
from op_test import OpTest


class TestFTRLOp(OpTest):
    def setUp(self):
        self.op_type = "ftrl"
        w = np.random.random((102, 105)).astype("float32")
        g = np.random.random((102, 105)).astype("float32")
        sq_accum = np.full((102, 105), 0.1).astype("float32")
        linear_accum = np.full((102, 105), 0.1).astype("float32")
        lr = np.array([0.01]).astype("float32")
        l1 = 0.1
        l2 = 0.2
        lr_power = -0.5

        self.inputs = {
            'Param': w,
            'SquaredAccumulator': sq_accum,
            'LinearAccumulator': linear_accum,
            'Grad': g,
            'LearningRate': lr
        }
        self.attrs = {
            'l1': l1,
            'l2': l2,
            'lr_power': lr_power,
            'learning_rate': lr
        }
        new_accum = sq_accum + g * g
        if lr_power == -0.5:
            linear_out = linear_accum + g - (
                (np.sqrt(new_accum) - np.sqrt(sq_accum)) / lr) * w
        else:
            linear_out = linear_accum + g - ((np.power(
                new_accum, -lr_power) - np.power(sq_accum, -lr_power)) / lr) * w

        x = (l1 * np.sign(linear_out) - linear_out)
        if lr_power == -0.5:
            y = (np.sqrt(new_accum) / lr) + (2 * l2)
            pre_shrink = x / y
            param_out = np.where(np.abs(linear_out) > l1, pre_shrink, 0.0)
        else:
            y = (np.power(new_accum, -lr_power) / lr) + (2 * l2)
            pre_shrink = x / y
            param_out = np.where(np.abs(linear_out) > l1, pre_shrink, 0.0)

        sq_accum_out = sq_accum + g * g

        self.outputs = {
            'ParamOut': param_out,
            'SquaredAccumOut': sq_accum_out,
            'LinearAccumOut': linear_out
        }

    def test_check_output(self):
        self.check_output()


if __name__ == "__main__":
    unittest.main()
