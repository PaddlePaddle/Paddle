import unittest
import numpy as np
import paddle.v2.framework.core as core
from paddle.v2.framework.op import Operator
from op_test import OpTest
import math


class TestAdagradOp1(OpTest):
    ''' Test Adagrad operator with explicit attributes
    '''

    def setUp(self):
        self.op_type = "adagrad"

        param = np.random.random((123, 321)).astype("float32")
        grad = np.random.random((123, 321)).astype("float32")
        moment = np.zeros((123, 321)).astype("float32")
        lr = 0.01
        epsilon = 1e-8

        self.inputs = {
            'Param': param,
            'Grad': grad,
            'Moment': moment,
            'LearningRate': np.array([lr]).astype("float32")
        }

        self.attrs = {'epsilon': epsilon}

        moment_out = moment + grad * grad
        param_out = param - lr * grad / (np.sqrt(moment_out) + epsilon)

        self.outputs = {'ParamOut': param_out, 'MomentOut': moment_out}

    def test_check_output(self):
        self.check_output()


class TestAdagradOp2(OpTest):
    ''' Test Adagrad operator with default attributes
    '''

    def setUp(self):
        self.op_type = "adagrad"

        param = np.random.random((123, 321)).astype("float32")
        grad = np.random.random((123, 321)).astype("float32")
        moment = np.zeros((123, 321)).astype("float32")
        lr = 0.01
        epsilon = 1e-6

        self.inputs = {
            'Param': param,
            'Grad': grad,
            'Moment': moment,
            'LearningRate': np.array([lr]).astype("float32")
        }

        self.attrs = {'epsilon': epsilon}

        moment_out = moment + grad * grad
        param_out = param - lr * grad / (np.sqrt(moment_out) + epsilon)

        self.outputs = {'ParamOut': param_out, 'MomentOut': moment_out}

    def test_check_output(self):
        self.check_output()


class TestSparseAdagradOp(unittest.TestCase):
    def check_with_place(self, place):
        scope = core.Scope()

        # create and initialize Grad Variable   
        height = 10
        rows = [0, 4, 7]
        row_numel = 12

        grad_selected_rows = scope.var('Grad').get_selected_rows()
        grad_selected_rows.set_height(height)
        grad_selected_rows.set_rows(rows)
        np_array = np.ones((len(rows), row_numel)).astype("float32")
        np_array[0, 0] = 2.0
        np_array[2, 8] = 4.0

        grad_tensor = grad_selected_rows.get_tensor()
        grad_tensor.set(np_array, place)

        # create and initialize Param Variable
        param = scope.var('Param').get_tensor()
        param_array = np.full((height, row_numel), 5.0).astype("float32")
        param.set(param_array, place)

        # create and initialize LeraningRate Variable
        lr = scope.var('LearningRate').get_tensor()
        lr_array = np.full((1), 2.0).astype("float32")
        lr.set(lr_array, place)

        # create and initialize moment Variable
        moment = scope.var('Moment').get_tensor()
        moment_np_array = np.full((height, row_numel), 2.0).astype("float32")
        moment.set(moment_np_array, place)

        # create and run sgd operator
        adagrad_op = Operator(
            "adagrad",
            Param='Param',
            Grad='Grad',
            ParamOut='Param',
            Moment='Moment',
            MomentOut='Moment',
            LearningRate='LearningRate',
            epsilon=2.0)

        ctx = core.DeviceContext.create(place)
        adagrad_op.run(scope, ctx)

        # get and compare moment result
        moment_result_array = np.array(moment)

        self.assertAlmostEqual(6.0, moment_result_array[rows[0], 0])
        self.assertAlmostEqual(3.0, moment_result_array[rows[0], 2])
        self.assertAlmostEqual(2.0, moment_result_array[1, 0])
        self.assertAlmostEqual(3.0, moment_result_array[rows[1], 10])
        self.assertAlmostEqual(2.0, moment_result_array[5, 8])
        self.assertAlmostEqual(3.0, moment_result_array[rows[2], 1])
        self.assertAlmostEqual(18.0, moment_result_array[rows[2], 8])

        # get and compare param result
        result_array = np.array(param)

        def get_out(param, lr, grad, m, epsilon):
            return param - lr * grad / (math.sqrt(m) + epsilon)

        self.assertAlmostEqual(
            get_out(5.0, 2.0, 2.0, 6.0, 2.0),
            result_array[rows[0], 0],
            places=5)
        self.assertAlmostEqual(
            get_out(5.0, 2.0, 1.0, 3.0, 2.0),
            result_array[rows[0], 2],
            places=5)
        self.assertAlmostEqual(
            get_out(5.0, 2.0, 0.0, 2.0, 2.0), result_array[1, 0], places=5)
        self.assertAlmostEqual(
            get_out(5.0, 2.0, 1.0, 3.0, 2.0),
            result_array[rows[1], 10],
            places=5)
        self.assertAlmostEqual(
            get_out(5.0, 2.0, 0.0, 2.0, 2.0), result_array[5, 8], places=5)
        self.assertAlmostEqual(
            get_out(5.0, 2.0, 1.0, 3.0, 2.0),
            result_array[rows[2], 1],
            places=5)
        self.assertAlmostEqual(
            get_out(5.0, 2.0, 4.0, 18.0, 2.0),
            result_array[rows[2], 8],
            places=5)

    def test_sparse_adagrad(self):
        places = [core.CPUPlace()]
        if core.is_compile_gpu():
            places.append(core.GPUPlace(0))
        for place in places:
            self.check_with_place(place)


if __name__ == "__main__":
    unittest.main()
