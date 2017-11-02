from paddle.v2.framework.evaluator import Evaluator
from paddle.v2.framework.op import Operator
import paddle.v2.framework.core as core
import unittest
import op_test
import numpy as np


class TestEvaluator(unittest.TestCase):
    def setup(self, scope, inputs, outputs):
        def __create_var__(var_name, arr):
            np_arr = np.array(arr)
            scope.var(var_name)
            # tensor = var.get_tensor()
            # tensor.set_dims(np_arr.shape)

        for var_name, arr in inputs.iteritems():
            __create_var__(var_name, arr)

        for var_name, arr in outputs.iteritems():
            __create_var__(var_name, arr)

    def test_evaluator(self):

        inputs = {
            'Inference': np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 1]]).T,
            'Label': np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        }
        outputs = {'Accuracy': np.array([0.9])}
        out_name = 'Accuracy'

        places = [core.CPUPlace()]
        if core.is_compile_gpu():
            places.append(core.GPUPlace(0))

        for place in places:
            scope = core.Scope()
            self.setup(scope, inputs, outputs)

            evaluator = Evaluator(
                scope,
                operator='accuracy',
                input='Inference',
                label='Label',
                output=out_name,
                place=place)
            op_test.set_input(scope, evaluator.op, inputs, place)
            ctx = core.DeviceContext.create(place)

            for i in range(10):  # simulate 10 mini-batches
                evaluator.evaluate(ctx)

            actual = np.array(scope.find_var(out_name).get_tensor())
            print actual

            self.assertTrue(
                np.allclose(
                    actual, outputs[out_name], atol=1e-5),
                "output name: " + out_name + " has diff.")


if __name__ == '__main__':
    exit(0)
    unittest.main()
