import paddle.v2 as paddle
import paddle.v2.framework.layers as layers
import paddle.v2.framework.core as core
import paddle.v2.framework.optimizer as optimizer

from paddle.v2.framework.framework import Program
from paddle.v2.framework.io import save_inference_model, load_inference_model
import paddle.v2.framework.executor as executor
import unittest
import numpy as np


class TestBook(unittest.TestCase):
    def test_fit_line_inference_model(self):
        MODEL_DIR = "./tmp/inference_model"

        init_program = Program()
        program = Program()
        x = layers.data(
            name='x',
            shape=[2],
            data_type='float32',
            main_program=program,
            startup_program=init_program)
        y = layers.data(
            name='y',
            shape=[1],
            data_type='float32',
            main_program=program,
            startup_program=init_program)

        y_predict = layers.fc(input=x,
                              size=1,
                              act=None,
                              main_program=program,
                              startup_program=init_program)

        cost = layers.square_error_cost(
            input=y_predict,
            label=y,
            main_program=program,
            startup_program=init_program)
        avg_cost = layers.mean(
            x=cost, main_program=program, startup_program=init_program)

        sgd_optimizer = optimizer.SGDOptimizer(learning_rate=0.001)
        opts = sgd_optimizer.minimize(avg_cost, init_program)

        place = core.CPUPlace()
        exe = executor.Executor(place)

        exe.run(init_program, feed={}, fetch_list=[])

        for i in xrange(100):
            x_data = np.array(
                [[1, 1], [1, 2], [3, 4], [5, 2]]).astype("float32")
            y_data = np.array([[-2], [-3], [-7], [-7]]).astype("float32")

            tensor_x = core.LoDTensor()
            tensor_x.set(x_data, place)
            tensor_y = core.LoDTensor()
            tensor_y.set(y_data, place)
            exe.run(program,
                    feed={'x': tensor_x,
                          'y': tensor_y},
                    fetch_list=[avg_cost])

        save_inference_model(MODEL_DIR, ["x", "y"], [avg_cost], exe, program)
        outs = exe.run(program,
                       feed={'x': tensor_x,
                             'y': tensor_y},
                       fetch_list=[avg_cost])
        expected = np.array(outs[0])

        reload(executor)  # reload to build a new scope
        exe = executor.Executor(place)

        [infer_prog, feed_var_names, fetch_vars] = load_inference_model(
            MODEL_DIR, exe)

        outs = exe.run(
            infer_prog,
            feed={feed_var_names[0]: tensor_x,
                  feed_var_names[1]: tensor_y},
            fetch_list=fetch_vars)
        actual = np.array(outs[0])

        self.assertEqual(feed_var_names, ["x", "y"])
        self.assertEqual(len(fetch_vars), 1)
        self.assertEqual(str(fetch_vars[0]), str(avg_cost))
        self.assertEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
