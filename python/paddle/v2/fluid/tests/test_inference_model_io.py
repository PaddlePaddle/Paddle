import unittest

import numpy as np
import paddle.v2.fluid.core as core

import paddle.v2.fluid.executor as executor
import paddle.v2.fluid.layers as layers
import paddle.v2.fluid.optimizer as optimizer
from paddle.v2.fluid.framework import Program
from paddle.v2.fluid.io import save_inference_model, load_inference_model


class TestBook(unittest.TestCase):
    def test_fit_line_inference_model(self):
        MODEL_DIR = "./tmp/inference_model"

        init_program = Program()
        program = Program()
        x = layers.data(
            name='x',
            shape=[2],
            dtype='float32',
            main_program=program,
            startup_program=init_program)
        y = layers.data(
            name='y',
            shape=[1],
            dtype='float32',
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
        sgd_optimizer.minimize(avg_cost, init_program)

        place = core.CPUPlace()
        exe = executor.Executor(place)

        exe.run(init_program, feed={}, fetch_list=[])

        for i in xrange(100):
            tensor_x = np.array(
                [[1, 1], [1, 2], [3, 4], [5, 2]]).astype("float32")
            tensor_y = np.array([[-2], [-3], [-7], [-7]]).astype("float32")

            exe.run(program,
                    feed={'x': tensor_x,
                          'y': tensor_y},
                    fetch_list=[avg_cost])

        save_inference_model(MODEL_DIR, ["x", "y"], [avg_cost], exe, program)
        expected = exe.run(program,
                           feed={'x': tensor_x,
                                 'y': tensor_y},
                           fetch_list=[avg_cost])[0]

        reload(executor)  # reload to build a new scope
        exe = executor.Executor(place)

        [infer_prog, feed_var_names, fetch_vars] = load_inference_model(
            MODEL_DIR, exe)

        outs = exe.run(
            infer_prog,
            feed={feed_var_names[0]: tensor_x,
                  feed_var_names[1]: tensor_y},
            fetch_list=fetch_vars)
        actual = outs[0]

        self.assertEqual(feed_var_names, ["x", "y"])
        self.assertEqual(len(fetch_vars), 1)
        self.assertEqual(str(fetch_vars[0]), str(avg_cost))
        self.assertEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
