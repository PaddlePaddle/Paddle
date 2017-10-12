import unittest

import paddle.v2.framework.graph as graph
import paddle.v2.framework.optimizer as optimizer


class TestOptimizer(unittest.TestCase):
    def test_sgd_optimizer(self):
        program = graph.g_program
        block = program.global_block()
        mul_x = block.create_var(
            dtype="float32", shape=[5, 10], lod_level=0, name="mul.x")
        mul_y = block.create_var(
            dtype="float32", shape=[10, 8], lod_level=0, name="mul.y")
        mul_out = block.create_var(
            dtype="float32", shape=[5, 8], lod_level=0, name="mul.out")
        mul_op = block.append_op(
            type="mul",
            inputs={"X": mul_x,
                    "Y": mul_y},
            outputs={"Out": [mul_out]},
            attrs={"x_num_col_dims": 1})
        loss = block.create_var("loss")
        sgd_optimizer = optimizer.SGDOptimizer(learning_rate=0.01)
        opts = sgd_optimizer.minimize(loss)
        print(opts)


if __name__ == '__main__':
    unittest.main()
