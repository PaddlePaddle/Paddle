import unittest

import paddle.v2.framework.graph as graph
import paddle.v2.framework.optimizer as optimizer


class TestOptimizer(unittest.TestCase):
    def test_sgd_optimizer(self):
        program = graph.g_program
        block = program.global_block()
        init_op = block.append_op(
            type="mul", inputs={}, outputs={"Out": "out1"})
        block.create_var()
        optimizer = optimizer.SGDOptimizer()


if __name__ == '__main__':
    unittest.main()
