from __future__ import print_function
import unittest

import paddle.v2.fluid.layers as layers
import paddle.v2.fluid.optimizer as optimizer
from paddle.v2.fluid.framework import Program, program_guard
from paddle.v2.fluid.memory_optimization_transpiler import memory_optimize, ControlFlowGraph


class TestControlFlowGraph(unittest.TestCase):
    def setUp(self):
        program = Program()
        with program_guard(program, startup_program=Program()):
            x = layers.data(name='x', shape=[13], dtype='float32')
            y_predict = layers.fc(input=x, size=1, act=None)
            y = layers.data(name='y', shape=[1], dtype='float32')
            cost = layers.square_error_cost(input=y_predict, label=y)
            avg_cost = layers.mean(x=cost)
            opt = optimizer.SGD(learning_rate=0.001)
            opt = opt.minimize(avg_cost)

        self.program = program

    def test_control_flow_graph(self):
        graph = ControlFlowGraph(self.program)
        graph._build_graph()
        graph.save_visualize_graph('a.png')

    def test_memory_optimization(self):
        print("before optimization")
        # print(str(self.program))
        result_program = memory_optimize(self.program)
        print("after optimization")
        print(str(result_program))


if __name__ == "__main__":
    unittest.main()
