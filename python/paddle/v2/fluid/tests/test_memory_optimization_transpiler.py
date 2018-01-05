from __future__ import print_function
import unittest

import paddle.v2.fluid.layers as layers
import paddle.v2.fluid.nets as nets
from paddle.v2.fluid.framework import Program, program_guard
from paddle.v2.fluid.param_attr import ParamAttr
from paddle.v2.fluid.memory_optimization_transpiler import ControlFlowGraph


class TestControlFlowGraph(unittest.TestCase):
    def setUp(self):
        program = Program()
        with program_guard(program, startup_program=Program()):
            x = layers.data(name='x', shape=[13], dtype='float32')
            y_predict = layers.fc(input=x, size=1, act=None)
            y = layers.data(name='y', shape=[1], dtype='float32')
            cost = layers.square_error_cost(input=y_predict, label=y)
            avg_cost = layers.mean(x=cost)
            self.assertIsNotNone(avg_cost)
            program.append_backward(avg_cost)
        self.program = program

    def test_control_flow_graph(self):
        # print(str(self.program))
        self.graph = ControlFlowGraph(self.program)
        self.graph.memory_optimize()
        result_program = self.graph.get_program()


if __name__ == "__main__":
    unittest.main()
