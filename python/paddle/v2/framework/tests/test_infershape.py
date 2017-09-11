import unittest

import paddle.v2.framework.core as core
from paddle.v2.framework.op import Operator
from paddle.v2.framework.variable import Variable


class TestCompileTimeInferShape(unittest.TestCase):
    def test_all(self):
        out = Variable("output")
        var_map = {
            "input1": Variable(
                "input1", dims=[1, 2]).desc(),
            "input2": Variable(
                "input1", dims=[1, 3]).desc(),
            "output": out.desc()
        }
        mul_op = Operator.desc("mul", X="input1", Y="input2", Out="output")
        core.Operator.infer_shape_compile(mul_op, var_map)
        print(str(out.desc()))


if __name__ == "__main__":
    unittest.main()
