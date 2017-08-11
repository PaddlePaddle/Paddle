import paddle.v2.framework.core as core
from paddle.v2.framework.op import Operator
import unittest


def fc(X, W, Y):
    ret_v = core.Net.create()

    ret_v.add_op(Operator("mul", X="X", Y="W", Out="pre_activation"))
    ret_v.add_op(Operator("sigmoid", X="pre_activation", Y=Y))
    ret_v.complete_add_op(True)
    return ret_v


class TestNet(unittest.TestCase):
    def test_net_all(self):
        net = core.Net.create()
        op1 = Operator("add_two", X="X", Y="Y", Out="Out")
        net.add_op(op1)

        net2 = core.Net.create()
        net2.add_op(fc(X="X", W="w", Y="fc.out"))
        net2.complete_add_op(True)
        net.add_op(net2)
        net.complete_add_op(True)

        expected = '''
Op(plain_net), inputs:(W, X, Y), outputs:(Out, fc.out, pre_activation).
    Op(add_two), inputs:(X, Y), outputs:(Out).
    Op(plain_net), inputs:(W, X), outputs:(fc.out, pre_activation).
        Op(plain_net), inputs:(W, X), outputs:(fc.out, pre_activation).
            Op(mul), inputs:(X, W), outputs:(pre_activation).
            Op(sigmoid), inputs:(pre_activation), outputs:(fc.out).
'''
        self.assertEqual(expected, "\n" + str(net))


if __name__ == '__main__':
    unittest.main()
