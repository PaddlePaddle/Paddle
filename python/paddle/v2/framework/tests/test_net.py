import paddle.v2.framework.core as core
from paddle.v2.framework.create_op_creation_methods import op_creations
import unittest


class TestNet(unittest.TestCase):
    def test_net_all(self):
        net = core.Net.create()
        op1 = op_creations.add_two(X="X", Y="Y", Out="Out")
        net.add_op(op1)

        net2 = core.Net.create()
        net2.add_op(op_creations.fc(X="X", W="w", Y="fc.out"))
        net2.complete_add_op(True)
        net.add_op(net2)
        net.complete_add_op(True)

        expected = '''
Op(plain_net), inputs:(@EMPTY@, X, Y, w), outputs:(@TEMP@fc@0, Out, fc.out).
    Op(add_two), inputs:(X, Y), outputs:(Out).
    Op(plain_net), inputs:(@EMPTY@, X, w), outputs:(@TEMP@fc@0, fc.out).
        Op(fc), inputs:(X, w, @EMPTY@), outputs:(fc.out, @TEMP@fc@0).
            Op(mul), inputs:(X, w), outputs:(@TEMP@fc@0).
            Op(sigmoid), inputs:(@TEMP@fc@0), outputs:(fc.out).
'''
        self.assertEqual(expected, "\n" + str(net))


if __name__ == '__main__':
    unittest.main()
