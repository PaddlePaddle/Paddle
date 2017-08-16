from paddle.v2.framework.network import Network
import unittest


class TestNetwork(unittest.TestCase):
    def test_all(self):
        net = Network()
        out = net.add_two(X="A", Y="B")
        out = net.mul(X=out, Y="D")
        net2 = Network()
        net2.add_two(X=out, Y="E")
        net2.complete_add_op()
        net.add_op(net2)
        net.complete_add_op()
        self.assertEqual(
            '''Op(plain_net), inputs:{all[A, B, D, E]}, outputs:{all[add_two@GENERATE_OUTPUT@0, add_two@GENERATE_OUTPUT@2, mul@GENERATE_OUTPUT@1]}.
    Op(add_two), inputs:{X[A], Y[B]}, outputs:{Out[add_two@GENERATE_OUTPUT@0]}.
    Op(mul), inputs:{X[add_two@GENERATE_OUTPUT@0], Y[D]}, outputs:{Out[mul@GENERATE_OUTPUT@1]}.
    Op(plain_net), inputs:{all[E, mul@GENERATE_OUTPUT@1]}, outputs:{all[add_two@GENERATE_OUTPUT@2]}.
        Op(add_two), inputs:{X[mul@GENERATE_OUTPUT@1], Y[E]}, outputs:{Out[add_two@GENERATE_OUTPUT@2]}.
''', str(net))


if __name__ == '__main__':
    unittest.main()
