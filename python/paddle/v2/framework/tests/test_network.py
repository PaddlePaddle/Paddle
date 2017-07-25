from paddle.v2.framework.network import Network
import paddle.v2.framework.core as core
import unittest


class TestNet(unittest.TestCase):
    def test_net_all(self):
        net = Network()
        out = net.add_two(X="X", Y="Y")
        fc_out = net.fc(X=out, W="w")
        net.complete_add_op()
        self.assertTrue(isinstance(fc_out, core.Variable))
        self.assertEqual(
            '''Op(plain_net), inputs:(@EMPTY@, X, Y, w), outputs:(@TEMP@fc@0, add_two@OUT@0, fc@OUT@1).
    Op(add_two), inputs:(X, Y), outputs:(add_two@OUT@0).
    Op(fc), inputs:(add_two@OUT@0, w, @EMPTY@), outputs:(fc@OUT@1, @TEMP@fc@0).
        Op(mul), inputs:(add_two@OUT@0, w), outputs:(@TEMP@fc@0).
        Op(sigmoid), inputs:(@TEMP@fc@0), outputs:(fc@OUT@1).
''', str(net))

        net2 = Network()
        tmp = net2.add_two(X="X", Y="Y")
        self.assertTrue(isinstance(tmp, core.Variable))
        net2.complete_add_op()
        self.assertEqual(
            '''Op(plain_net), inputs:(X, Y), outputs:(add_two@OUT@2).
    Op(add_two), inputs:(X, Y), outputs:(add_two@OUT@2).
''', str(net2))


if __name__ == '__main__':
    unittest.main()
