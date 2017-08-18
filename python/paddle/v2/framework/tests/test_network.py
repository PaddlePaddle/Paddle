from paddle.v2.framework.network import Network
import unittest


class TestNetwork(unittest.TestCase):
    def test_add_op(self):
        net = Network()
        out = net.add_op("add_two", X="A", Y="B")
        out = net.add_op("mul", X=out, Y="D")
        net2 = Network()
        net2.add_op("add_two", X=out, Y="E")
        net2.complete_add_op()
        net.add_op(net2)
        net.complete_add_op()
        print str(net)


if __name__ == '__main__':
    unittest.main()
