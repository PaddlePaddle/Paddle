from paddle.v2.framework.network import Network
import unittest


class TestNetwork(unittest.TestCase):
    def test_directly_invoke_op(self):
        net = Network()
        out = net.add_two(X="A", Y="B")
        out = net.mul(X=out, Y="D")
        net2 = Network()
        net2.add_two(X=out, Y="E")
        net2.complete_add_op()
        net.add_op(net2)
        net.complete_add_op()
        # TODO(yuyang18): Since the output of op is generated. It should write
        # a regex to match net::DebugString()
        print str(net)

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
