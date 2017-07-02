from lib_loader import _C_LIB
from ops import Op


class Net(object):
    """
    Net hold all Operators and run with scope. Scope is used to fetch Variable.
    """
    def __init__(self, name):
        self._net = _C_LIB.GetOrCreateNet(name)

    def add_op(self, op):
        assert isinstance(op, Op)
        self._net.AddOp(op)

    def proto(self):
        return self._net.ToProto()

    def add_gradient_ops(self):
        """
        add
        :return:
        """
        self._net.AddGradientOps()

    def optimize(self, type, lr):
        """
        add optimize on in net
        :param type:
        :param lr:
        :return:
        """
        self._net.Optimize(type, lr)


    def run(self, scope):
        """
        Run this net in a given scope, the needed variable should all have been
        initialized in this scope.l
        :param scope:
        :return:
        """
        self._net.run(scope)
