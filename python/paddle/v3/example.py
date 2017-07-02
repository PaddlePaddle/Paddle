import numpy as np

from net import Net
from ops import *
from scope import Scope


def example():
    """Tests.
    """
    scope = Scope("test")
    net = Net("test")

    scope.create_var("X")

    X = np.random.randn(2, 3).astype(np.float32)
    scope.feed_var("X", X)

    mul_op = xx_op(inputs=["X"], outputs=["Y"], attrs=[])
    net.add_op(mul_op)
    net.add_gradient_ops()
    net.optimize(type="adam", lr=0.01)

    for i in range(10):
        net.run(scope)

    w = scope.get_var("W")
    b = scope.get_var("B")

