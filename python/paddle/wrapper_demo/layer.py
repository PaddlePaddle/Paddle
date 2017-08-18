from var import Var
from op import add_two, mul


class Layer(object):
    def __init__(self):
        self.name = 'layer'


class FC(layer):
    def __init__(self, name=None):
        self.type = "fc"

    def __call__(self, input, size):
        assert isinstance(input, Var)
        self.input = input
        mul_res = mul(input, self.W)
        return mul_res

    def _create_param(self):
        self.W = Var(shape=self.input.shape)


class data(layer):
    pass


fc = FC()
