__all__ = ['Sigmoid', 'Tanh']


class Activation(object):
    def __init__(self, type, attrs=None):
        self.type = type
        if attrs is None:
            attrs = {}
        self.attrs = attrs


class Sigmoid(Activation):
    def __init__(self):
        super(Sigmoid, self).__init__("sigmoid")


class Tanh(Activation):
    def __init__(self):
        super(Tanh, self).__init__("tanh")
