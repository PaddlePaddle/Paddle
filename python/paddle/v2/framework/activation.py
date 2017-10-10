__all__ = ['Sigmoid', 'Tanh']


class ActivationBase(object):
    def __init__(self, type, attrs=None):
        self.type = type
        if attrs is None:
            attrs = {}
        self.attrs = attrs


class Sigmoid(ActivationBase):
    def __init__(self):
        super(Sigmoid, self).__init__("sigmoid")


class Tanh(ActivationBase):
    def __init__(self):
        super(Tanh, self).__init__("tanh")
