class Op(object):
    def __init__(self):
        self.outputs = []

    def __call__(self, inputs):
        self.inputs = inputs


class Layer(Op):
    def __init__(self):
        self.outputs = []

    def __call__(self, inputs):
        self.inputs = inputs

