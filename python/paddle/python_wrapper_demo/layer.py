class Layer(object):
    def __init__(self, type, *args, **kwargs):
        self.inputs = {}
        self.outputs = {}

    def __hash__(self):
        raise NotImplemented

    def run(self):
        raise NotImplemented
