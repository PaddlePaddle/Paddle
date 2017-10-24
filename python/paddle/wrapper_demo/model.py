class Model(object):
    __opset__ = {}

    def __init__(self):
        self.ops = []

    def run(self):
        for op in self.ops:
            op.run()

    def __getattribute__(self, type):
        op = Model.__opset__[type](type)
        self.ops.append(op)
        return op
