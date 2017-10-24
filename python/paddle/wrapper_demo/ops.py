class Op(object):
    def __init__(self, type):
        self.type = type

    def __call__(self, inputs):
        '''
        inputs: list of Variable
        outputs: list of Variable

        returns:
            output varialbes

        # NOTE all the output variables are not parameters, so they are Variable(is_param=False).
        '''
        self.inputs = inputs
        self.outputs = self._create_output_variables()
        self._core_op = self._create_core_op()
        self._infer_shape()

    def _create_output_variables(self):
        pass

    def _create_core_op(self):
        pass

    def _infer_shape(self):
        pass

    def run(self):
        pass
