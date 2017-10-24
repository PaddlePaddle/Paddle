import paddle.v2 as pd
import paddle.v2.framework.core as core
from util import Graph

__all__ = [
    "Variable",
]


class Block(object):
    '''
    Block is an implementation concept like a programming code's scope between two curly braces.

    It stores variables and ops in a local code scope.

    This should be implemented in C++, here is a demonstration of its interfaces.
    '''
    def __init__(self):
        self.ops = []
        self.local_vars = []
        # a local core_scope, and all the local_vars are stored in this scope.
        self._core_scope = core.Scope()

    def append(self, op):
        self.ops.append(op)

    def scope(self):
        return self._core_scope

    def execute(self):
        '''
        Execute a block.
        '''
        for op in self.ops:
            op.run()


# A global block as default, all the variables or operators defined are stored in global_block.
#
# for example:
#
#   a = 1
#
#   int main() {
#     b = 2
#   }
#
# `a` will stored in global block.
#
global_block = Block()


class Variable(object):
    '''
    Unified type for inputs and outputs.
    '''

    def __init__(self, data=None, name=None, shape=[], is_param=True, block=g_block):
        '''
        data: numpy data or initializer op.
        name: name of this variable
        shape: shape of the tensor
        is_param: whether this variable is a parameter
           in short, all the user-defined `Variable`s are parameters, all the outputs of operators are not parameters.
        '''
        self.is_param = is_param

        if data is None:
            self.data = pd.zeros_op()

        self.name = name if name else self._make_unique_name()

        assert shape
        self.shape = shape

        self._core_var = self._create_core_var()

    def assign(self, py_data):
        '''
        assign `Variable` with a python data.
        '''
        pass

    def py_value(self):
        '''
        return a value in python.
        '''
        pass

    def _make_unique_name(self):
        '''
        make a unique name across all the scopes.
        '''
        pass

    def _create_core_var(self):
        '''
        create a var in current block's local scope.
        '''
        pass


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


class Layer(object):
    pass


def eval(fetches=[], block=global_block):
    '''
    fetches: list
        variables that user want to fetches the latest value.

    It will help to run operators help to eval specific target variables. In details,
    it will build a `Block` and put related operators in it, execute and return
    all the python value of variables in the fetches list.

    For example:
        def data_provider(path):
            # ...
            yield batch

        # data slots
        image = pd.data('image')
        label = pd.data('label')

        # model config
        fc_out = pd.fc(image, size=128)
        prediction = pd.softmax(fc_out, size=10)
        cost = pd.cross_entropy(label, prediction)
        loss = pd.SGDOptimizer([cost])

        # train
        for batch_id, batch in enumerate(data_provider("./data.txt")):
            _loss, _cost = pd.eval([loss, eval],
                                   feed_data={'image': xxx, 'label':xxx})
            print 'batch %d cost %f" % (batch_id, _cost)

        # infer
        _prediction = pd.eval([prediction], feed_data={'image': xxx})
    '''
    graph = Graph(block.ops)
    local_block = Block()
    for op in graph.reverse_dfs(endpoints=fetches):
        local_block.append(op)
    local_block.execute()
    return [var.py_value() for var in fetches]


def variable_initializer(vars=[]):
    '''
    run variable's initializer ops and return
    '''
    pass
