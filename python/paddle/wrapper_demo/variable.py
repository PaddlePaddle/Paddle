from block import block, global_block


class Variable(object):
    '''
    Unified type for inputs and outputs.
    '''
    namespace = '' # namespace of this variable to store in a scope
    last_namespace = ''
    counter = 0    # to help make unique name

    def __init__(self, data=None, name=None, shape=[], is_param=True, block=None):
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
        unique_key = "var-%d" % Variable.counter
        Variable.counter += 1
        if Variable.namespace:
            unique_key = Variable.namespace + '/' + unique_key
        return unique_key

    def _create_core_var(self):
        '''
        create a var in current block
        '''
        pass


def namespace_begin(namespace):
    if namespace:
        Variable.last_namespace = Variable.namespace
        Variable.namespace += '/' + namespace

def namespace_end():
    Variable.namespace = Variable.last_namespace
