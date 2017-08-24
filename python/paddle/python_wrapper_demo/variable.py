from namespace import Namespace
from common import g_scope


class Variable(object):
    '''
    Variable is the data type of Operator's inputs and outputs.
    '''
    counter = 0
    __varset__ = set()

    def __init__(self, name=None, shape=[], is_param=True, scope=g_scope):
        '''
        name: str
        shape: list of int
        is_param: bool
            whether this variable is a parameter
        '''
        self.name = self._gen_unique_name(name)
        assert shape, "shape of Variable should be set"
        self.shape = shape
        self.is_param = is_param

    def val(self):
        '''
        get python value from this Variable.
        '''
        return self._tensor.as_numpy()

    def __repr__(self):
        return "<Var %s>" % self.name

    def _create_core_variable(self):
        self._core_var = self.scope.new_var(self.name)
        self._tensor = self._core_var.get_tensor()
        self._tensor.set_dims(self.shape)

    def _gen_unique_name(self, name=None):
        if not name:
            name = "var-%d" % Variable.counter
            Variable += 1
            name = Namespace.gen_name(name)
        else:
            name = Namespace.gen_name(name)

        assert name not in Variable.__varset__, "Variable name [%s] duplicate" % name
        Variable.__varset__.add(name)
        return name
