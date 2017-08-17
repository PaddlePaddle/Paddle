import logging
import paddle.v2.framework.core as core
from paddle.v2.framework.op import Operator, RecurrentOp
'''
Variable acts as the inputs and outputs' arguments of a op/layer,
same with the role of tensor in TF or pytorch.
'''
g_scope = core.Scope()
g_device = core.CPUPlace()


class Var(object):
    '''
    variable
    '''
    count = 0
    name_set = set()

    def __init__(self,
                 name=None,
                 shape=[],
                 data=None,
                 scope=None,
                 device=None,
                 is_parameter=False):
        if name is None:
            name = "var-%d" % self.count
            self.count += 1
        assert name not in self.name_set, "var name %s duplicate with others"
        self.name = name
        # assert shape or data, "either shape or data should be set, or paddle cannot create the var"
        self.shape = shape
        if shape:
            self.shape = shape if shape else data.shape
        self.data = data
        self.scope = scope if scope else g_scope
        self.device = device if device else g_device

        self._create_var()

    def val(self):
        return np.array(self._tensor)

    def __str__(self):
        return '<Var %s>' % self.name

    def __repr__(self):
        return self.__str__()

    def _create_var(self):
        self._var = self.scope.new_var(self.name)
        if self.shape:
            self._tensor = self._var.get_tensor()
            self._tensor.set_dims(self.shape)
        if self.data:
            self._tensor.set(data, self.device)


class PVar(Var):
    '''
    paramter variable.
    '''

    def __init__(self, name=None, shape=[], data=None, scope=None, device=None):
        Var.__init__(self, name, shape, data, scope, device, is_parameter=True)


if __name__ == '__main__':
    var = Var("var0", (10, 12, 31))
    print var
