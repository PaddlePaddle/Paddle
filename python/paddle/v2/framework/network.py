import paddle.v2.framework.core as core
from paddle.v2.framework.create_op_creation_methods import op_creations
from default_scope_funcs import new_var, find_var, get_cur_scope

__all__ = ['Network']  # Only expose Network


class NetworkFunctor(object):
    """
    Network Op Creation Function. Used internally in this module.
    It convert string input to Variable. If it is not created before, just 
    create in scope.
    
    It is a functor object. means the instances are callable.
    
    :param func: The op creation function which generated in Python.
    :param net: The Network instance.
    """

    def __init__(self, func, net):
        self.func = func
        self.net = net

    def __call__(self, *args, **kwargs):
        if len(args) != 0:
            raise ValueError("Paddle must use keyword argument")
        inputs = self.func.all_input_args
        for ipt in inputs:
            if ipt in kwargs:
                var = kwargs[ipt]
                if isinstance(var, basestring):
                    tmp = new_var(var)
                    self.net.var_names[tmp] = var
                    var = tmp

                if not isinstance(var, core.Variable):
                    raise TypeError(
                        "Input of op creation must be string or variable")

                kwargs[ipt] = self.net.var_names[var]

        notemp_outputs = self.func.all_not_temp_output_args

        for name in notemp_outputs:
            if name not in kwargs:
                kwargs[
                    name] = self.func.__name__ + "@OUT@%d" % core.unique_integer(
                    )

        outputs = self.func.all_output_args
        for opt in outputs:
            if opt in kwargs:
                var = kwargs[opt]
                if isinstance(var, basestring):
                    tmp = new_var(var)
                    self.net.var_names[tmp] = var
                    var = tmp

                if not isinstance(var, core.Variable):
                    raise TypeError(
                        "Output of op creation must be string or variable")
                kwargs[opt] = self.net.var_names[var]

        op = self.func(**kwargs)

        self.net.net.add_op(op)

        lst = [find_var(kwargs[opt]) for opt in notemp_outputs]
        if len(lst) == 1:
            return lst[0]
        elif len(lst) == 0:
            return None
        else:
            return lst


class Network(object):
    """
    The network concept. It avoid user to manually create operator, create 
    variable, and combine them into a Net. Just use Network.xxx can create the
    operator, create variables in default scope, and add them into `self.net`.
    
    For example:
    
    ..  code-block: python
    
        net = Network()
        out = net.add_two(X="a", Y="b")
        fc_out = net.fc(X="out", W="fc.w")
        
        net.run(...)
    """

    def __init__(self):
        self.net = core.Net.create()
        funcs = (func_name for func_name in dir(op_creations)
                 if not func_name.startswith("__"))
        self.var_names = dict()

        # TODO(yuyang18): This code can work, but do not generate a good
        # docstring, try to give a better way generate function in runtime
        # later.
        for func_name in funcs:
            func = getattr(op_creations, func_name)
            impl = NetworkFunctor(func, self)
            setattr(self, func_name, impl.__call__)
        self.__complete_add_op__ = False

    def infer_shape(self):
        self.complete_add_op()
        self.net.infer_shape(get_cur_scope())

    def run(self, device_context):
        self.complete_add_op()
        self.net.run(get_cur_scope(), device_context)

    def __str__(self):
        return str(self.net)

    def complete_add_op(self):
        if not self.__complete_add_op__:
            self.net.complete_add_op()
            self.__complete_add_op__ = True


if __name__ == '__main__':
    net = Network()
    out = net.add_two(X="a", Y="b")
    fc_out = net.fc(X=out, W="fc.w", b="fc.b", activation="softmax")
    net.complete_add_op()
    print net
