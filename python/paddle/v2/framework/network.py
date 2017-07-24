import paddle.v2.framework.core as core
from paddle.v2.framework.create_op_creation_methods import op_creations
from default_scope_funcs import create_var, get_var, get_cur_scope


class NetworkFunctor(object):
    def __init__(self, func, net):
        self.func = func
        self.net = net

    def __call__(self, **kwargs):
        inputs = self.func.all_input_args
        for ipt in inputs:
            if ipt in kwargs:
                var = kwargs[ipt]
                if isinstance(var, basestring):
                    var_name = var
                    var = create_var(var)
                    self.net.var_name_map[var] = var_name
                if not isinstance(var, core.Variable):
                    raise TypeError(
                        "Input of op creation must be string or variable")

                kwargs[ipt] = self.net.var_name_map[var]

        notemp_outputs = self.func.all_not_temp_output_args

        for name in notemp_outputs:
            if name not in kwargs:
                kwargs[
                    name] = self.func.__name__ + "@OUT@%d" % self.net.generate_idx
                self.net.generate_idx += 1

        outputs = self.func.all_output_args
        for opt in outputs:
            if opt in kwargs:
                var = kwargs[opt]
                if isinstance(var, basestring):
                    var_name = var
                    var = create_var(var)
                    self.net.var_name_map[var] = var_name
                if not isinstance(var, core.Variable):
                    raise TypeError(
                        "Output of op creation must be string or variable")
                kwargs[opt] = self.net.var_name_map[var]

        op = self.func(**kwargs)

        self.net.net.add_op(op)

        lst = [get_var(kwargs[opt]) for opt in notemp_outputs]
        if len(lst) == 1:
            return lst[0]
        elif len(lst) == 0:
            return None
        else:
            return lst


class Network(object):
    def __init__(self):
        self.net = core.Net.create()
        funcs = (func_name for func_name in dir(op_creations)
                 if not func_name.startswith("__"))
        self.generate_idx = 0
        self.var_name_map = dict()

        for func_name in funcs:
            func = getattr(op_creations, func_name)
            impl = NetworkFunctor(func, self)
            setattr(self, func_name, impl.__call__)
        self.__complete_add_op__ = False

    def infer_shape(self):
        self.net.infer_shape(get_cur_scope())

    def __str__(self):
        return str(self.net)


if __name__ == '__main__':
    net = Network()
    out = net.add_two(X="a", Y="b")
    fc_out = net.fc(X=out, W="fc.w", b="fc.b", activation="softmax")

    print str(net)
