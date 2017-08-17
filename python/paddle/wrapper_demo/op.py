'''
implementation of all the operators' python wrapper
'''
import paddle.v2.framework.core as core
from paddle.v2.framework.op import Operator

from var import Var, PVar


class Op(object):
    '''
    NOTE both op's inputs and outputs are Var
    usage:

       out0, out1 = op(in0, in1, attr=attr0)
    '''

    def __init__(self, type):
        self.type = type

    def __call__(self, *args, **kwargs):
        '''
        get inputs and output outputs
        '''
        input_and_attr = self._prepare_inputs(args, kwargs)
        output = self._create_outputs()
        str_arg = {}
        # extract string name from Var
        for k, v in input_and_attr.items():
            str_arg[k] = v.name if isinstance(v, Var) or isinstance(v,
                                                                    PVar) else v
        for k, v in output.items():
            str_arg[k] = v.name

        self.op = Operator(self.type, **str_arg)

    def _create_outputs(self):
        out_var = {}
        for out in Operator.get_op_output_names(self.type):
            name = "%s-%d" % (out, Var.count)
            var = Var(name)
            out_var[name] = var
        return out_var

    def _prepare_inputs(self, args, kwargs):
        arg = {}
        for idx, input_name in enumerate(
                Operator.get_op_input_names(self.type)):
            print 'input_name', input_name
            if idx > len(args):
                arg[input_name] = kwargs[input_name]
                continue
            arg[input_name] = args[idx]
        for k, v in kwargs.items():
            arg[k] = v
        return arg

    def __str__(self):
        return "<Op %s>" % self.type


add_two = Op("add_two")
mul = Op("mul")

if __name__ == '__main__':
    in0 = Var(shape=[10, 20])
    in1 = Var(shape=[10, 20])
    out = add_two(in0, in1)
