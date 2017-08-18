'''
implementation of all the operators' python wrapper
'''
import paddle.v2.framework.core as core
from paddle.v2.framework.op import Operator
from session import g_session

from var import Var, PVar


class Op(object):
    '''
    NOTE both op's inputs and outputs are Var
    usage:

       out0, out1 = op(in0, in1, attr=attr0)
    '''

    def __init__(self, type):
        self.type = type
        # input vars
        self.inputs = {}
        # output vars
        self.outputs = {}

    def __call__(self, *args, **kwargs):
        '''
        get inputs and output outputs
        '''
        self._prepare_inputs(args, kwargs)
        self._prepare_outputs()
        str_arg = {}
        # extract string name from Var
        for k, v in self.inputs.items():
            str_arg[k] = v.name if isinstance(v, Var) or isinstance(v,
                                                                    PVar) else v
        for k, v in self.outputs.items():
            str_arg[k] = v.name

        self.op = Operator(self.type, **str_arg)
        self.hash_str = self._gen_hash(str_arg)

        g_session.add_op(self)

    def _prepare_outputs(self):
        for out in Operator.get_op_output_names(self.type):
            name = "%s-%d" % (out, Var.count)
            var = Var(name)
            self.outputs[name] = var

    def _prepare_inputs(self, args, kwargs):
        for idx, input_name in enumerate(
                Operator.get_op_input_names(self.type)):
            print 'input_name', input_name
            if idx > len(args):
                self.inputs[input_name] = kwargs[input_name]
                continue
            self.inputs[input_name] = args[idx]
        for k, v in kwargs.items():
            self.inputs[k] = v

    def _gen_hash(self, str_arg):
        '''
        generate a hashkey for this op
        '''
        return hash("%s-%s" % (self.type, str_arg))

    def __str__(self):
        return "<Op {name}:\nInputs:{inputs}\nOutputs:{outputs}\n>".format(
            name=self.hash_str,
            inputs='\n -'.join(repr(v) for v in self.inputs),
            outputs='\n -'.join(repr(v) for v in self.outputs), )

    def __repr__(self):
        return "<Op %s %d>" % (self.type, self.hash_str)


sgd = Op("sgd")
add_two = Op("add_two")
mul = Op("mul")
mean = Op("mean")
mul = Op("mul")
rowwise_add = Op("rowwise_add")
sigmoid = Op("sigmoid")
softmax = Op("softmax")
gaussian = Op("gaussian")
cross_entropy = Op("cross_entropy")
fill_zeros_like = Op("fill_zeros_like")
uniform_random = Op("uniform_random")

if __name__ == '__main__':
    in0 = Var(shape=[10, 20])
    in1 = Var(shape=[10, 20])
    out = add_two(in0, in1)
