import paddle.v2.framework.core as core
from paddle.v2.framework.op import Operator
from variable import Variable


class Op(object):
    '''
    Operator wrapper for core operators.
    '''
    def __init__(self, type):
        self.type = type
        self.inputs = {}
        self.outputs = {}

    def __call__(self, *args, **kwargs):
        self._prepare_inputs(args, kwargs)
        self._prepare_outputs()
        self._extract_str_args_for_op()
        self._create_op()

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

    def _prepare_outputs(self):
        for out in Operator.get_op_output_names(self.type):
            name = "%s-%d" % (out, Var.count)
            var = Var(name)
            self.outputs[name] = var

    def _extract_str_args_for_op(self):
        self.str_arg = {}
        for k, v in self.inputs.items():
            self.str_arg[k] = v.name if isinstance(v, Variable) else v

        for k, v in self.outputs.items():
            self.str_arg[k] = v.name

    def _create_op(self, str_arg):
        self.op = Operator(self.type, **str_arg)

    def __hash__(self):
        return hash("%s-%s" % (self.type, self.str_arg))

    def __repr__(self):
        return "<Op {type}: {args}>".format(
            type=self.type,
            args=' '.join('%s->%s' % (k, v) for k, v in self.str_arg.items))
