import paddle.v2.framework.core as core
import paddle.v2.framework.proto.framework_pb2 as framework_pb2
import collections

__all__ = ['Block', 'Variable', 'Program', 'Operator']


def get_all_op_protos():
    """
    Get all registered op proto from PaddlePaddle C++ end.
    :return: A list of registered OpProto.
    """
    protostrs = core.get_all_op_protos()
    ret_values = []
    for pbstr in protostrs:
        op_proto = framework_pb2.OpProto.FromString(str(pbstr))
        ret_values.append(op_proto)
    return ret_values


class OpProtoHolder(object):
    @classmethod
    def instance(cls):
        if not hasattr(cls, '_instance'):
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        assert not hasattr(
            self.__class__,
            '_instance'), 'Please use `instance()` to get OpProtoHolder opject!'
        op_protos = get_all_op_protos()
        self.op_proto_map = {}
        for proto in op_protos:
            self.op_proto_map[proto.type] = proto

    def get_op_proto(self, type):
        assert type in self.op_proto_map, "Operator with type \"%s\" has not been registered." % type
        return self.op_proto_map[type]


class Variable(object):
    def __init__(self, block, name=None, shape=None, dtype=None,
                 lod_level=None):
        self.block = block

        if name is None:
            name = Variable._unique_var_name_()
        self.desc = self.block.desc.new_var(name)

        if shape is not None:
            self.desc.set_shape(shape)

        if dtype is not None:
            # TODO(yuyang18): Convert dtype from numpy.dtype
            self.desc.set_data_type(dtype)

        if lod_level is not None:
            # TODO(yuyang18): set_lod_level is not defined.
            self.desc.set_lod_level(lod_level)

        self.block.vars[name] = self
        self.op = None

    # TODO(yuyang18): Get methods

    @staticmethod
    def _unique_var_name_():
        uid = core.unique_integer()  # unique during whole process.
        return "_generated_var_%d" % uid


class Operator(object):
    def __init__(self, block, desc, type, inputs=None, outputs=None,
                 attrs=None):
        self.block = block
        self.desc = desc
        self.proto = OpProtoHolder.instance().get_op_proto(type)
        self.desc.set_type(type)

        if inputs is not None:
            for in_proto in self.proto.inputs:
                in_argus = inputs[in_proto.name]
                if not isinstance(in_argus, list):
                    in_argus = [in_argus]
                if not in_proto.duplicable and len(in_argus) > 1:
                    raise ValueError(
                        "Input %s expects only one input, but %d are given." %
                        (in_proto.name, len(in_argus)))
                in_argu_names = []
                for argu in in_argus:
                    in_argu_names.append(argu.name())
                self.desc.set_input(in_proto.name, in_argu_names)

        if outputs is not None:
            for out_proto in self.proto.outputs:
                out_argus = outputs[out_proto.name]
                if not isinstance(out_argus, list):
                    out_argus = [out_argus]
                if not out_proto.duplicable and len(out_argus) > 1:
                    raise ValueError(
                        "Output %s expects only one output, but %d are given." %
                        (out_proto.name, len(out_argus)))
                out_argu_names = []
                for argu in out_argus:
                    out_argu_names.append(argu.name())
                self.desc.set_output(out_proto.name, out_argu_names)

        if attrs is not None:
            for attr in self.proto.attrs:
                attr_name = attr.name
                if not attr_name in attrs:
                    continue
                if not isinstance(attrs[attr_name], Block):
                    self.desc.set_attr(attr_name, attrs[attr_name])
                else:
                    self.desc.set_block_attr(attr_name, attrs[attr_name].desc)

    @property
    def type(self):
        return self.desc.type()

    def input(self, name):
        return self.desc.input(name)

    @property
    def input_names(self):
        return self.desc.input_names()

    def output(self, name):
        return self.desc.output(name)

    @property
    def output_names(self):
        return self.desc.output_names()

    def has_attr(self, name):
        return self.desc.has_attr(name)

    def attr_type(self, name):
        return self.desc.attr_type(name)

    @property
    def attr_names(self):
        return self.desc.attr_names()

    def attr(self, name):
        return self.desc.attr(name)

    def block_attr(self, name):
        return self.desc.block_attr(name)


class Block(object):
    def __init__(self, program, idx):
        self.desc = program.desc.block(idx)
        self.vars = dict()  # var_name --> var
        self.ops = collections.deque()  # operator list
        self.program = program

    @property
    def parent_idx(self):
        return self.desc.parent

    @property
    def idx(self):
        return self.desc.id

    def create_var(self, *args, **kwargs):
        return Variable(self, *args, **kwargs)

    def append_op(self, *args, **kwargs):
        op_desc = self.desc.append_op()
        op = Operator(self, op_desc, *args, **kwargs)
        self.ops.append(op)
        return op

    def prepend_op(self, *args, **kwargs):
        op_desc = self.desc.prepend_op()
        op = Operator(self, op_desc, *args, **kwargs)
        self.ops.appendleft(op)
        return op


class Program(object):
    @classmethod
    def instance(cls):
        # From https://stackoverflow.com/questions/8212053
        # Making Program as a Singleton class.
        if not hasattr(cls, '_instance'):
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        assert not hasattr(self.__class__,
                           '_instance'), 'Do not call constructor directly!'
        self.desc = core.ProgramDesc.instance()
        self.blocks = [Block(self, 0)]
        self.current_block_idx = 0

    def global_block(self):
        return self.blocks[0]

    def current_block(self):
        return self.blocks[self.current_block_idx]

    def create_block(self):
        new_block_idx = len(self.blocks)
        self.desc.append_block(self.current_block().desc)
        self.current_block_idx = new_block_idx
        self.blocks.append(Block(self, self.current_block_idx))
        return self.current_block()

    def rollback(self):
        self.current_block_idx = self.current_block().parent_idx


# program is a global instance.
g_program = Program.instance()
