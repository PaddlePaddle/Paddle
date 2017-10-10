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
            sefl.op_proto_map[proto.type] = proto

    def get_op_proto(self, type):
        assert type in self.op_proto_map, "Operator with type \"%s\" has not been registered." % type
        return self.op_proto_map[type]


class Variable(object):
    def __init__(self, block, name=None, shape=None, dtype=None,
                 lod_level=None):
        self.block = block

        if name is None:
            name = Variable._unique_var_name_()
        self.proto = self.block.proto.new_var(name)

        if shape is not None:
            self.proto.set_shape(shape)

        if dtype is not None:
            # TODO(yuyang18): Convert dtype from numpy.dtype
            self.proto.set_data_type(dtype)

        if lod_level is not None:
            # TODO(yuyang18): set_lod_level is not defined.
            self.proto.set_lod_level(lod_level)

        self.block.vars[name] = self
        self.op = None

    # TODO(yuyang18): Get methods

    @staticmethod
    def _unique_var_name_():
        uid = core.unique_integer()  # unique during whole process.
        return "_generated_var_%d" % uid


class Operator(object):
    def __init__(self,
                 block,
                 proto,
                 type=None,
                 inputs=None,
                 outputs=None,
                 attrs=None):
        self.block = block
        self.proto = proto
        if type is not None:
            self.proto.set_type(type)
        if inputs is not None:
            for k, v in inputs.iteritems():
                self.proto.set_input(k, v)
        if outputs is not None:
            for k, v in outputs.iteritems():
                self.proto.set_output(k, v)
        if attrs is not None:
            # TODO
            pass

        # TODO: Getters


class Block(object):
    def __init__(self, program, idx):
        self.proto = program.proto.block(idx)
        self.vars = dict()  # var_name --> var
        self.ops = collections.deque()  # operator list
        self.program = program

    @property
    def parent_idx(self):
        return self.proto.parent

    @property
    def idx(self):
        return self.proto.id

    def create_var(self, *args, **kwargs):
        return Variable(self, *args, **kwargs)

    def append_op(self, *args, **kwargs):
        op_proto = self.proto.append_op()
        op = Operator(self, op_proto, *args, **kwargs)
        self.ops.append(op)
        return op

    def prepend_op(self, *args, **kwargs):
        op_proto = self.proto.prepend_op()
        op = Operator(self, op_proto, *args, **kwargs)
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
        self.proto = core.ProgramDesc.instance()
        self.blocks = [Block(self, 0)]
        self.current_block_idx = 0

    def global_block(self):
        return self.blocks[0]

    def current_block(self):
        return self.blocks[self.current_block_idx]

    def create_block(self):
        new_block_idx = len(self.blocks)
        self.proto.append_block(self.current_block().proto)
        self.current_block_idx = new_block_idx
        self.blocks.append(Block(self, self.current_block_idx))
        return self.current_block()

    def rollback(self):
        self.current_block_idx = self.current_block().parent_idx


# program is a global instance.
g_program = Program.instance()
