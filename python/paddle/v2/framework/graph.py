import paddle.v2.framework.core as core
import collections
import numpy as np

__all__ = ['Block', 'Variable', 'Program', 'Operator']


class Variable(object):
    def __init__(self, block, name=None, shape=None, dtype=None,
                 lod_level=None):
        self.block = block

        if name is None:
            name = Variable._unique_var_name_()
        try:
            self.proto = self.block.proto.var(name)
            is_new_var = False
        except core.EnforceNotMet:
            self.proto = self.block.proto.new_var(name)
            is_new_var = True

        if shape is not None:
            if is_new_var:
                self.proto.set_shape(shape)
            else:
                old_shape = self.shape
                shape = tuple(shape)
                if shape != old_shape:
                    raise ValueError(
                        "Variable {0} has been created before. the previous "
                        "shape is {1}; the new shape is {2}. They are not "
                        "matched.".format(self.name, old_shape, shape))
        if dtype is not None:
            if not isinstance(dtype, core.DataType):
                dtype = Variable._convert_np_dtype_to_dtype_(dtype)
            if is_new_var:
                self.proto.set_data_type(dtype)
            else:
                old_dtype = self.data_type()
                if dtype != old_shape:
                    raise ValueError("Variable {0} has been created before. "
                                     "The previous data type is {1}; the new "
                                     "data type is {2}. They are not "
                                     "matched.".format(self.name, old_dtype,
                                                       dtype))

        if lod_level is not None:
            if is_new_var:
                self.proto.set_lod_level(lod_level)
            else:
                if lod_level != self.lod_level:
                    raise ValueError("Variable {0} has been created before. "
                                     "The previous lod_level is {1}; the new "
                                     "lod_level is {2}. They are not "
                                     "matched".format(self.name, self.lod_level,
                                                      lod_level))

        self.block.vars[name] = self
        self.op = None

    @property
    def name(self):
        return self.proto.name()

    @property
    def shape(self):
        # convert to tuple, make it as same as numpy API.
        return tuple(self.proto.shape())

    @property
    def data_type(self):
        return self.proto.data_type()

    @property
    def lod_level(self):
        return self.proto.lod_level()

    @staticmethod
    def _unique_var_name_():
        uid = core.unique_integer()  # unique during whole process.
        return "_generated_var_%d" % uid

    @staticmethod
    def _convert_np_dtype_to_dtype_(np_dtype):
        dtype = np.dtype(np_dtype)
        if dtype == np.float32:
            return core.DataType.FP32
        elif dtype == np.float64:
            return core.DataType.FP64
        elif dtype == np.float16:
            return core.DataType.FP16
        elif dtype == np.int32:
            return core.DataType.INT32
        elif dtype == np.int16:
            return core.DataType.INT16
        elif dtype == np.int64:
            return core.DataType.INT64
        elif dtype == np.bool:
            return core.DataType.BOOL
        else:
            raise ValueError("Not supported numpy dtype " + str(dtype))


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
            # TODO.
            pass
        if inputs is not None:
            # TODO
            pass
        if outputs is not None:
            # TODO
            pass
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
