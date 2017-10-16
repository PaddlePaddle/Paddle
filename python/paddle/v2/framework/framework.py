import paddle.v2.framework.core as core
import paddle.v2.framework.proto.framework_pb2 as framework_pb2
import collections
import numpy as np
import copy

__all__ = ['Block', 'Variable', 'Program', 'Operator']


class Variable(object):
    def __init__(self,
                 block,
                 type=core.VarDesc.VarType.LOD_TENSOR,
                 name=None,
                 shape=None,
                 dtype=None,
                 lod_level=None,
                 **kwargs):
        self.block = block

        if name is None:
            name = Variable._unique_var_name_()
        is_new_var = False
        self.desc = self.block.desc.find_var(name)

        if self.desc is None:
            self.desc = self.block.desc.var(name)
            is_new_var = True

        if is_new_var:
            self.desc.set_type(type)
        elif self.desc.type() != type:
            raise ValueError("Variable {0} has been created before. The "
                             "previous type is {1}; the new type is {2}. They"
                             " are not matched".format(self.name,
                                                       self.desc.type(), type))

        if shape is not None:
            if is_new_var:
                self.desc.set_shape(shape)
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
                self.desc.set_data_type(dtype)
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
                self.desc.set_lod_level(lod_level)
            else:
                if lod_level != self.lod_level:
                    raise ValueError("Variable {0} has been created before. "
                                     "The previous lod_level is {1}; the new "
                                     "lod_level is {2}. They are not "
                                     "matched".format(self.name, self.lod_level,
                                                      lod_level))
        self.block.vars[name] = self
        self.op = None

    def __str__(self):
        protostr = self.desc.serialize_to_string()
        proto = framework_pb2.VarDesc.FromString(str(protostr))
        return proto.__str__()

    __repr__ = __str__

    @property
    def name(self):
        return self.desc.name()

    @property
    def shape(self):
        # convert to tuple, make it as same as numpy API.
        return tuple(self.desc.shape())

    @property
    def data_type(self):
        return self.desc.data_type()

    @property
    def lod_level(self):
        return self.desc.lod_level()

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
        if type not in self.op_proto_map:
            raise ValueError("Operator \"%s\" has not been registered." % type)
        return self.op_proto_map[type]


class Operator(object):
    def __init__(self,
                 block,
                 desc,
                 type=None,
                 inputs=None,
                 outputs=None,
                 attrs=None):
        self.block = block
        self.desc = desc
        if len(self.desc.type()) != 0:
            return
        if type is None:
            raise ValueError(
                "`type` to initilized an Operator can not be None.")
        self.desc.set_type(type)
        proto = OpProtoHolder.instance().get_op_proto(type)

        if inputs is not None:
            given = set()
            need = set()
            for n in inputs:
                given.add(n)
            for m in proto.inputs:
                need.add(m.name)
            if not given == need:
                raise ValueError(
                    "Incorrect setting for input(s) of operator \"%s\". Need: [%s] Given: [%s]"
                    % (type, ", ".join(str(e) for e in need), ", ".join(
                        str(e) for e in given)))

            for in_proto in proto.inputs:
                in_argus = inputs[in_proto.name]
                if not isinstance(in_argus, list):
                    in_argus = [in_argus]
                if not in_proto.duplicable and len(in_argus) > 1:
                    raise ValueError(
                        "Input %s expects only one input, but %d are given." %
                        (in_proto.name, len(in_argus)))
                in_argu_names = []
                for argu in in_argus:
                    in_argu_names.append(argu.name)
                self.desc.set_input(in_proto.name, in_argu_names)

        if outputs is not None:
            given = set()
            need = set()
            for n in outputs:
                given.add(n)
            for m in proto.outputs:
                need.add(m.name)
            if not given == need:
                raise ValueError(
                    "Incorrect setting for output(s) of operator \"%s\". Need: [%s] Given: [%s]"
                    % (type, ", ".join(str(e) for e in need), ", ".join(
                        str(e) for e in given)))

            for out_proto in proto.outputs:
                out_argus = outputs[out_proto.name]
                if not isinstance(out_argus, list):
                    out_argus = [out_argus]
                if not out_proto.duplicable and len(out_argus) > 1:
                    raise ValueError(
                        "Output %s expects only one output, but %d are given." %
                        (out_proto.name, len(out_argus)))
                out_argu_names = []
                for argu in out_argus:
                    out_argu_names.append(argu.name)
                    argu.op = self
                self.desc.set_output(out_proto.name, out_argu_names)

        if attrs is not None:
            for attr in proto.attrs:
                attr_name = attr.name
                if not attr_name in attrs:
                    continue
                if not isinstance(attrs[attr_name], Block):
                    self.desc.set_attr(attr_name, attrs[attr_name])
                else:
                    self.desc.set_block_attr(attr_name, attrs[attr_name].desc)

        self.desc.check_attrs()
        self.desc.infer_shape(self.block.desc)

    def __str__(self):
        protostr = self.desc.serialize_to_string()
        proto = framework_pb2.OpDesc.FromString(str(protostr))
        return proto.__str__()

    __repr__ = __str__

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

    def __str__(self):
        protostr = self.desc.serialize_to_string()
        proto = framework_pb2.BlockDesc.FromString(str(protostr))
        return proto.__str__()

    __repr__ = __str__

    @property
    def parent_idx(self):
        return self.desc.parent

    @property
    def idx(self):
        return self.desc.id

    def create_var(self, *args, **kwargs):
        return Variable(self, *args, **kwargs)

    def has_var(self, name):
        return name in self.vars

    def create_parameter(self, *args, **kwargs):
        global_block = self.program.global_block()
        return Parameter(global_block, *args, **kwargs)

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

    def sync_with_cpp(self):
        # sync variables from cpp
        for var in self.desc.all_vars():
            if not self.has_var(var.name()):
                self.create_var(name=var.name(), desc=var, type=var.type())

        # sync operators from cpp
        ops_in_cpp = self.desc.all_ops()
        first_op_in_python = self.ops[0].desc
        last_op_in_python = self.ops[len(self.ops) - 1].desc
        start_index = None
        end_index = None
        for index in range(len(ops_in_cpp)):
            if first_op_in_python == ops_in_cpp[index]:
                start_index = index
            if last_op_in_python == ops_in_cpp[index]:
                end_index = index
        assert start_index is not None
        assert end_index is not None
        assert start_index <= end_index

        # sync ops append to the head of cpp_ops
        for index in range((start_index - 1 - 1), -1, -1):
            op_desc = ops_in_cpp[index]
            op = Operator(self, op_desc)
            self.ops.appendleft(op)

        # sync ops append to the end of cpp_ops
        for index in range((end_index + 1), len(ops_in_cpp)):
            op_desc = ops_in_cpp[index]
            op = Operator(self, op_desc)
            self.ops.append(op)

        assert len(self.ops) == len(ops_in_cpp)
        for index in range(len(self.ops)):
            assert self.ops[index].desc == ops_in_cpp[index]


class Program(object):
    @classmethod
    def instance(cls):
        # From https://stackoverflow.com/questions/8212053
        # Making Program as a Singleton class.
        if not hasattr(cls, '_instance'):
            cls._instance = cls()
        return cls._instance

    def __init__(self, desc=None):
        if desc is None:
            desc = core.ProgramDesc.instance()
        self.desc = desc
        self.blocks = [Block(self, 0)]
        self.current_block_idx = 0

    def __str__(self):
        protostr = self.desc.serialize_to_string()
        proto = framework_pb2.ProgramDesc.FromString(str(protostr))
        return proto.__str__()

    __repr__ = __str__

    def global_block(self):
        return self.blocks[0]

    def current_block(self):
        return self.blocks[self.current_block_idx]

    def append_backward(self, target, no_grad_set):
        assert isinstance(target, Variable)
        param_to_grad_info = self.desc.append_backward(target.desc, no_grad_set)
        self.sync_with_cpp()
        return param_to_grad_info

    def create_block(self):
        new_block_idx = len(self.blocks)
        self.desc.append_block(self.current_block().desc)
        self.current_block_idx = new_block_idx
        self.blocks.append(Block(self, self.current_block_idx))
        return self.current_block()

    def rollback(self):
        self.current_block_idx = self.current_block().parent_idx

    def sync_with_cpp(self):
        for block_idx in range(len(self.blocks), self.desc.num_blocks()):
            self.blocks.append(Block(self, block_idx))
        for block in self.blocks:
            block.sync_with_cpp()


class Parameter(Variable):
    def __init__(self, block, shape, dtype, **kwargs):
        if shape is None or dtype is None:
            raise ValueError("Parameter must set shape and dtype")
        if len(shape) == 0:
            raise ValueError("Parameter shape cannot be empty")

        for each in shape:
            if each < 0:
                raise ValueError("Parameter shape should not be related with "
                                 "batch-size")
        Variable.__init__(self, block, shape=shape, dtype=dtype, **kwargs)
        self.trainable = kwargs.get('trainable', True)
        self.init_attr = kwargs.get('initialize_attr', {
            'type': 'uniform_random',
            'min': -1.0,
            'max': 1.0
        })

        self.optimize_attr = kwargs.get('optimize_attr', {'learning_rate': 1.0})
        self._append_initialize_ops_()

    def _append_initialize_ops_(self):
        attr = self.init_attr
        op_type = attr.pop('type', None)
        block = self.block
        assert isinstance(block, Block)
        shape = self.shape
        attr['dims'] = shape
        attr['data_type'] = int(self.data_type)
        op = block.prepend_op(
            type=op_type, inputs=None, outputs={'Out': [self]}, attrs=attr)
        self.op = op


# program is a global instance.
g_program = Program.instance()
