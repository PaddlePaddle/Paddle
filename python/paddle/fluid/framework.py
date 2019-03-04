#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import collections
from collections import defaultdict
from collections import Iterable
import contextlib
from .wrapped_decorator import signature_safe_contextmanager
import os
import re
import traceback
import six

import numpy as np
import subprocess

from .. import compat as cpt
from .proto import framework_pb2
try:
    if os.name == 'nt':
        import sys
        third_lib_path = os.path.abspath(os.path.dirname(
            __file__)) + os.sep + '..' + os.sep + 'libs'
        os.environ['path'] += ';' + third_lib_path
        sys.path.append(third_lib_path)

    from . import core
except ImportError as e:
    if os.name == 'nt':
        executable_path = os.path.abspath(os.path.dirname(sys.executable))
        raise ImportError(
            """NOTE: You may need to run \"set PATH=%s;%%PATH%%\"
        if you encounters \"DLL load failed\" errors. If you have python
        installed in other directory, replace \"%s\" with your own
        directory. The original error is: \n %s""" %
            (executable_path, executable_path, cpt.get_exception_message(e)))
    else:
        raise ImportError(
            """NOTE: You may need to run \"export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH\"
        if you encounters \"libmkldnn.so not found\" errors. If you have python
        installed in other directory, replace \"/usr/local/lib\" with your own
        directory. The original error is: \n""" + cpt.get_exception_message(e))
except Exception as e:
    raise e
from . import unique_name

__all__ = [
    'Program',
    'default_startup_program',
    'default_main_program',
    'program_guard',
    'name_scope',
]

EMPTY_VAR_NAME = core.kEmptyVarName()
TEMP_VAR_NAME = core.kTempVarName()
GRAD_VAR_SUFFIX = core.kGradVarSuffix()
ZERO_VAR_SUFFIX = core.kZeroVarSuffix()
CONTROL_DEP_VAR_PREFIX = core.kControlDepVarName()

_imperative_tracer_ = None
_imperative_current_expected_place_ = None


def _in_imperative_mode():
    return _imperative_tracer_ is not None


def _imperative_tracer():
    return _imperative_tracer_


def _current_expected_place():
    return _imperative_current_expected_place_


class NameScope(object):
    def __init__(self, name="", parent=None):
        self._children = dict()
        self._name = name
        self._parent = parent

    def child(self, prefix):
        if prefix not in self._children:
            new_child = NameScope(prefix, self)
            self._children[prefix] = [new_child]
        else:
            new_child = NameScope(prefix + "_%d" % len(self._children[prefix]),
                                  self)
            self._children[prefix].append(new_child)
        return new_child

    def parent(self):
        return self._parent

    def name(self):
        return self._name


_name_scope = NameScope()


@signature_safe_contextmanager
def name_scope(prefix=None):
    """
    Generate hierarchical name prefix for the operators.

    Note: This should only used for debugging and visualization purpose.
    Don't use it for serious analysis such as graph/program transformations.

    Args:
        prefix(str): prefix.

    Examples:
        .. code-block:: python

          with name_scope("encoder"):
             ...
          with name_scope("decoder"):
             ...
          with name_scope("attention"):
             ...
    """
    # TODO(panyx0718): Only [0-9a-z].
    assert prefix, "namescope prefix cannot be empty."
    global _name_scope
    _name_scope = _name_scope.child(prefix)
    yield
    _name_scope = _name_scope.parent()


def _full_name_scope():
    global _name_scope
    scope = _name_scope
    name = ""
    while scope:
        name = scope.name() + "/" + name
        scope = scope.parent()
    return name


def generate_control_dev_var_name():
    import random
    return CONTROL_DEP_VAR_PREFIX + "@" + str(random.random())


def grad_var_name(var_name):
    """
    Returns:
        str: gradient name for a certain var name
    """
    return var_name + GRAD_VAR_SUFFIX


def convert_np_dtype_to_dtype_(np_dtype):
    """
    Convert the data type in numpy to the data type in Paddle

    Args:
        np_dtype(np.dtype): the data type in numpy.

    Returns:
        core.VarDesc.VarType: the data type in Paddle.

    """
    dtype = np.dtype(np_dtype)
    if dtype == np.float32:
        return core.VarDesc.VarType.FP32
    elif dtype == np.float64:
        return core.VarDesc.VarType.FP64
    elif dtype == np.float16:
        return core.VarDesc.VarType.FP16
    elif dtype == np.int32:
        return core.VarDesc.VarType.INT32
    elif dtype == np.int16:
        return core.VarDesc.VarType.INT16
    elif dtype == np.int64:
        return core.VarDesc.VarType.INT64
    elif dtype == np.bool:
        return core.VarDesc.VarType.BOOL
    elif dtype == np.uint16:
        return core.VarDesc.VarType.INT16
    elif dtype == np.uint8:
        return core.VarDesc.VarType.UINT8
    elif dtype == np.int8:
        return core.VarDesc.VarType.INT8
    else:
        raise ValueError("Not supported numpy dtype %s" % dtype)


def dtype_is_floating(dtype):
    """
    Check the data type is floating or not.
    Args:
        dtype(np.dtype|core.VarDesc.VarType): data type.
            Could be numpy format or Paddle format

    Returns(bool): True if data type is a float value

    """
    if not isinstance(dtype, core.VarDesc.VarType):
        dtype = convert_np_dtype_to_dtype_(dtype)

    return dtype in [
        core.VarDesc.VarType.FP16, core.VarDesc.VarType.FP32,
        core.VarDesc.VarType.FP64
    ]


def _debug_string_(proto, throw_on_error=True):
    """
    Get the debug string of a protobuf message. The message could be not
    initialized.
    Args:
        proto(google.protobuf.message.Message): The protobuf message
        throw_on_error(bool): True if raise an error when the protobuf message
            is not initialized.

    Returns(str): The debug string of the protobuf message

    """
    error_fields = list()
    if not proto.IsInitialized(error_fields) and throw_on_error:
        raise ValueError("{0} are not initialized.\nThe message is {1}:\n".
                         format(error_fields, proto))
    return proto.__str__()


class Variable(object):
    """
    In Fluid, every input and output of an operator is a variable. In most
    cases, variables are used for holding different kinds of data or training
    labels. A variable belongs to a block. All variable has its own name and
    two variables in different blocks could have the same name.

    There are many kinds of variables. Each kind of them has its own attributes
    and usages. Please reference the framework.proto for details.

    Most of a Variable's member variables can be setted to be None. It mean
    it is not available or will be specified later.

    Args:
        block(Block): The block that the variable belongs to.
        type(core.VarDesc.VarType): Variable type. Please reference the
            framework.proto for details.
        name(str|None): The name of the variable. If setted None, it will be
            generated automatically. Default: None
        shape(tuple|list|None): The shape of the variable. -1 means the batch size.
            Some kinds of variable do not contain shape, just set it to None.
            Default: None
        dtype(np.dtype|core.VarDesc.VarType|str|None): The data type of variable.
            Default: None
        lod_level (int|None): The level of lod tensor. 0 means it is not a time
            series data.
            Default: None
        capacity (int|None): The capacity of Channel variable. Ignored for other
            types. Default: None
        persistable (bool|None): True if the variable is persistable. A persistable
            variable will not be deleted after an iteration ending. Defaults: None.
        error_clip (BaseErrorClipAttr|None): The error clip attributes of the
            corresponding gradient variable. Default: None
        stop_gradient (bool): True if the variable will stop to calculate its
            gradients when backward. Default: False.
        is_data (bool): True if the variable is an input data. Default: False

    Notes:
        The constructor of Variable should not be invoked directly. Please
        use `Block.create_var` to create a variable.

    Examples:
        .. code-block:: python

            cur_program = Program()
            cur_block = cur_program.current_block()
            new_variable = cur_block.create_var(name="X",
                                                shape=[-1, 23, 48],
                                                dtype='float32')
    """

    def __init__(self,
                 block,
                 type=core.VarDesc.VarType.LOD_TENSOR,
                 name=None,
                 shape=None,
                 dtype=None,
                 lod_level=None,
                 capacity=None,
                 persistable=None,
                 error_clip=None,
                 stop_gradient=False,
                 is_data=False,
                 **kwargs):
        self.block = block
        self.error_clip = error_clip

        if name is None:
            name = unique_name.generate('_generated_var')
        is_new_var = False
        name = cpt.to_text(name)
        self.desc = self.block.desc.find_var(cpt.to_bytes(name))

        if self.desc is None:
            self.desc = self.block.desc.var(cpt.to_bytes(name))
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
            if not isinstance(dtype, core.VarDesc.VarType):
                dtype = convert_np_dtype_to_dtype_(dtype)
            if is_new_var:
                self.desc.set_dtype(dtype)
            else:
                old_dtype = self.dtype
                if dtype != old_dtype:
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
        if persistable is not None:
            if is_new_var:
                self.desc.set_persistable(persistable)
            else:
                if persistable != self.persistable:
                    raise ValueError(
                        "Variable {0} has been created before."
                        "The previous persistable is {1}; the new "
                        "persistable is {2}. They are not matched".format(
                            self.name, self.persistable, persistable))

        if capacity is not None:
            if is_new_var:
                self.desc.set_capacity(capacity)
            else:
                # TODO(abhinavarora) : Compare with set capacity once,
                # get_capacity is implemented
                pass

        if _in_imperative_mode():
            # record vars in tracer rather than blocks
            self._ivar = kwargs.get("ivar", None)
            if not self._ivar:
                self._ivar = core.VarBase(stop_gradient)
            self._ivar.desc = self.desc
            self._ivar.block = block.desc
            self._ivar.name = name
            self._ivar.persistable = persistable
            if persistable:
                self.block.vars[name] = self
        else:
            self.block.vars[name] = self
        self.op = None
        self.stop_gradient = stop_gradient
        self.is_data = is_data

    def _numpy(self):
        new_ivar = self._ivar._copy_to(core.CPUPlace(), True)
        return np.array(new_ivar.value().get_tensor())

    def _backward(self):
        self._ivar._run_backward()

    def _gradient(self):
        return np.array(self._ivar._grad_value())

    def _clear_gradient(self):
        self._ivar._clear_gradient()

    def __str__(self):
        return self.to_string(True)

    def to_string(self, throw_on_error, with_details=False):
        """
        Get debug string.

        Args:
            throw_on_error(bool): True if raise an exception when self is
                not initialized.
            with_details(bool): more details about variables and parameters
                (e.g. trainable, optimize_attr, ...) will be printed when
                with_details is True. Default False;

        Returns:
            str: The debug string.
        """
        assert isinstance(throw_on_error, bool) and isinstance(with_details,
                                                               bool)
        protostr = self.desc.serialize_to_string()
        proto = framework_pb2.VarDesc.FromString(six.binary_type(protostr))
        res_str = _debug_string_(proto, throw_on_error)
        if with_details:
            additional_attr = ("error_clip", "stop_gradient")
            for attr_name in additional_attr:
                res_str += "%s: %s\n" % (
                    attr_name, six.binary_type(getattr(self, attr_name)))
        return res_str

    __repr__ = __str__

    def _set_desc(self, input):
        """
        Set the variable description.

        Args:
            input(core.VarDesc): The new VarDesc.

        Returns:
            None
        """
        self.desc = input

    @property
    def _stop_gradient(self):
        if _in_imperative_mode():
            return self._ivar.stop_gradient
        else:
            return self.stop_gradient

    @_stop_gradient.setter
    def _stop_gradient(self, s):
        if _in_imperative_mode():
            self._ivar.stop_gradient = s
        self.stop_gradient = s

    @property
    def persistable(self):
        return self.desc.persistable()

    @persistable.setter
    def persistable(self, p):
        self.desc.set_persistable(p)

    @property
    def name(self):
        return cpt.to_text(self.desc.name())

    @name.setter
    def name(self, new_name):
        self.desc.set_name(new_name)

    @property
    def shape(self):
        # convert to tuple, make it as same as numpy API.
        return tuple(self.desc.shape())

    @property
    def dtype(self):
        return self.desc.dtype()

    @property
    def lod_level(self):
        return self.desc.lod_level()

    @property
    def type(self):
        return self.desc.type()

    def _set_error_clip(self, error_clip):
        """
        Set the error_clip.

        Args:
            error_clip(BaseErrorClipAttr) : The new error_clip.

        Returns:
            None
        """
        self.error_clip = error_clip


def get_all_op_protos():
    """
    Get all registered op proto from PaddlePaddle C++ end.

    Returns:
       list: list of OpProto.
    """
    protostrs = core.get_all_op_protos()
    ret_values = []
    for pbstr in protostrs:
        op_proto = framework_pb2.OpProto.FromString(six.binary_type(pbstr))
        ret_values.append(op_proto)
    return ret_values


class OpProtoHolder(object):
    """
    A global variable to hold all OpProtos from C++ as a map
    """

    @classmethod
    def instance(cls):
        if not hasattr(cls, '_instance'):
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        assert not hasattr(
            self.__class__,
            '_instance'), 'Please use `instance()` to get OpProtoHolder object!'
        op_protos = get_all_op_protos()
        self.op_proto_map = {}
        for proto in op_protos:
            self.op_proto_map[proto.type] = proto

    def get_op_proto(self, type):
        """
        Get OpProto by a type string.
        Args:
            type(str): The type that operator registered in C++ side.

        Returns(framework_pb2.OpProto): The OpProto

        """
        if type not in self.op_proto_map:
            raise ValueError("Operator \"%s\" has not been registered." % type)
        return self.op_proto_map[type]

    @staticmethod
    def generated_op_attr_names():
        return {
            core.op_proto_and_checker_maker.kOpRoleAttrName(),
            core.op_proto_and_checker_maker.kOpRoleVarAttrName(),
            core.op_proto_and_checker_maker.kOpNameScopeAttrName(),
            core.op_proto_and_checker_maker.kOpCreationCallstackAttrName()
        }


class Operator(object):
    """
    In Fluid, all the operation are represented by Operator, and Operator
    is regarded as a build in an instruction of a Block. Users can use the
    build in instructions to describe their neural network.

    Args:
        block(Block): The block has the current operator.
        desc(core.OpDesc): The protobuf description of Operator.
        type(str): The type of operator. Default None.
        inputs(dict): The input of this Operator. it is a dictionary, for every
            element, key is the input parameter name, and value is a list of
            variables. Default None.
        outputs(dict): The output of this Operator. it is a dictionary, for
            every element, key is the input parameter name, and value is a list
            of variables. Default None.
        attrs(dict): The attributes of this Operator. it is a dictionary, for
            every element, key is attribute name, and value is the attribute value.
            The attribute type should be as same as the type registered in C++ side.
            Default None.

    Returns:
        Operator: The initialized Operator.

    Raises:
        ValueError: If the passed input, output and attrs doesn't match the
            initializing Operator's that registered in C++ side.

    Notes:
        The constructor of operator should not be invoked directly. Use
        Block.append_op or Block._prepend_op instead.

    Examples:
        .. code-block:: python

            cur_program = Program()
            cur_block = cur_program.current_block()
            # var1 += var2 + var3
            cur_block.append_op(type="sum",
                                inputs={"X": [var1, var2, var3]},
                                outputs={"Out": [var1]})
    """
    OP_WITHOUT_KERNEL_SET = {
        'feed', 'fetch', 'save', 'load', 'recurrent', 'go',
        'rnn_memory_helper_grad', 'conditional_block', 'while', 'send', 'recv',
        'listen_and_serv', 'save_combine', 'load_combine', 'ncclInit', 'select',
        'checkpoint_notify', 'gen_nccl_id'
    }

    def __init__(self,
                 block,
                 desc,
                 type=None,
                 inputs=None,
                 outputs=None,
                 attrs=None):
        self.block = block
        self.desc = desc
        # note: not add self.attrs here:
        # https://github.com/PaddlePaddle/Paddle/pull/12583#pullrequestreview-145093173
        op_attrs = attrs
        if op_attrs is None:
            op_attrs = dict()
        del attrs

        op_maker = core.op_proto_and_checker_maker

        if op_maker.kOpRoleAttrName() not in op_attrs:
            op_attrs[op_maker.kOpRoleAttrName()] = self.block.program.op_role

        role_var_name = op_maker.kOpRoleVarAttrName()
        if len(self.block.program.
               op_role_var) != 0 and role_var_name not in op_attrs:
            op_attrs[role_var_name] = self.block.program.op_role_var

        if role_var_name in op_attrs and len(op_attrs[role_var_name]) == 0:
            del op_attrs[role_var_name]

        if len(self.desc.type()) != 0:
            return
        if type is None:
            raise ValueError(
                "`type` to initilized an Operator can not be None.")
        else:
            callstack_var_name = op_maker.kOpCreationCallstackAttrName()
            op_attrs[callstack_var_name] = list(
                reversed(traceback.format_stack()))[1:]

        self.desc.set_type(type)
        proto = OpProtoHolder.instance().get_op_proto(type)

        namescope_var_name = op_maker.kOpNameScopeAttrName()
        op_attrs[namescope_var_name] = _full_name_scope()

        def find_name(var_list, name):
            for var_name in var_list:
                if var_list[var_name] is not None and var_name == name:
                    return True
            return False

        if inputs is not None:
            for in_proto in proto.inputs:
                found = find_name(inputs, in_proto.name)
                assert found or in_proto.dispensable, "Input {} not found".format(
                    in_proto.name)

                if found:
                    in_args = inputs[in_proto.name]
                    if not isinstance(in_args, list):
                        in_args = [in_args]
                    if not in_proto.duplicable and len(in_args) > 1:
                        raise ValueError(
                            "Input %s expects only one input, but %d are given."
                            % (in_proto.name, len(in_args)))
                    in_arg_names = []
                    for arg in in_args:
                        if isinstance(arg, six.string_types):
                            in_arg_names.append(arg)
                        elif isinstance(arg, six.binary_type):
                            in_arg_names.append(arg.decode())
                        else:
                            in_arg_names.append(cpt.to_text(arg.name))
                    self.desc.set_input(in_proto.name, in_arg_names)
                else:
                    self.desc.set_input(in_proto.name, [])

        if outputs is not None:
            for m in proto.outputs:
                if (m.name not in outputs) and m.dispensable:
                    continue
                if not ((m.name in outputs) or m.dispensable):
                    raise ValueError(
                        ("Incorrect setting for output(s) of "
                         "operator \"%s\", should set: [%s].") % (type, m.name))
            for out_proto in proto.outputs:
                if out_proto.name not in outputs:
                    continue
                out_args = outputs[out_proto.name]
                if not isinstance(out_args, list):
                    out_args = [out_args]
                if not out_proto.duplicable and len(out_args) > 1:
                    raise ValueError(
                        "Output %s expects only one output, but %d are given." %
                        (out_proto.name, len(out_args)))
                out_arg_names = []
                for arg in out_args:
                    out_arg_names.append(cpt.to_text(arg.name))
                    # TODO(minqiyang): could we remove variable's op in static mode?
                    if not _in_imperative_mode():
                        arg.op = self
                self.desc.set_output(out_proto.name, out_arg_names)

        if op_attrs is not None:
            if not isinstance(op_attrs, dict):
                raise TypeError("'attrs' should be a dict.")
            for attr in proto.attrs:
                attr_name = attr.name
                if (attr_name not in op_attrs) or (op_attrs[attr_name] is None):
                    continue
                attr_val = op_attrs[attr_name]
                self._update_desc_attr(attr_name, attr_val)

        self.desc.check_attrs()
        if self._has_kernel(type):
            self.desc.infer_var_type(self.block.desc)
            self.desc.infer_shape(self.block.desc)

        if _in_imperative_mode():
            self.iop = core.OpBase()
            self.iop.desc = self.desc

            self.inputs = defaultdict(list)
            if inputs is not None:
                for k, v in six.iteritems(inputs):
                    if isinstance(v, Variable):
                        self.inputs[k].append(v._ivar)
                    elif isinstance(v, list) or isinstance(v, tuple):
                        self.inputs[k].extend([var._ivar for var in v])

            self.outputs = defaultdict(list)
            if outputs is not None:
                for k, v in six.iteritems(outputs):
                    if isinstance(v, Variable):
                        self.outputs[k].append(v._ivar)
                    elif isinstance(v, list) or isinstance(v, tuple):
                        self.outputs[k].extend([var._ivar for var in v])

    def _has_kernel(self, op_type):
        return op_type not in self.OP_WITHOUT_KERNEL_SET

    def to_string(self, throw_on_error):
        """
        Get debug string.

        Args:
            throw_on_error(bool): Whether to raise exception if self is not
                initialized.

        Returns:
            str: The debug string.

        """
        protostr = self.desc.serialize_to_string()
        proto = framework_pb2.OpDesc.FromString(six.binary_type(protostr))
        return _debug_string_(proto, throw_on_error)

    def __str__(self):
        return self.to_string(True)

    __repr__ = __str__

    @property
    def type(self):
        return self.desc.type()

    def input(self, name):
        """
        Get the input arguments according to the input parameter name.

        Args:
            name(str): The input parameter name.

        Returns:
            list: return the list of argument names that associated with \
                the specific parameter name.
        """
        return self.desc.input(name)

    def _rename_input(self, old_name, new_name):
        """
        Rename the `old_name` to `new_name`.

        Args:
            old_name(str): The old name of the Operator's input.
            new_name(str): The new name of the Operator's input.

        Returns:
            None
        """
        self.desc._rename_input(old_name, new_name)

    def _rename_output(self, old_name, new_name):
        """
        Rename the `old_name` to `new_name`.

        Args:
            old_name(str): The old name of the Operator's output.
            new_name(str): The new name of the Operator's output.

        Returns:
            None
        """
        self.desc._rename_output(old_name, new_name)

    @property
    def input_names(self):
        return self.desc.input_names()

    @property
    def input_arg_names(self):
        return self.desc.input_arg_names()

    @property
    def output_arg_names(self):
        return self.desc.output_arg_names()

    def output(self, name):
        """
        Get output arguments by the output parameter name.

        Args:
            name(str): The output parameter name.

        Returns:
            list: return the list of argument names associated with \
                the specific parameter name.
        """
        return self.desc.output(name)

    @property
    def output_names(self):
        return self.desc.output_names()

    @property
    def idx(self):
        for i, op in enumerate(self.block.ops):
            if op == self:
                return i
        raise ValueError(
            "Can't find op itself in it's block. It could be a bug of Paddle.")

    def has_attr(self, name):
        """
        Whether this Operator has the attribute with name or not.

        Args:
            name(str): the attribute name.

        Returns:
            bool: True if has this attribute.

        """
        return self.desc.has_attr(name)

    def attr_type(self, name):
        """
        Get the type of attribute by attribute's name.

        Args:
            name(str): the attribute name.

        Returns:
            core.AttrType: the attribute type.
        """
        return self.desc.attr_type(name)

    def _set_attr(self, name, val):
        """
        Set the value of attribute by attribute's name.

        Args:
            name(str): the attribute name.
            val(bool|int|str|float|list): the value of the attribute.

        Raises:
            ValueError: If the type of value doesn't match with desc.attr_type(name).
        """
        self._update_desc_attr(name, val)

    def _update_desc_attr(self, name, val):
        """
        Update the value of desc's attribute by attribute's name.

        Args:
            name(str): the attribute name.
            val(bool|int|str|float|list): the value of the attribute.

        Raises:
            ValueError: If the type of value doesn't match with desc.attr_type(name).
        """
        if isinstance(val, Block):
            self.desc.set_block_attr(name, val.desc)
        elif isinstance(val, list) and val and all(
                isinstance(v, Block) for v in val):
            self.desc.set_blocks_attr(name, [v.desc for v in val])
        elif isinstance(val, core.BlockDesc) or \
                isinstance(val, core.ProgramDesc):
            self.desc.set_serialized_attr(name, val.serialize_to_string())
        else:
            self.desc._set_attr(name, val)

    @property
    def attr_names(self):
        return self.desc.attr_names()

    def attr(self, name):
        """
        Get the attribute by name.

        Args:
            name(str): the attribute name.

        Returns:
            bool|int|str|float|list: The attribute value. The return value
            can be any valid attribute type.
        """
        return self.desc.attr(name)

    def _block_attr_id(self, name):
        """
        Get the block attribute's id by name.

        Args:
            name(str): the attribute name.

        Returns:
            int: the block index.
        """
        return self.desc._block_attr_id(name)

    def _block_attr(self, name):
        """
        Get the block attribute  by name.

        Args:
            name(str): the attribute name.

        Returns:
            block: the block attribute.
        """

        id = self._block_attr_id(name)
        assert (id >= 0 and id < len(self.block.program.blocks))
        return self.block.program.blocks[id]

    def _blocks_attr(self, name):
        """
        Get the blocks attribute  by name.

        Args:
            name(str): the attribute name.

        Returns:
            list: list of the blocks attribute.
        """
        attrs = []
        for i in self._blocks_attr_ids(name):
            assert (i >= 0 and i < len(self.block.program.blocks))
            attrs.append(self.block.program.blocks[i])

        return attrs

    def _blocks_attr_ids(self, name):
        """
        Get the blocks attribute's ids by name.

        Args:
            name(str): the attribute name.

        Returns:
            list: list of the blocks ids.
        """

        return self.desc._blocks_attr_ids(name)

    def all_attrs(self):
        """
        Get the attribute dict.

        Returns:
            dict: The Operator's attribute dict, name->attr.
        """
        attr_names = self.attr_names
        attr_map = {}
        for n in attr_names:
            attr_type = self.desc.attr_type(n)
            if attr_type == core.AttrType.BLOCK:
                attr_map[n] = self._block_attr(n)
                continue

            if attr_type == core.AttrType.BLOCKS:
                attr_map[n] = self._blocks_attr(n)
                continue

            attr_map[n] = self.attr(n)

        return attr_map


class Block(object):
    """
    In Fluid, a Program is consistence of multi-Block, and Block stores
    VarDesc and OpDesc. In a specific Block, a VarDesc have a unique name.
    One block could have some child blocks, and child block's name scopes
    should inherit the parent's so that OpDesc in child block can reference
    a VarDesc that is stored in the parent block.
    Please reference the framework.proto for details.

    Args:
        program(Program): The Program that the Block belongs to.
        idx(int): The block's id in the Program.

    Notes:
        The constructor of Block should not be invoked directly. Please
        use `Program._create_block()` to create a block.

    Examples:
        .. code-block:: python

            cur_program = Program()
            cur_block = cur_program.current_block()
            var = cur_block.create_var(name="X",
                                       shape=[-1, 23, 48],
                                       dtype='float32')
            cur_block.append_op(type="abs",
                                inputs={"X": [var]},
                                outputs={"Out": [var]})
    """

    def __init__(self, program, idx):
        self.desc = program.desc.block(idx)
        self.vars = collections.OrderedDict()  # var_name --> var
        self.ops = list()  # operator list
        self.program = program
        self.removed_vars = collections.OrderedDict()

    def __str__(self):
        return self.to_string(True)

    def to_string(self, throw_on_error, with_details=False):
        """
        Get debug string.

        Args:
            throw_on_error(bool): raise exception when self is not initialized
                when throw_on_error is True.
            with_details(bool): more details about variables and parameters
                (e.g. trainable, optimize_attr, ...) will be printed when
                with_details is True. Default False.

        Returns:
            str: The debug string.
        """
        assert isinstance(throw_on_error, bool) and isinstance(with_details,
                                                               bool)
        if with_details:
            re_add_indent = re.compile(r"\n(.)")
            res_str = "blocks {\n  idx: %d\n  parent_idx: %d" % (
                self.idx, self.parent_idx)
            for var in list(self.vars.values()):
                res_str += "\n  vars {\n    %s  }" % re_add_indent.sub(
                    r"\n    \1", var.to_string(throw_on_error, with_details))
            for op in self.ops:
                res_str += "\n  ops {\n    %s  }" % re_add_indent.sub(
                    r"\n    \1", op.to_string(throw_on_error))
            res_str += "\n}"
        else:
            protostr = self.desc.serialize_to_string()
            proto = framework_pb2.BlockDesc.FromString(
                six.binary_type(protostr))
            res_str = _debug_string_(proto, throw_on_error)
        return res_str

    __repr__ = __str__

    @property
    def parent_idx(self):
        return self.desc.parent

    @property
    def forward_block_idx(self):
        return self.desc.get_forward_block_idx()

    def _set_forward_block_idx(self, idx):
        """
        Set the forward block Idx.

        Args:
            idx(int): the block index.

        Returns:
            None
        """
        self.desc._set_forward_block_idx(idx)

    @property
    def idx(self):
        return self.desc.id

    def var(self, name):
        """
        Get a Variable by name from this block.

        Args:
            name(str): the Variable's name.

        Raises:
            ValueError: The If input's type is not str, or this block
                doesn't have a Variable with the giving name.

        Returns:
            Variable: the Variable with the giving name.
        """
        if not isinstance(name, six.string_types):
            raise TypeError(
                "var require string as parameter, but get %s instead." %
                (type(name)))
        v = self.vars.get(name, None)
        if v is None:
            raise ValueError("var %s not in this block" % name)
        return v

    def _find_var_recursive(self, name):
        """
        Get a Variable by name from this block recursively.

        Args:
            name(str): the Variable's name.

        Returns:
            Variable: the Variable with the giving name. Or None if not found.
        """
        frontier = list()
        visited = set()

        frontier.append(self)

        prog = self.program

        while len(frontier) != 0:  # BFS
            cur = frontier[0]
            frontier = frontier[1:]

            if id(cur) in visited:
                continue

            if cur.has_var(name):
                return cur.var(name)

            if cur.parent_idx != -1:
                frontier.append(prog.block(cur.parent_idx))

            if cur.forward_block_idx != -1:
                frontier.append(prog.block(cur.forward_block_idx))

            visited.add(id(cur))
        return None

    def _var_recursive(self, name):
        """
        Get a Variable by name from this block recursively.

        Args:
            name(str): the Variable's name.

        Raises:
            ValueError: this block and this parent block doesn't
                have a Variable with the giving name.

        Returns:
            Variable: the Variable with the giving name.
        """
        var = self._find_var_recursive(name)
        if var:
            return var
        else:
            raise ValueError("Var {0} is not found recursively".format(name))

    def all_parameters(self):
        return list(self.iter_parameters())

    def iter_parameters(self):
        return (item[1] for item in six.iteritems(self.vars)
                if isinstance(item[1], Parameter))

    def create_var(self, *args, **kwargs):
        var = Variable(block=self, *args, **kwargs)
        if 'initializer' in kwargs:
            kwargs['initializer'](var, self)
        return var

    def has_var(self, name):
        return name in self.vars

    def _rename_var(self, name, new_name):
        """
        Rename variable in vars and ops' inputs and outputs

        Args:
            name(str): the name that need to be renamed.
            new_name(str): the name that need to rename to.

        Raises:
            ValueError: If this block doesn't have this the giving name,
                or the type of the var with the giving name is not Parameter
                or Variable.

        Returns:
            Variable: the Variable with the giving name.
        """
        name = cpt.to_text(name)
        new_name = cpt.to_text(new_name)

        if not self.has_var(name):
            raise ValueError("var %s is not in current block" % name)
        v = self.var(name)
        if type(v) == Parameter:
            var_type = "Parameter"
            stop_gradient = v.stop_gradient
            trainable = v.trainable
            optimize_attr = v.optimize_attr
            regularizer = v.regularizer
            gradient_clip_attr = v.gradient_clip_attr
            error_clip = v.error_clip
        elif type(v) == Variable:
            var_type = "Variable"
            error_clip = v.error_clip
            stop_gradient = v.stop_gradient
        else:
            raise ValueError("unsupported var type: %s", type(v))
        orig_var_type = v.type
        self.desc._rename_var(cpt.to_bytes(name), cpt.to_bytes(new_name))
        # NOTE: v is destroyed by C++ after calling _rename_var.
        d = self.desc.find_var(cpt.to_bytes(new_name))
        if var_type == "Parameter":
            var = Parameter(
                self,
                d.shape(),
                d.dtype(),
                type=orig_var_type,
                name=new_name,
                stop_gradient=stop_gradient,
                trainable=trainable,
                optimize_attr=optimize_attr,
                regularizer=regularizer,
                gradient_clip_attr=gradient_clip_attr,
                error_clip=error_clip)
        elif var_type == "Variable":
            var = Variable(
                self,
                type=orig_var_type,
                name=new_name,
                error_clip=error_clip,
                stop_gradient=stop_gradient)

        # rename the python side, _sync_with_cpp will only add
        # new vars/ops to python side.
        self.vars[new_name] = var
        del self.vars[name]
        self._sync_with_cpp()
        return var

    def _remove_var(self, name):
        self._sync_with_cpp()
        self.desc._remove_var(cpt.to_bytes(name))
        del self.vars[name]

    def create_parameter(self, *args, **kwargs):
        global_block = self.program.global_block()
        param = Parameter(global_block, *args, **kwargs)
        if 'initializer' in kwargs:

            def _is_inited_by(block, var):
                init_ops = []
                for op in block.ops:
                    if var.name in op.output_arg_names:
                        init_ops.append(op)
                return init_ops

            initializer = kwargs['initializer']
            init_ops = _is_inited_by(global_block, param)
            init_ops_len = len(init_ops)
            if init_ops_len > 1:
                raise RuntimeError("param " + param.name +
                                   " is inited by multiple init ops " + str(
                                       init_ops))
            elif init_ops_len == 1:
                #TODO already inited, do nothing, should log a warning
                pass
            else:
                initializer(param, self)
        return param

    def append_op(self, *args, **kwargs):
        """
        Appends a new Operator according to the giving arguments.

        Returns:
            Operator: the append Operator.
        """
        op_desc = self.desc.append_op()
        op = Operator(
            block=self,
            desc=op_desc,
            type=kwargs.get("type", None),
            inputs=kwargs.get("inputs", None),
            outputs=kwargs.get("outputs", None),
            attrs=kwargs.get("attrs", None))

        if _in_imperative_mode():
            # record ops in tracer rather than blocks
            #
            # TODO(minqiyang): add op stop_gradient support in static mode too.
            # currently, we only support stop_gradient in imperative mode.
            _imperative_tracer().trace_op(op,
                                          kwargs.get("stop_gradient", False))
        else:
            self.ops.append(op)

        return op

    def _insert_op(self, index, *args, **kwargs):
        """
        Insert a Operator according to the giving arguments.

        Args:
            index(int): the place that the operator to insert.

        Returns:
            Operator: the insert Operator.
        """
        self._sync_with_cpp()
        op_desc = self.desc._insert_op(index)
        op = Operator(block=self, desc=op_desc, *args, **kwargs)
        self.ops.insert(index, op)
        return op

    def _remove_op(self, index):
        """
        Remove the specific position operator.

        Args:
            index(int): the position that the operator to insert.

        Returns:
            None
        """
        self._sync_with_cpp()
        self.desc._remove_op(index, index + 1)
        del self.ops[index]

    def _slice_ops(self, start, end):
        """
        Return the Operator between start and end.

        Args:
            start(int): the start position.
            end(int): the end position.

        Returns:
            list: the Operators between start and end.
        """
        return self.ops[start:end]

    def _prepend_op(self, *args, **kwargs):
        op_desc = self.desc._prepend_op()
        op = Operator(
            self,
            op_desc,
            type=kwargs.get("type", None),
            inputs=kwargs.get("inputs", None),
            outputs=kwargs.get("outputs", None),
            attrs=kwargs.get("attrs", None))
        if _in_imperative_mode():
            _imperative_tracer().trace_op(op,
                                          kwargs.get("stop_gradient", False))
        else:
            self.ops.insert(0, op)
        return op

    def _sync_with_cpp(self):
        """
        Sync from the desc on the c++ end. This method is used to synchronize
        the c++ desc instance generated by backward.
        """
        # sync variables from cpp
        for var in self.desc.all_vars():
            if not self.has_var(var.name()):
                self.create_var(name=var.name(), desc=var, type=var.type())

        # sync variables removed from c++ end
        for var in list(self.vars.keys()):
            if not self.desc.find_var(cpt.to_bytes(var)):
                self.vars.pop(var)

        # sync operators from cpp
        ops_in_cpp = []
        for op_idx in range(0, self.desc.op_size()):
            ops_in_cpp.append(self.desc.op(op_idx))

        if len(self.ops) != 0:
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
        else:
            start_index = 0
            end_index = -1

        # sync ops append to the head of cpp_ops
        for index in range((start_index - 1 - 1), -1, -1):
            op_desc = ops_in_cpp[index]
            op = Operator(self, op_desc)
            self.ops.insert(0, op)

        # sync ops append to the end of cpp_ops
        for index in range((end_index + 1), len(ops_in_cpp)):
            op_desc = ops_in_cpp[index]
            op = Operator(self, op_desc)
            self.ops.append(op)

        # sync ops removed from c++ end
        if end_index != -1 and end_index < len(self.ops):
            ops_in_cpp_index = 0
            ops_in_python_index = 0
            while ops_in_python_index < len(
                    self.ops) and ops_in_cpp_index < len(ops_in_cpp):
                if self.ops[ops_in_python_index].desc != ops_in_cpp[
                        ops_in_cpp_index]:
                    del self.ops[ops_in_python_index]
                else:
                    ops_in_cpp_index += 1
                    ops_in_python_index += 1

        assert len(self.ops) == len(ops_in_cpp)
        for index in range(len(self.ops)):
            assert self.ops[index].desc == ops_in_cpp[index]

    def _copy_param_info_from(self, other):
        """
        Copy the information of parameters from the other block.

        Args:
            other(Block): the other block.

        Raises:
            ValueError: If type of input is not Block, or the `other` and this
                block is not in the same topology.

        Returns:
            None
        """
        if not isinstance(other, Block):
            raise TypeError(
                "_copy_param_info_from should be invoked with Block")
        for p in other.iter_parameters():
            assert isinstance(p, Parameter)
            v = self.vars.get(p.name, None)
            if v is None:
                raise ValueError("_copy_param_info_from should be invoked with "
                                 "same topology")
            assert isinstance(v, Variable)
            new_p = Parameter(
                block=self,
                shape=v.shape,
                dtype=v.dtype,
                type=v.type,
                lod_level=v.lod_level,
                stop_gradient=p.stop_gradient,
                trainable=p.trainable,
                optimize_attr=p.optimize_attr,
                regularizer=p.regularizer,
                gradient_clip_attr=p.gradient_clip_attr,
                error_clip=p.error_clip,
                name=v.name)
            self.vars[new_p.name] = new_p

    def _clone_variable(self, var):
        """
        Clone a variable into current block.

        Args:
            var: the variable to be cloned.

        Returns:
            Variable: the new  variable cloned from 'var' in current block.
        """
        assert isinstance(var, Variable)
        ret_var = None
        # make STEP_SCOPES var can be safely cloned.
        if var.type == core.VarDesc.VarType.STEP_SCOPES:
            ret_var = self.create_var(
                name=var.name, persistable=var.persistable, type=var.type)
        elif var.type == core.VarDesc.VarType.RAW:
            ret_var = self.create_var(
                name=var.name, persistable=var.persistable, type=var.type)
        elif var.type == core.VarDesc.VarType.SELECTED_ROWS:
            ret_var = self.create_var(
                name=var.name,
                shape=var.shape,
                dtype=var.dtype,
                type=var.type,
                persistable=True,
                is_data=var.is_data)
        else:
            ret_var = self.create_var(
                name=var.name,
                shape=var.shape,
                dtype=var.dtype,
                type=var.type,
                lod_level=var.lod_level,
                persistable=True,
                is_data=var.is_data)
        return ret_var


class IrNode(object):
    """
    Python IrNode. Beneath it is a core.Node, which is used for Ir Pass.
    """

    def __init__(self, node):
        """
        Construct an IrNode using core.Node.

        Args:
            node(core.Node): C++ Node.
        """
        assert isinstance(node,
                          core.Node), 'node must be the instance of core.Node.'
        self.node = node

    def name(self):
        """
        Return the node name.

        Returns:
            str: node name.
        """
        return self.node.name()

    def node_type(self):
        """
        Return the node type.

        Returns:
            core.Node.Type: node type(core.Node.Type.Operation or core.Node.Type.Variable).
        """
        return self.node.node_type()

    def var(self):
        """
        Return the node variable description.

        Returns:
            core.VarDesc: node variable description.
        """
        return self.node.var()

    def op(self):
        """
        Return the node operator description.

        Returns:
            core.OpDesc: node operator description.
        """
        return self.node.op()

    def id(self):
        """
        Return the node id.

        Returns:
            int: node id.
        """
        return self.node.id()

    def is_op(self):
        """
        If the node is an operator, then return true.

        Returns:
            bool: indicate whether the node is an operator.
        """
        return self.node.is_op()

    def is_var(self):
        """
        If the node is a variable, then return true.

        Returns:
            bool: indicate whether the node is a variable.
        """
        return self.node.is_var()

    def is_ctrl_var(self):
        """
        If the node is a control dependence variable, then return true.

        Returns:
            bool: indicate whether the node is a control dependence variable.
        """
        return self.node.is_ctrl_var()

    def clear_inputs(self):
        """
        Clear the node inputs. After executing the `clear_inputs` function,
        the node inputs will be empty.
        """
        self.node.clear_inputs()

    def remove_input_by_id(self, node_id):
        """
        Remove a node from inputs by the given node id.

        Args:
            node_id(int): the given node id.
        """
        self.node.remove_input(node_id)

    def remove_input(self, node):
        """
        Remove a node from inputs.

        Args:
            node(IrNode): the node being removed.
        """
        self.node.remove_input(node.node)

    def append_input(self, node):
        """
        Append a node in inputs.

        Args:
            node(IrNode): the node being appended.
        """
        self.node.append_input(node.node)

    def clear_outputs(self):
        """
        Clear the node outputs. After executing the `clear_outputs` function,
        the node outputs will be empty.
        """
        self.node.clear_outputs()

    def remove_output_by_id(self, node_id):
        """
        Remove a node from outputs by the given node id.

        Args:
            node_id(int): the given node id.
        """
        self.node.remove_output(node_id)

    def remove_output(self, node):
        """
        Remove a node from outputs.

        Args:
            node(IrNode): the node being removed.
        """
        self.node.remove_output(node.node)

    def append_output(self, node):
        """
        Append a node in outputs.

        Args:
            node(IrNode): the node being appended.
        """
        self.node.append_output(node.node)

    @property
    def inputs(self):
        """
        Return the node inputs.

        Returns:
            list(IrNode): node inputs wrapped by IrNode.
        """
        return [IrNode(n) for n in self.node.inputs]

    @property
    def outputs(self):
        """
        Return the node outputs.

        Returns:
            list(IrNode): node outputs wrapped by IrNode.
        """
        return [IrNode(n) for n in self.node.outputs]


class IrVarNode(IrNode):
    """
    Python IrVarNode. Beneath it is a core.Node, it inherits from IrNode.
    """

    def __init__(self, node):
        """
        Construct an IrVarNode using core.Node.

        Args:
            node(core.Node): C++ Node.
        """
        assert isinstance(node, core.Node) and node.is_var(), \
            'node must be the instance of core.Node and it must be a variable node.'
        super(IrVarNode, self).__init__(node)
        self.node = node

    def set_shape(self, shape):
        """
        Set the node variable shape.

        Args:
            shape(list): shape to be set.
        """
        assert self.node.var() is not None, \
            "The node variable description cannot be None."
        self.node.var().set_shape(shape)

    def persistable(self):
        """
        If the variable node is a persistable variable, then return true.

        Returns:
            bool: indicate whether the variable is persistable.
        """
        assert self.node.var() is not None, \
            "The node variable description cannot be None."
        return self.node.var().persistable()

    def type(self):
        """
        Return the variable type.

        Returns:
            core.VarDesc.VarType: the variable type.
        """
        assert self.node.var() is not None, \
            "The node variable description cannot be None."
        return self.node.var().type()

    def dtype(self):
        """
        Return the variable data type.

        Returns:
            core.VarDesc.VarType: the variable data type.
        """
        assert self.node.var() is not None, \
            "The node variable description cannot be None."
        return self.node.var().dtype()

    def shape(self):
        """
        Return the variable shape.

        Returns:
            list: the variable shape.
        """
        assert self.node.var() is not None, \
            "The node variable description cannot be None."
        return self.node.var().shape()

    @property
    def inputs(self):
        """
        Return the node inputs.

        Returns:
            list(IrOpNode): node inputs wrapped by IrOpNode.
        """
        return [IrOpNode(n) for n in self.node.inputs]

    @property
    def outputs(self):
        """
        Return the node outputs.

        Returns:
            list(IrOpNode): node outputs wrapped by IrOpNode.
        """
        return [IrOpNode(n) for n in self.node.outputs]


class IrOpNode(IrNode):
    """
    Python IrOpNode. Beneath it is a core.Node, it inherits from IrNode.
    """

    def __init__(self, node):
        """
        Construct an IrOpNode using core.Node.

        Args:
            node(core.Node): C++ Node.
        """
        assert isinstance(node, core.Node) and node.is_op(), \
            'node must be the instance of core.Node and it must be a operator node.'
        super(IrOpNode, self).__init__(node)
        self.node = node

    def rename_input(self, old_input_name, new_input_name):
        """
        Rename the input of this node.

        Args:
            old_input_name(str): the old input name.
            new_input_name(str): the new input name.
        """
        assert self.node.op() is not None, \
            "The node operator description cannot be None."
        self.node.op()._rename_input(old_input_name, new_input_name)

    def input(self, name):
        """
        Get the argument name list by the parameter name for input.

        Args:
            name(str): the parameter name.

        Returns:
            list(str): the argument name list.
        """
        assert self.node.op() is not None, \
            "The node operator description cannot be None."
        return self.node.op().input(name)

    def output(self, name):
        """
        Get the argument name list by the parameter name for output.

        Args:
            name(str): the parameter name.

        Returns:
            list(str): the argument name list.
        """
        assert self.node.op() is not None, \
            "The node operator description cannot be None."
        return self.node.op().output(name)

    def set_type(self, new_type):
        """
        Change the operator type into new type.

        Args:
            new_type(str): new operator type to be set.
        """
        assert self.node.op() is not None, \
            "The node operator description cannot be None."
        return self.node.op().set_type(new_type)

    def set_attr(self, name, val):
        """
        Set the value of attribute by attribute's name.

        Args:
            name(str): the attribute name.
            val(bool|int|str|float|list): the value of the attribute.
        """
        self._update_desc_attr(name, val)

    def _update_desc_attr(self, name, val):
        """
        Update the value of the op desc's attribute by attribute's name.
        """
        assert self.node.op() is not None, \
            "The node operator description cannot be None."
        desc = self.node.op()
        if isinstance(val, Block):
            desc.set_block_attr(name, val.desc)
        elif isinstance(val, list) and val and \
            all(isinstance(v, Block) for v in val):
            desc.set_blocks_attr(name, [v.desc for v in val])
        elif isinstance(val, core.BlockDesc) or \
            isinstance(val, core.ProgramDesc):
            desc.set_serialized_attr(name, val.serialize_to_string())
        else:
            desc._set_attr(name, val)

    @property
    def inputs(self):
        """
        Return the node inputs.

        Returns:
            list(IrVarNode): node inputs wrapped by IrVarNode.
        """
        return [IrVarNode(n) for n in self.node.inputs]

    @property
    def outputs(self):
        """
        Return the node outputs.

        Returns:
            list(IrVarNode): node outputs wrapped by IrVarNode.
        """
        return [IrVarNode(n) for n in self.node.outputs]


class IrGraph(object):
    """
    Python IrGraph. Beneath it is a core.Graph, which is used for
    creating a c++ Ir Pass Graph. An IrGraph is just a graph view of
    a Program. In an IrGraph, both Variables and Operators are graph
    nodes.
    """

    def __init__(self, graph, for_test=False):
        """
        Construct an IrGraph using core.Graph.

        Args:
            graph(core.Graph): C++ Graph.
            for_test(bool): True for the test graph and false for the train graph.
        """
        assert isinstance(
            graph, core.Graph), 'graph must be the instance of core.Graph.'
        self.graph = graph
        self._for_test = for_test

    def is_test(self):
        """
        If the graph is used for testing, the function returns true. Otherwise, returns false.
        """
        return self._for_test

    def all_nodes(self):
        """
        Return all nodes included in the graph as a set.
        """
        return {IrNode(node) for node in self.graph.nodes()}

    def all_var_nodes(self):
        """
        Return all variable nodes included in the graph as a set.
        """
        return {IrVarNode(node) for node in self.graph.nodes() if node.is_var()}

    def all_persistable_nodes(self):
        """
        Return all persistable variable nodes included in the graph as a set.
        """
        persistable_nodes = set()
        for node in self.graph.nodes():
            if node.is_var() and node.var() is not None and node.var(
            ).persistable():
                persistable_nodes.add(node)
        return {IrVarNode(p) for p in persistable_nodes}

    def all_op_nodes(self):
        """
        Return all operator nodes included in the graph as a set.
        """
        return {IrOpNode(node) for node in self.graph.nodes() if node.is_op()}

    def var_node(self, name):
        """
        Get a variable node by name from the graph.

        Args:
            name(str): the name of the variable node.

        Raises:
            ValueError: The If input's type is not str, or this graph
            doesn't have a variable with the giving name.

        Returns:
            IrVarNode: the variable node with the giving name.
        """
        if not isinstance(name, six.string_types):
            raise TypeError(
                "var require string as parameter, but get %s instead." %
                (type(name)))
        target_var_node = None
        var_nodes = self.all_var_nodes()
        for var_node in var_nodes:
            if var_node.name() == name:
                target_var_node = var_node
        if target_var_node is None:
            raise ValueError("var_node %s not in this graph" % name)
        return target_var_node

    def create_persistable_node(self, name, var_type, shape, var_dtype):
        """
        Create a persistable variable node in the graph. In IrGraph,
        it can not distinguish between persistable variables and parameters.

        Args:
            name(str): the name of the persistable variable node.
            vart_type(core.VarDesc.VarType): the type of the persistable variable node.
            shape(list): the shape of the persistable variable node.
            var_dtype(core.VarDesc.VarType): the data type of the persistable variable node.

        Returns:
            IrVarNode: the created persistable variable node.
        """
        var_desc = core.VarDesc(name)
        var_desc.set_type(var_type)
        var_desc.set_shape(shape)
        var_desc.set_dtype(var_dtype)
        var_desc.set_persistable(True)
        return IrVarNode(self.graph.create_var_node(var_desc))

    def create_var_node(self, name, var_type, shape, var_dtype):
        """
        Create a variable node in the graph. The created variable node is
        not persistable.

        Args:
            name(str): the name of the variable node.
            vart_type(core.VarDesc.VarType): the type of the variable node.
            shape(list): the shape of the variable node.
            var_dtype(core.VarDesc.VarType): the data type of the variable node.

        Returns:
            IrVarNode: the created variable node.
        """

        var_desc = core.VarDesc(name)
        var_desc.set_type(var_type)
        var_desc.set_shape(shape)
        var_desc.set_dtype(var_dtype)
        return IrVarNode(self.graph.create_var_node(var_desc))

    def create_var_node_from_desc(self, var_desc):
        """
        Create a variable node by using an existing VarDesc in the graph.
        Depend on the giving VarDesc, the created variable node may be persistable.

        Args:
            var_desc(core.VarDesc): the giving variable description.

        Returns:
            IrVarNode: the created variable node.
        """
        return IrVarNode(self.graph.create_var_node(var_desc))

    def create_op_node(self, op_type, attrs, inputs, outputs):
        """
        Create a operator node in the graph.

        Args:
            op_type(str): the type of the operator node.
            attrs(dict): the attributes of the operator node.
            inputs(dict): the inputs of the operator node.
            outputs(dict): the outpus of the operator node.

        Returns:
            IrOpNode: the created operator node.
        """
        op_desc = core.OpDesc()
        op_desc.set_type(op_type)
        for attr, value in six.iteritems(attrs):
            self._update_desc_attr(op_desc, attr, value)
        for input_name, var_nodes in six.iteritems(inputs):
            if not isinstance(var_nodes, list):
                var_nodes = [var_nodes]
            op_desc.set_input(input_name,
                              [var_node.name() for var_node in var_nodes])
        for output_name, var_nodes in six.iteritems(outputs):
            if not isinstance(var_nodes, list):
                var_nodes = [var_nodes]
            op_desc.set_output(output_name,
                               [var_node.name() for var_node in var_nodes])
        return IrOpNode(self.graph.create_op_node(op_desc))

    def create_op_node_from_desc(self, op_desc):
        """
        Create a operator node by using an existing OpDesc in the graph.

        Args:
            op_desc(core.VarDesc): the giving operator description.

        Returns:
            IrOpNode: the created operator node.
        """
        return IrOpNode(self.graph.create_op_node(op_desc))

    def update_input_link(self, old_input_node, new_input_node, op_node):
        """
        Update the input's link of a operator node.

        Args:
            old_input_node(IrNode): the old input node of the giving op_node.
            new_input_node(IrNode): the new input node of the giving op_node.
            op_node(IrOpNode): the operator node that is needed to update input's link.
        """
        assert old_input_node.node in self.graph.nodes() and new_input_node.node in \
        self.graph.nodes() and op_node.node in self.graph.nodes(), \
        'The three arguments(old_input_node&new_input_node&op_node) must be in the graph nodes.'
        old_input_node.remove_output(op_node)
        op_node.remove_input(old_input_node)
        new_input_node.append_output(op_node)
        op_node.append_input(new_input_node)
        op_node.rename_input(old_input_node.name(), new_input_node.name())

    def link_to(self, node_in, node_out):
        """
        Connect two nodes.

        Args:
            node_in(IrNode): the input node.
            node_out(IrNode): the output node.
        """
        assert node_in.node in self.graph.nodes() and node_out.node in self.graph.nodes(), \
            'The two arguments(node_in&node_out) must be in the graph nodes.'
        node_in.append_output(node_out)
        node_out.append_input(node_in)

    def safe_remove_nodes(self, remove_nodes):
        """
        Remove nodes safely since links connected to these removed nodes are
        also removed.

        Args:
            remove_nodes(set): the nodes prepared to be removed.
        """
        if not isinstance(remove_nodes, set):
            if isinstance(remove_nodes, Iterable):
                remove_nodes = set(remove_nodes)
            else:
                remove_nodes = {remove_nodes}
        original_nodes = {n.node for n in remove_nodes}
        core.graph_safe_remove_nodes(self.graph, original_nodes)

    def has_circle(self):
        """
        Check if the graph has a circle.

        Returns:
            bool: True if the graph has a circle else False.
        """
        return core.has_circle(self.graph)

    def graph_num(self):
        """
        Count the number of unconnected graphs in this graph.

        Returns:
            int: the number of unconnected graphs.
        """
        return core.graph_num(self.graph)

    def topology_sort(self):
        """
        Perform the topology sort operation on the graph.

        Notes: the `graph` cannot contain a circle.

        Returns:
            set(IrNode): nodes in topology order.
        """
        ordered_nodes = core.topology_sort(self.graph)
        return {IrNode(n) for n in ordered_nodes}

    def build_adjacency_list(self):
        """
        Build an adjacency list of operations for the `graph`.

        Returns:
            dict{IrNode: set(IrNode)}: the adjacency list.
        """
        adj_list = core.build_adjacency_list(self.graph)
        wrapped_adj_list = dict()
        for k, v in six.iteritems(adj_list):
            wrapped_adj_list[IrNode(k)] = {IrNode(n) for n in v}
        return wrapped_adj_list

    def draw(self, save_path, name, marked_nodes=None, remove_ctr_var=True):
        """
        Draw the graph. If `dot` command is installed, the drawn graph
        will be saved as pdf file type, otherwise dot file type is used.

        Args:
            save_path(str): the save path of drawn graph.
            name(str): the name of drawn graph.
            marked_nodes(set(IrNode)): nodes that are needed to be marked.
            Default value is None.
            remove_ctr_var(bool): If it is set True, all control variable nodes
            in the graph will be removed. Default value is True.
        """

        def _convert_to_pdf(dot_file_path):
            pdf_save_path = os.path.splitext(dot_file_path)[0] + '.pdf'
            exited_code = subprocess.call('dot -Tpdf ' + dot_file_path \
                            + ' -o ' + pdf_save_path, shell=True)
            if exited_code != 0:
                print('The dot command is needed for creating pdf files.')
                print('The {} is saved as the dot filetype.'.format(
                    dot_file_path))

        remove_ctr_vars = set()
        if remove_ctr_var:
            for node in self.all_var_nodes():
                if node.is_ctrl_var():
                    remove_ctr_vars.add(node)
            self.safe_remove_nodes(remove_ctr_vars)
        print('Total ops num = {}.'.format(len(self.all_op_nodes())))

        if marked_nodes is not None:
            if not isinstance(marked_nodes, set):
                if isinstance(marked_nodes, Iterable):
                    marked_nodes = set(marked_nodes)
                else:
                    marked_nodes = {marked_nodes}
            marked_nodes = {n.node for n in marked_nodes}
            remove_ctr_vars = {n.node for n in remove_ctr_vars}
            marked_nodes = marked_nodes - remove_ctr_vars
            if self.graph.has('__graphviz__marked_node__'):
                self.graph.erase('__graphviz__marked_node__')
            self.graph.set('__graphviz__marked_node__', marked_nodes)
        viz_dot_path = os.path.join(save_path, name) + '.dot'
        viz_pass = core.get_pass('graph_viz_pass')
        viz_pass.set('graph_viz_path', viz_dot_path)
        viz_pass.apply(self.graph)
        _convert_to_pdf(viz_dot_path)

    def to_program(self):
        """
        Convert the graph into a Program.

        Notes: When the graph includes backward operator nodes, the
        conversion process may be failed. Usually, this function is
        only used to convert a test graph.

        Returns:
            Program: a program converted from the graph.
        """
        convert_pass = core.get_pass('graph_to_program_pass')
        desc = core.ProgramDesc()
        convert_pass.set_not_owned('program', desc)
        convert_pass.apply(self.graph)
        program = Program._construct_from_desc(desc)
        return program

    def _update_desc_attr(self, desc, name, val):
        """
        Update the value of desc's attribute by attribute's name.
        """
        if isinstance(val, Block):
            desc.set_block_attr(name, val.desc)
        elif isinstance(val, list) and val and all(
                isinstance(v, Block) for v in val):
            desc.set_blocks_attr(name, [v.desc for v in val])
        elif isinstance(val, core.BlockDesc) or \
                isinstance(val, core.ProgramDesc):
            desc.set_serialized_attr(name, val.serialize_to_string())
        else:
            desc._set_attr(name, val)


class Program(object):
    """
    Python Program. Beneath it is a ProgramDesc, which is used for
    create c++ Program. A program is a self-contained programing
    language like container. It has at least one Block, when the
    control flow op like conditional_block, while_op is included,
    it will contains nested block.
    Please reference the framework.proto for details.

    Notes: we have default_startup_program and default_main_program
    by default, a pair of them will shared the parameters.
    The default_startup_program only run once to initialize parameters,
    default_main_program run in every mini batch and adjust the weights.

    Returns:
        A empty program.

    Examples:
        >>> main_program = fluid.Program()
        >>> startup_program = fluid.Program()
        >>> with fluid.program_guard(main_program=main_program, startup_program=startup_program):
        >>>     fluid.layers.data(name="x", shape=[-1, 784], dtype='float32')
        >>>     fluid.layers.data(name="y", shape=[-1, 1], dtype='int32')
        >>>     fluid.layers.fc(name="fc", shape=[10], dtype='float32', act="relu")

    """

    def __init__(self):
        self.desc = core.ProgramDesc()
        self.blocks = [Block(self, 0)]
        self.current_block_idx = 0
        self._seed = 0
        self._current_role = core.op_proto_and_checker_maker.OpRole.Forward
        self._op_role_var = []

        # for distribute training
        # _is_distributed = True if under distributed training
        self._is_distributed = False
        # _is_chief = True if the trainer is the first one, usually No.0
        self._is_chief = False
        # _parameters_on_pservers records all the parameters distributed on parameter servers.
        self._parameters_on_pservers = None
        # _endpoints is a list about parameter servers ip:port, such as ["ip:port","ip:port"]
        self._endpoints = []
        # if current role is parameter server, the _ps_endpoint is its "ip:port"
        self._ps_endpoint = None
        # trainers_endpoints, it is used for distribution.
        self._trainers_endpoints = []
        # the distributed lookup table names
        self._distributed_lookup_table = None
        # @deprecated(the python memory optimize transpiler is deprecated)
        # whether the program is optimized by memory_optimize_transpiler
        self.__is_mem_optimized = False

    @property
    def _is_mem_optimized(self):
        # if the program is optimized, operator input/outputs
        # maybe same, which conflict with save_inference_model.
        return self.__is_mem_optimized

    @_is_mem_optimized.setter
    def _is_mem_optimized(self, target):
        self.__is_mem_optimized = target

    @property
    def op_role(self):
        """
        The operator role. In a enum {Forward, Backward, Optimize}.

        Notes: this is a low level API. It is used only for ParallelExecutor to
        duplicate or schedule operator to devices.

        For example, the forward operator should be executed on every device.
        The backward operator should be executed on every device and the
        parameter gradient of backward (use :code:`op_role_var` to get this
        variable) operator should be merged to one device. The optimization
        operators should be executed on only one device and broadcast the
        optimization result, i.e., the new parameter, to every other device.
        """
        return self._current_role

    @op_role.setter
    def op_role(self, role):
        self._current_role = role

    @property
    def op_role_var(self):
        """
        The auxiliary variables for :code:`op_role` property.

        See Also: :code:`Program.op_role`'s documentation for details.

        Notes: This is a very low-level API. Users should not use it directly.
        """
        return self._op_role_var

    @op_role_var.setter
    def set_op_role_var(self, var_name):
        self._op_role_var = [var_name]

    @signature_safe_contextmanager
    def _optimized_guard(self, param_and_grads):
        """
        A with guard to set :code:`Optimization` :code:`OpRole` and
        :code:`OpRoleVar` automatically.

        Notes: This is a very low level API. Users should not use it directly.

        Args:
            param_and_grads(list): The variables (names) to be optimized.

        Examples:

            >>> p, g = backward(...)
            >>> with program._optimized_guard([p,g]):
            >>>     p = p - 0.001 * g
        """
        tmp_role = self._current_role
        tmp_var = self._op_role_var

        OpRole = core.op_proto_and_checker_maker.OpRole
        self._current_role = OpRole.Optimize
        self._op_role_var = [
            var.name if isinstance(var, Variable) else var
            for var in param_and_grads
        ]
        yield
        self._op_role_var = tmp_var
        self._current_role = tmp_role

    @signature_safe_contextmanager
    def _lr_schedule_guard(self, is_with_opt=False):
        """
        A with guard to set :code:`LRSched` :code:`OpRole` and
        :code:`OpRoleVar` automatically. The :code:`OpRoleVar` is
        set to the target learning rate.

        Notes: This is a very low level API. Users should not use it directly.

        Args:
            is_with_opt: Only set to true if these ops a in the middle
                 of a bunch of optimize ops so that it can be treated
                 correctly. For example, sgd->lr_op->sgd->lr_op->sgd.

        Examples:

            >>> p, g = backward(...)
            >>> with program.lr_schedule_guard():
            >>>     lr = lr * decay
        """

        tmp_role = self._current_role
        tmp_var = self._op_role_var

        OpRole = core.op_proto_and_checker_maker.OpRole
        self._current_role = OpRole.LRSched
        if is_with_opt:
            self._current_role = int(OpRole.LRSched) | int(OpRole.Optimize)
        # TODO(typhoonzero): how to set target learning rate var
        self._op_role_var = []
        yield
        self._op_role_var = tmp_var
        self._current_role = tmp_role

    def __str__(self):
        """
        Get the protobuf debug string of this Program.

        Returns:
            (str): The protobuf debug string.

        Raises:
            ValueError: If any of required fields is not set.
        """
        return self.to_string(True)

    def to_string(self, throw_on_error, with_details=False):
        """
        To debug string.

        Args:
            throw_on_error(bool): raise Value error when any of required fields
                is not set.

            with_details(bool): True if more details about variables and
                parameters, e.g., :code:`trainable`, :code:`optimize_attr`, need
                to print.

        Returns:
            str : The debug string.

        Raises:
            ValueError: If any of required fields is not set and throw_on_error is
                True.

        """
        assert isinstance(throw_on_error, bool) and isinstance(with_details,
                                                               bool)
        if with_details:
            res_str = ""
            for block in self.blocks:
                res_str += block.to_string(throw_on_error, with_details)
        else:
            protostr = self.desc.serialize_to_string()
            proto = framework_pb2.ProgramDesc.FromString(
                six.binary_type(protostr))
            res_str = _debug_string_(proto, throw_on_error)
        return res_str

    def _get_desc(self):
        """
        Get the C++ side of `ProgramDesc` object pointer. The C++ object is
        exposed by :code:`pybind`.

        Notes: This is a very low level API. Users should not use this API
        directly.
        """
        return self.desc

    def _version(self):
        return self.desc._version()

    def clone(self, for_test=False):
        """
        Create a new, duplicated program.


        Some operators, e.g., :code:`batch_norm`, behave differently between
        training and testing. They have an attribute, :code:`is_test`, to
        control this behaviour. This method will change the :code:`is_test`
        attribute of them to :code:`True` when :code:`for_test=True`.

        * Set for_test to False when we want to clone the program for training.
        * Set for_test to True when we want to clone the program for testing.

        Notes: This API DOES NOT prune any operator. Use
        :code:`clone(for_test=True)` before backward and optimization please. e.g.

            >>> test_program = fluid.default_main_program().clone(for_test=True)
            >>> optimizer = fluid.optimizer.Momentum(learning_rate=0.01, momentum=0.9)
            >>> optimizer.minimize()

        Args:
            for_test(bool): True if change the :code:`is_test` attribute of
                operators to :code:`True`.

        Returns:
            Program: The new, duplicated Program object.

        Examples:

            1. To clone a test program, the sample code is:

            >>> import paddle.fluid as fluid
            >>> train_program = fluid.Program()
            >>> startup_program = fluid.Program()
            >>> with fluid.program_guard(train_program, startup_program):
            >>>     img = fluid.layers.data(name='image', shape=[784])
            >>>     hidden = fluid.layers.fc(input=img, size=200, act='relu')
            >>>     hidden = fluid.layers.dropout(hidden, dropout_prob=0.5)
            >>>     loss = fluid.layers.cross_entropy(
            >>>                 input=fluid.layers.fc(hidden, size=10, act='softmax'),
            >>>                 label=fluid.layers.data(name='label', shape=[1], dtype='int64'))
            >>>
            >>> test_program = train_program.clone(for_test=True)
            >>>
            >>> sgd = fluid.optimizer.SGD(learning_rate=1e-3)
            >>> with fluid.program_guard(train_program, startup_program):
            >>>     sgd.minimize(loss)

            2. The :code:`clone` method can be avoid if you create program for
            training and program for testing individually.

            >>> import paddle.fluid as fluid
            >>>
            >>> def network(is_test):
            >>>     img = fluid.layers.data(name='image', shape=[784])
            >>>     hidden = fluid.layers.fc(input=img, size=200, act='relu')
            >>>     hidden = fluid.layers.dropout(hidden, dropout_prob=0.5, is_test=is_test)
            >>>     loss = fluid.layers.cross_entropy(
            >>>                 input=fluid.layers.fc(hidden, size=10, act='softmax'),
            >>>                 label=fluid.layers.data(name='label', shape=[1], dtype='int64'))
            >>>     return loss
            >>>
            >>> train_program = fluid.Program()
            >>> startup_program = fluid.Program()
            >>> test_program = fluid.Program()
            >>>
            >>> with fluid.program_guard(train_program, startup_program):
            >>>     with fluid.unique_name.guard():
            >>>         loss = network(is_test=False)
            >>>         sgd = fluid.optimizer.SGD(learning_rate=1e-3)
            >>>         sgd.minimize(loss)
            >>>
            >>> # the test startup program is not used.
            >>> with fluid.program_guard(test_program, fluid.Program()):
            >>>     with fluid.unique_name.guard():
            >>>         loss = network(is_test=True)

            The two code snippets above will generate same programs.
        """
        if for_test:
            p = self._inference_optimize(prune_read_op=False)
        else:
            p = Program()
            p.current_block_idx = self.current_block_idx
            p._seed = self._seed
            p.desc = core.ProgramDesc(self.desc)
            p.blocks = [
                Block(p, i) for i in six.moves.range(self.desc.num_blocks())
            ]

            p._current_role = self._current_role
            p._op_role_var = self._op_role_var

            p._sync_with_cpp()

        p._copy_param_info_from(self)
        p._copy_data_info_from(self)
        p._copy_dist_param_info_from(self)
        return p

    def _prune(self, targets):
        """
        Prune operators and variables which are not needed to generate
        :code:`targets`.

        Notes: This is a very low level API. Users should not use this API
        directly. This API is in flux and not stable.

        Args:
            targets(list|Variable|Operator): A list of variables or operators
                need to be pruned

        Returns:
            Program:  A new, pruned program.

        """
        if not isinstance(targets, list):
            targets = [targets]
        targets_idx = []
        for t in targets:
            if not isinstance(t, Operator):
                if isinstance(t, Variable):
                    # After transpiler processing, the op that output this
                    # variable maybe has been changed, so t.op is not reliable
                    # and we need to find the current op that generate this
                    # variable here.
                    t.op = None
                    global_block = self.global_block()
                    for idx, op in enumerate(global_block.ops):
                        if t.name in op.output_arg_names:
                            t.op = op
                            break

                    t = t.op
                    if t is None:
                        raise ValueError(
                            "The target variable must have an "
                            "associated operator that generates it.")
                else:
                    raise ValueError("All targets of prune() can only be "
                                     "Variable or Operator.")

            targets_idx.append([t.block.idx, t.idx])
        res = Program()
        res.desc = core.prune(self.desc, targets_idx)
        res.blocks = [
            Block(res, i) for i in six.moves.range(res.desc.num_blocks())
        ]
        res._sync_with_cpp()
        return res

    def _inference_optimize(self, prune_read_op=True):
        """
        This method will create a new program and do following adjustments on it:
        1. Remove all reader variables and their creator ops if exist.

        2. Remove the :code:`read_op` if exists.

        3. change the :code:`is_test`
        attribute of operators to :code:`True`. All the :code:`Parameter`
        information will be lost.

        Args:
            prune_read_op(bool): remove the read ops that are added by py_reader
                                 for cpp inference library

        Notes: This API is a very low level API. Use
        :code:`Program.clone(for_test=True)` instead.

        Returns:
            Program: The new program.
        """
        res = Program()
        res.desc = core.ProgramDesc(self.desc)

        # remove all readers and the read_op if exist
        read_op_idx = 0
        root_block = res.desc.block(0)
        if prune_read_op:
            while True:
                if read_op_idx >= root_block.op_size() or root_block.op(
                        read_op_idx).type() == 'read':
                    break
                read_op_idx += 1
            if read_op_idx < root_block.op_size():
                root_block._remove_op(0, read_op_idx + 1)
            for var in root_block.all_vars():
                if var.type() == core.VarDesc.VarType.READER:
                    root_block._remove_var(cpt.to_bytes(var.name()))

        # change all `is_test` attributes to True
        for i in six.moves.range(res.desc.num_blocks()):
            block = res.desc.block(i)
            for j in six.moves.range(block.op_size()):
                op = block.op(j)
                if op.has_attr('is_test'):
                    op._set_attr('is_test', True)
        res.blocks = [
            Block(res, i) for i in six.moves.range(res.desc.num_blocks())
        ]
        res._sync_with_cpp()
        return res

    @staticmethod
    def parse_from_string(binary_str):
        """
        Deserialize a program desc from protobuf binary string.

        Notes: All information about parameters will be lost after serialization
        and deserialization.

        Args:
            binary_str_type(str): The binary prootbuf string.

        Returns:
            Program: A deserialized program desc.
        """
        p = Program()
        p.desc = core.ProgramDesc(binary_str)
        p.blocks = [Block(p, i) for i in six.moves.range(p.desc.num_blocks())]
        p._sync_with_cpp()
        return p

    @staticmethod
    def _construct_from_desc(desc):
        """
        Construct a program from program desc.

        Args:
            desc(core.ProgramDesc): The program desc for constructing.

        Returns:
            Program: A program.
        """
        p = Program()
        p.desc = desc
        p.blocks = [Block(p, i) for i in six.moves.range(p.desc.num_blocks())]
        p._sync_with_cpp()
        return p

    @property
    def random_seed(self):
        """
        The default random seed for random operators in Program. Zero means get
        the random seed from random device.

        Notes: It must be set before the operators have been added.
        """
        return self._seed

    @property
    def num_blocks(self):
        """
        The number of blocks in this program.
        """
        return self.desc.num_blocks()

    @random_seed.setter
    def random_seed(self, seed):
        if not isinstance(seed, int):
            raise ValueError("Seed must be a integer.")
        self._seed = seed

    def __repr__(self):
        return self.__str__()

    def global_block(self):
        """
        Get the first block of this program.
        """
        return self.blocks[0]

    def block(self, index):
        """
        Get the :code:`index` block of this program
        Args:
            index(int): The index of block to get

        Returns:
            Block: The :code:`index` block
        """
        return self.blocks[index]

    def current_block(self):
        """
        Get the current block. The :code:`current` block is the block to append
        operators.
        """
        return self.blocks[self.current_block_idx]

    def _create_block(self, parent_idx=None):
        """
        Create a new block with the :code:`parent_idx` and change the current block
        to new block.

        Args:
            parent_idx(int): The parent block index.

        Returns:
            Block: The new block.
        """
        new_block_idx = len(self.blocks)
        parent = self.current_block() if parent_idx is None else self.block(
            parent_idx)
        self.desc.append_block(parent.desc)
        self.current_block_idx = new_block_idx
        self.blocks.append(Block(self, self.current_block_idx))
        return self.current_block()

    def _rollback(self):
        """
        Exit a code block, i.e., roll back to the parent block.
        Returns:
            None
        """
        self.current_block_idx = self.current_block().parent_idx

    def _sync_with_cpp(self):
        """
        Synchronize Python instance to its binding C++ object instance.
        If the program is modified in C++ space, this method should be invoked.

        Notes: This is a very low level API. Users should not invoke it
        directly.

        Returns:
            None
        """
        for block_idx in range(len(self.blocks), self.desc.num_blocks()):
            self.blocks.append(Block(self, block_idx))
        for block in self.blocks:
            block._sync_with_cpp()

    def _copy_param_info_from(self, other):
        """
        Copy the information of parameters from other program.

        Notes: This is a very low level API. Users should not invoke it
        directly.

        Args:
            other(Program): Other program

        Returns:
            None
        """
        if not isinstance(other, Program):
            raise TypeError("_copy_param_info_from should be invoked with "
                            "Program")

        if len(self.blocks) != len(other.blocks):
            raise ValueError("_copy_param_info_from should be invoked with two "
                             "program, with represent the same topology")
        self.global_block()._copy_param_info_from(other.global_block())

    def _copy_dist_param_info_from(self, other):
        """
        Copy the information of distributed information from other program.

        Args:
            other(Program): Other program

        Returns:
            None
        """
        if not isinstance(other, Program):
            raise TypeError("_copy_dist_param_info_from should be invoked with "
                            "Program")
        self._is_distributed = other._is_distributed
        self._is_chief = other._is_chief
        self._parameters_on_pservers = other._parameters_on_pservers
        self._endpoints = other._endpoints
        self._ps_endpoint = other._ps_endpoint
        self._distributed_lookup_table = other._distributed_lookup_table

    def _copy_data_info_from(self, other):
        """
        Copy the information of data variables from other program.

        Notes: This is a very low level API. Users should not invoke it
        directly.

        Args:
            other(Program): Other program

        Returns:
            None
        """
        if not isinstance(other, Program):
            raise TypeError("_copy_param_info_from should be invoked with "
                            "Program")

        if len(self.blocks) != len(other.blocks):
            raise ValueError("_copy_param_info_from should be invoked with two "
                             "program, with represent the same topology")
        for var in list(other.global_block().vars.values()):
            if var.is_data:
                self.global_block().var(var.name).is_data = True

    def list_vars(self):
        """
        Get all variables from this Program. A iterable object is returned.

        Returns:
            iterable: The generator will yield every variable in this program.
        """
        for each_block in self.blocks:
            for each_var in list(each_block.vars.values()):
                yield each_var


class Parameter(Variable):
    """
    Parameter is derived from Variable. A parameter is a persistable
    Variable, and will be updated by optimizers after each iteration.
    The training of a neural network is essentially the updating of
    its parameters.

    Relative to a general Variable, a Parameter has several its own
    member variables:

    Args:
        trainable(bool): True if the parameter need to be updated after
            iterations.
        optimize_attr(map): Parameter attributes related with optimizing.
            Currently, it only contains 'learning_rate'.
            Default: {'learning_rate': 1.0}
        regularizer(WeightDecayRegularizer): The Regularizer which will
            be applied on the parameter. Default: None
        gradient_clip_attr(BaseGradientClipAttr): The gradint clip strategy
            which will be applied on the parameter. Default: None
        do_model_average(bool): True if the model average strategy will
            be applied on this parameter.
    """

    def __init__(self, block, shape, dtype, **kwargs):
        if shape is None or dtype is None:
            raise ValueError("Parameter must set shape and dtype")
        if len(shape) == 0:
            raise ValueError("Parameter shape cannot be empty")

        for each in shape:
            if each < 0:
                raise ValueError("Parameter shape should not be related with "
                                 "batch-size")

        Variable.__init__(
            self, block, persistable=True, shape=shape, dtype=dtype, **kwargs)
        self.trainable = kwargs.get('trainable', True)

        self.optimize_attr = kwargs.get('optimize_attr', {'learning_rate': 1.0})

        self.regularizer = kwargs.get('regularizer', None)

        self.gradient_clip_attr = kwargs.get('gradient_clip_attr', None)

        self.do_model_average = kwargs.get('do_model_average', None)

    def __str__(self):
        return self.to_string(True)

    def to_string(self, throw_on_error, with_details=False):
        """
        To debug string.

        Args:
            throw_on_error(bool): raise exception when self is not initialized
                when throw_on_error is True
            with_details(bool): more details about variables and parameters
                (e.g. trainable, optimize_attr, ...) will be printed when with_details is True

        Returns(str): The debug string.

        """
        assert isinstance(throw_on_error, bool) and isinstance(with_details,
                                                               bool)
        if with_details:
            res_str = Variable.to_string(self, throw_on_error, True)
            additional_attr = ("trainable", "optimize_attr", "regularizer",
                               "gradient_clip_attr", "do_model_average")
            for attr_name in additional_attr:
                res_str += "%s: %s\n" % (
                    attr_name, six.binary_type(getattr(self, attr_name)))
        else:
            res_str = Variable.to_string(self, throw_on_error, False)
        return res_str

    __repr__ = __str__


# program is a global instance.
_main_program_ = Program()
_startup_program_ = Program()


def default_startup_program():
    """
    Get default/global startup program.

    The layer function in :code:`fluid.layers` will create parameters, readers,
    NCCL handles as global variables. The :code:`startup_program` will
    initialize them by the operators in startup program. The layer function will
    append these initialization operators into startup program.

    This method will return the :code:`default` or the :code:`current` startup
    program. Users can use :code:`fluid.program_guard` to switch program.

    Returns:
        Program: startup program
    """
    return _startup_program_


def default_main_program():
    """
    Get default/global main program. The main program is used for training or
    testing.

    All layer function in :code:`fluid.layers` will append operators and
    variables to the :code:`default_main_program`.

    The :code:`default_main_program` is the default program in a lot of APIs.
    For example, the :code:`Executor.run()` will execute the
    :code:`default_main_program` when the program is not specified.

    Returns:
        Program: main program
    """
    return _main_program_


def switch_main_program(program):
    """
    Switch the main program to a new program.

    Args:
        program(Program): The new main program

    Returns:
        Program: The previous main program
    """
    global _main_program_
    prev_program = _main_program_
    _main_program_ = program
    return prev_program


def switch_startup_program(program):
    """
    Switch the startup program to a new program
    Args:
        program(Program): The new startup program

    Returns:
        Program: The previous startup program
    """
    global _startup_program_
    prev_program = _startup_program_
    _startup_program_ = program
    return prev_program


@signature_safe_contextmanager
def program_guard(main_program, startup_program=None):
    """
    Change the global main program and startup program with `with` statement.
    Layer functions in the Python `with` block will append operators and
    variables to the new main programs.

    Examples:

        >>> import paddle.fluid as fluid
        >>> main_program = fluid.Program()
        >>> startup_program = fluid.Program()
        >>> with fluid.program_guard(main_program, startup_program):
        >>>     data = fluid.layers.data(...)
        >>>     hidden = fluid.layers.fc(...)

    Notes: The temporary :code:`Program` can be used if the user does not need
    to construct either of startup program or main program.

    Examples:

        >>> import paddle.fluid as fluid
        >>> main_program = fluid.Program()
        >>> # does not care about startup program. Just pass a temporary value.
        >>> with fluid.program_guard(main_program, fluid.Program()):
        >>>     data = ...

    Args:
        main_program(Program): New main program inside `with` statement.
        startup_program(Program): New startup program inside `with` statement.
            None means do not change startup program.
    """
    if not isinstance(main_program, Program):
        raise TypeError("main_program should be Program")
    main_program = switch_main_program(main_program)
    if startup_program is not None:
        if not isinstance(startup_program, Program):
            raise TypeError("startup_program should be Program")
        startup_program = switch_startup_program(startup_program)
    yield
    switch_main_program(main_program)
    if startup_program is not None:
        switch_startup_program(startup_program)


def _get_var(name, program=None):
    """
    Get a variable by name from the global block of a program.

    Args:
        name(str): name of the variable
        program(Program|None): program object.
        If None, default_global_program() will be used.

    Returns:
        Variable
    """
    if program is None:
        program = default_main_program()
    assert isinstance(name, str)
    assert isinstance(program, Program)

    return program.global_block().var(name)


@signature_safe_contextmanager
def _imperative_guard(tracer):
    global _imperative_tracer_
    tmp_trace = _imperative_tracer_
    _imperative_tracer_ = tracer

    yield

    _imperative_tracer_ = tmp_trace


@signature_safe_contextmanager
def _imperative_place_guard(place):
    global _imperative_current_expected_place_
    tmp_place = _imperative_current_expected_place_
    _imperative_current_expected_place_ = place

    yield

    _imperative_current_expected_place_ = tmp_place
