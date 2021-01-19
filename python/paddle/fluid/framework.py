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
from .wrapped_decorator import signature_safe_contextmanager, wrap_decorator
import os
import re
import traceback
import six
import copy

import numpy as np
import subprocess
import multiprocessing
import sys
import logging
from .. import compat as cpt
from .proto import framework_pb2

from . import core
from . import unique_name
import paddle.version as fluid_version
import warnings
import functools

__all__ = [
    'Program',
    'default_startup_program',
    'default_main_program',
    'program_guard',
    'name_scope',
    'cuda_places',
    'cpu_places',
    'xpu_places',
    'cuda_pinned_places',
    'in_dygraph_mode',
    'is_compiled_with_cuda',
    'is_compiled_with_xpu',
    'Variable',
    'load_op_library',
    'require_version',
    'device_guard',
    'set_flags',
    'get_flags',
]

EMPTY_VAR_NAME = core.kEmptyVarName()
TEMP_VAR_NAME = core.kTempVarName()
GRAD_VAR_SUFFIX = core.kGradVarSuffix()
ZERO_VAR_SUFFIX = core.kZeroVarSuffix()
CONTROL_DEP_VAR_PREFIX = core.kControlDepVarName()

_dygraph_tracer_ = None
_global_expected_place_ = None
_current_device = None
global_prog_seed = 0


def require_version(min_version, max_version=None):
    """
        Check if the installed version of PaddlePaddle is in [min_version, max_version],
        if the installed version is lower than ``min_version`` or higher than ``max_version``,
        an exception will be thrown, NO returns if the installed version is satisfied.

        Args:
            min_version (str): the minimum version required (like '1.4.0').
            max_version (str, optional): the max version required (like '1.6.0'), default is None,
                meaning any version equal or higher than ``min_version`` is acceptable.

        Returns:
            None.

        Raises:
            TypeError: if the type of ``min_version`` is not str.
            TypeError: if the type of ``max_version`` is not str or type(None).
            ValueError: if the value of ``min_version`` is not in version format.
            ValueError: if the value of ``max_version`` is not in version format or None.
            Exception: if the installed version is lower than ``min_version`` or higher than ``max_version``.

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid

                # any version >= 0.1.0 is acceptable.
                fluid.require_version('0.1.0')

                # if 0.1.0 <= version <= 10.0.0, it is acceptable.
                fluid.require_version(min_version='0.1.0', max_version='10.0.0')
        """
    if not isinstance(min_version, str):
        raise TypeError(
            "The type of 'min_version' in require_version must be str, but received %s."
            % (type(min_version)))

    if not isinstance(max_version, (str, type(None))):
        raise TypeError(
            "The type of 'max_version' in require_version must be str or type(None), but received %s."
            % (type(max_version)))

    check_format = re.match(r'\d+(\.\d+){0,3}', min_version)
    if check_format is None or check_format.group() != min_version:
        raise ValueError(
            "The value of 'min_version' in require_version must be in format '\\d+(\\.\\d+){0,3}', "
            "like '1.5.2.0', but received %s" % min_version)

    if max_version is not None:
        check_format = re.match(r'\d+(\.\d+){0,3}', max_version)
        if check_format is None or check_format.group() != max_version:
            raise ValueError(
                "The value of 'max_version' in require_version must be in format '\\d+(\\.\\d+){0,3}', "
                "like '1.5.2.0', but received %s" % max_version)

    version_installed = [
        fluid_version.major, fluid_version.minor, fluid_version.patch,
        fluid_version.rc
    ]
    zero_version = ['0', '0', '0', '0']

    def version_cmp(ver_a, ver_b):
        for i in six.moves.range(len(ver_a)):
            if int(ver_a[i]) > int(ver_b[i]):
                return 1
            elif int(ver_a[i]) < int(ver_b[i]):
                return -1
        return 0

    if version_cmp(version_installed, zero_version) == 0:
        if max_version is not None:
            warnings.warn(
                "PaddlePaddle version in [%s, %s] required, but %s installed. "
                "Maybe you are using a develop version, "
                "please make sure the version is good with your code." %
                (min_version, max_version, fluid_version.full_version))
        else:
            warnings.warn(
                "PaddlePaddle version %s or higher is required, but %s installed, "
                "Maybe you are using a develop version, "
                "please make sure the version is good with your code." %
                (min_version, fluid_version.full_version))
        return

    min_version_split = min_version.split('.')
    min_version_to_check = min_version_split + zero_version[len(
        min_version_split):]

    if max_version is not None:
        max_version_split = max_version.split('.')
        max_version_to_check = max_version_split + zero_version[len(
            max_version_split):]

        if version_cmp(version_installed,
                       max_version_to_check) > 0 or version_cmp(
                           version_installed, min_version_to_check) < 0:
            raise Exception(
                "VersionError: PaddlePaddle version in [%s, %s] required, but %s installed."
                % (min_version, max_version, fluid_version.full_version))
    else:
        if version_cmp(version_installed, min_version_to_check) < 0:
            raise Exception(
                "VersionError: PaddlePaddle version %s or higher is required, but %s installed, "
                "please upgrade your PaddlePaddle to %s or other higher version."
                % (min_version, fluid_version.full_version, min_version))


def in_dygraph_mode():
    """

    .. note::
        Dynamic graph mode is turn ON by default since paddle 2.0.0

    This API checks whether paddle runs in dynamic graph mode.

    You can turn ON static graph mode by `enable_static <../dygraph/base/disable_dygraph_en.html>`_ ,
    and turn OFF static graph mode by `disable_static <../dygraph/base/enable_dygraph_en.html>`_  .

    Returns:
        bool: Whether paddle runs in dynamic graph mode.

    Examples:
        .. code-block:: python

            import paddle
            print(paddle.in_dynamic_mode())  # True, dynamic mode is turn ON by default since paddle 2.0.0

            paddle.enable_static()
            print(paddle.in_dynamic_mode())  # False, Now we are in static mode

            paddle.disable_static()
            print(paddle.in_dynamic_mode())  # True, Now we are in dynamic mode

    """
    return _dygraph_tracer_ is not None


def _dygraph_not_support_(func):
    def __impl__(*args, **kwargs):
        assert not in_dygraph_mode(
        ), "We don't support %s in imperative mode" % func.__name__
        return func(*args, **kwargs)

    return __impl__


def _dygraph_only_(func):
    def __impl__(*args, **kwargs):
        assert in_dygraph_mode(
        ), "We only support '%s()' in dynamic graph mode, please call 'paddle.disable_static()' to enter dynamic graph mode." % func.__name__
        return func(*args, **kwargs)

    return __impl__


def _static_only_(func):
    def __impl__(*args, **kwargs):
        assert not in_dygraph_mode(
        ), "In PaddlePaddle 2.x, we turn on dynamic graph mode by default, and '%s()' is only supported in static graph mode. So if you want to use this api, please call 'paddle.enable_static()' before this api to enter static graph mode." % func.__name__
        return func(*args, **kwargs)

    return __impl__


# NOTE(zhiqiu): This decorator is used for the APIs of Variable which is only
# used to make Variable and VarBase has same interfaces, like numpy. Since VarBase is not exposed in our
# official docments, logically, we want to keep VarBase and logically consistent. While, actually,
# in our implementation, there some APIs not supported, like numpy, because Variable contains the desc.
# So, those APIs are listed under class Variable to generate docs only.
# TODO(zhiqiu): We should make VarBase consistent with Variable in future, for example, by inheritting
# same base class.
def _fake_interface_only_(func):
    def __impl__(*args, **kwargs):
        raise AssertionError(
            "'%s' should be called by imperative Varible in imperative mode, please run it in dygraph "
            "mode. You can turn off paddle.enable_static() if you are in static mode, or turn off "
            "ProgramTranslator if you are using @paddle.jit.to_static" %
            func.__name__)

    return __impl__


# NOTE(chenweihang): There is argument name typo (stat_dict, correct name is state_dict)
# in fluid api Layer.set_dict, Optimizer.load, in order to correct the argument without
# introducing compatibility issues, add this decorator
# NOTE(chenweihang): not using `wrap_decorator` here is because `wrap_decorator` will
# move kwargs to args, which doesn't work in this decorate case
def deprecate_stat_dict(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if 'stat_dict' in kwargs:
            warnings.warn(
                "The argument `stat_dict` has deprecated, please change it to `state_dict`.",
                DeprecationWarning)
            kwargs['state_dict'] = kwargs['stat_dict']
            kwargs.pop('stat_dict')
        return func(*args, **kwargs)

    return wrapper


dygraph_not_support = wrap_decorator(_dygraph_not_support_)
dygraph_only = wrap_decorator(_dygraph_only_)
static_only = wrap_decorator(_static_only_)
fake_interface_only = wrap_decorator(_fake_interface_only_)


def _dygraph_tracer():
    return _dygraph_tracer_


def _current_expected_place():
    global _global_expected_place_
    if _global_expected_place_ is None:
        if core.is_compiled_with_cuda():
            try:
                device_count = core.get_cuda_device_count()
            except Exception as e:
                device_count = 0
            if device_count > 0:
                _global_expected_place_ = core.CUDAPlace(0)
            else:
                warnings.warn(
                    "You are using GPU version Paddle, but your CUDA device is not set properly. CPU device will be used by default."
                )
                _global_expected_place_ = core.CPUPlace()
        else:
            _global_expected_place_ = core.CPUPlace()

    return _global_expected_place_


def _set_dygraph_tracer_expected_place(place):
    global _dygraph_tracer_
    if _dygraph_tracer_ is not None:
        _dygraph_tracer_._expected_place = place


def _set_expected_place(place):
    global _global_expected_place_
    _global_expected_place_ = place
    _set_dygraph_tracer_expected_place(place)


# TODO(zhiqiu): remove this function.
def _var_base_to_np(var_base):
    """	
    convert VarBase tp numpy	

    Args:	
        var_base(VarBase) : the VarBase to convert	
    Returns (np.ndarray): the np.ndarray contain the value of VarBase	
    """

    warnings.warn(
        "paddle.fluid.framework._var_base_to_np is deprecated, please use var_base.numpy() instead of _var_base_to_np(var_base)."
    )

    return var_base.numpy()


def _cpu_num():
    if "CPU_NUM" not in os.environ.keys():
        if multiprocessing.cpu_count() > 1:
            sys.stderr.write(
                '!!! The CPU_NUM is not specified, you should set CPU_NUM in the environment variable list.\n'
                'CPU_NUM indicates that how many CPUPlace are used in the current task.\n'
                'And if this parameter are set as N (equal to the number of physical CPU core) the program may be faster.\n\n'
                'export CPU_NUM={} # for example, set CPU_NUM as number of physical CPU core which is {}.\n\n'
                '!!! The default number of CPU_NUM=1.\n'.format(
                    multiprocessing.cpu_count(), multiprocessing.cpu_count()))
        os.environ['CPU_NUM'] = str(1)
    cpu_num = os.environ.get('CPU_NUM')
    return int(cpu_num)


def _cuda_ids():
    gpus_env = os.getenv("FLAGS_selected_gpus")
    if gpus_env:
        device_ids = [int(s) for s in gpus_env.split(",")]
    else:
        device_ids = six.moves.range(core.get_cuda_device_count())
    return device_ids


def _xpu_ids():
    xpus_env = os.getenv("FLAGS_selected_xpus")
    if xpus_env:
        device_ids = [int(s) for s in xpus_env.split(",")]
    else:
        device_ids = six.moves.range(core.get_xpu_device_count())
    return device_ids


def is_compiled_with_xpu():
    """
    Whether this whl package can be used to run the model on XPU.

    Returns (bool): support xpu or not.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            support_xpu = fluid.is_compiled_with_xpu()
    """
    return core.is_compiled_with_xpu()


def is_compiled_with_cuda():
    """
    Whether this whl package can be used to run the model on GPU.

    Returns (bool): `True` if CUDA is currently available, otherwise `False`.

    Examples:
        .. code-block:: python

            import paddle
            support_gpu = paddle.is_compiled_with_cuda()
    """
    return core.is_compiled_with_cuda()


def cuda_places(device_ids=None):
    """
    **Note**:
        For multi-card tasks, please use `FLAGS_selected_gpus` environment variable to set the visible GPU device.
        The next version will fix the problem with `CUDA_VISIBLE_DEVICES` environment variable.

    This function creates a list of :code:`paddle.CUDAPlace` objects.

    If :code:`device_ids` is None, environment variable of
    :code:`FLAGS_selected_gpus` would be checked first. For example, if
    :code:`FLAGS_selected_gpus=0,1,2`, the returned list would
    be [paddle.CUDAPlace(0), paddle.CUDAPlace(1), paddle.CUDAPlace(2)].
    If :code:`FLAGS_selected_gpus` is not set, all visible
    gpu places would be returned according to the :code:`CUDA_VISIBLE_DEVICES` environment variable.

    If :code:`device_ids` is not None, it should be the device
    ids of GPUs. For example, if :code:`device_ids=[0,1,2]`,
    the returned list would be 
    [paddle.CUDAPlace(0), paddle.CUDAPlace(1), paddle.CUDAPlace(2)].

    Parameters:
        device_ids (list or tuple of int, optional): list of GPU device ids.

    Returns:
        list of paddle.CUDAPlace: Created GPU place list.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.static as static

            paddle.enable_static()

            cuda_places = static.cuda_places()

    """
    assert core.is_compiled_with_cuda(), \
        "Not compiled with CUDA"
    if device_ids is None:
        device_ids = _cuda_ids()
    elif not isinstance(device_ids, (list, tuple)):
        device_ids = [device_ids]
    return [core.CUDAPlace(dev_id) for dev_id in device_ids]


def xpu_places(device_ids=None):
    """
    **Note**:
        For multi-card tasks, please use `FLAGS_selected_xpus` environment variable to set the visible XPU device.
    This function creates a list of :code:`paddle.XPUPlace` objects.
    If :code:`device_ids` is None, environment variable of
    :code:`FLAGS_selected_xpus` would be checked first. For example, if
    :code:`FLAGS_selected_xpus=0,1,2`, the returned list would
    be [paddle.XPUPlace(0), paddle.XPUPlace(1), paddle.XPUPlace(2)].
    If :code:`FLAGS_selected_xpus` is not set, all visible
    xpu places would be returned.
    If :code:`device_ids` is not None, it should be the device
    ids of XPUs. For example, if :code:`device_ids=[0,1,2]`,
    the returned list would be 
    [paddle.XPUPlace(0), paddle.XPUPlace(1), paddle.XPUPlace(2)].
    
    Parameters:
        device_ids (list or tuple of int, optional): list of XPU device ids.
    Returns:
        list of paddle.XPUPlace: Created XPU place list.
    Examples:
        .. code-block:: python
        
            import paddle
            import paddle.static as static
            
            paddle.enable_static()
            xpu_places = static.xpu_places()
    """
    assert core.is_compiled_with_xpu(), \
        "Not compiled with XPU"
    if device_ids is None:
        device_ids = _xpu_ids()
    elif not isinstance(device_ids, (list, tuple)):
        device_ids = [device_ids]
    return [core.XPUPlace(dev_id) for dev_id in device_ids]


def cpu_places(device_count=None):
    """
    This function creates a list of :code:`paddle.CPUPlace` objects, and returns the created list.

    If :code:`device_count` is None, the device count would
    be determined by environment variable :code:`CPU_NUM`. 
    If :code:`CPU_NUM` is not set, the default value is 1,
    i.e. CPU_NUM=1.
    :code:`CPU_NUM` indicates the number of devices used in the current task.
    The running of the program can be accelerated if :code:`CPU_NUM` is the same as the number of physical cores.

    Parameters:
        device_count (int, optional): device number. Default: None.

    Returns:
        list of paddle.CPUPlace: Created list of CPU places.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.static as static

            paddle.enable_static()

            cpu_places = static.cpu_places()
    """

    if device_count is None:
        device_count = _cpu_num()
    return [core.CPUPlace()] * device_count


def cuda_pinned_places(device_count=None):
    """
    This function creates a list of :code:`fluid.CUDAPinnedPlace` objects.

    If :code:`device_count` is None, the device count would
    be determined by environment variable :code:`CPU_NUM`. 
    If :code:`CPU_NUM` is not set, the default value is 1,
    i.e. CPU_NUM=1.
    :code:`CPU_NUM` indicates the number of devices used in the current task.
    The running of the program can be accelerated if :code:`CPU_NUM` is the same as the number of physical cores.

    Parameters:
        device_count (int, optional): device number. Default: None.

    Returns:
        list of fluid.CUDAPinnedPlace: Created list of CUDA pinned places.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            cuda_pinned_places_cpu_num = fluid.cuda_pinned_places()
            # or
            cuda_pinned_places = fluid.cuda_pinned_places(1)

    """
    assert core.is_compiled_with_cuda(), \
        "Not compiled with CUDA"
    if device_count is None:
        device_count = len(_cuda_ids())
    return [core.CUDAPinnedPlace()] * device_count


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
    :api_attr: Static Graph

    Generate hierarchical name prefix for the operators in Static Graph.

    Note: 
        This should only used for debugging and visualization purpose.
        Don't use it for serious analysis such as graph/program transformations.
        Don't use it in dygraph, since it will cause memory leak.

    Args:
        prefix(str, optional): prefix. Default is none.

    Examples:
        .. code-block:: python

          import paddle
          paddle.enable_static()
          with paddle.static.name_scope("s1"):
             a = paddle.static.data(name='data', shape=[None, 1], dtype='int32')
             b = a + 1
             with paddle.static.name_scope("s2"):
                c = b * 1
             with paddle.static.name_scope("s3"):
                d = c / 1
          with paddle.static.name_scope("s1"):
                f = paddle.tensor.pow(d, 2.0)
          with paddle.static.name_scope("s4"):
                g = f - 1

          # Op are created in the default main program.  
          for op in paddle.static.default_main_program().block(0).ops:
              # elementwise_add is created in /s1/
              if op.type == 'elementwise_add':
                  assert op.desc.attr("op_namescope") == '/s1/'
              # elementwise_mul is created in '/s1/s2'
              elif op.type == 'elementwise_mul':
                  assert op.desc.attr("op_namescope") == '/s1/s2/'
              # elementwise_div is created in '/s1/s3'
              elif op.type == 'elementwise_div':
                  assert op.desc.attr("op_namescope") == '/s1/s3/'
              # elementwise_sum is created in '/s4'
              elif op.type == 'elementwise_sub':
                  assert op.desc.attr("op_namescope") == '/s4/'
              # pow is created in /s1_1/
              elif op.type == 'pow':
                  assert op.desc.attr("op_namescope") == '/s1_1/'
    """
    # TODO(panyx0718): Only [0-9a-z].
    # in dygraph we don't need namescope since it will cause mem leak
    if in_dygraph_mode():
        yield
    else:
        assert prefix, "namescope prefix can not be empty."
        global _name_scope
        _name_scope = _name_scope.child(prefix)
        try:
            yield
        finally:
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
        # since there is still no support for bfloat16 in NumPy,
        # uint16 is used for casting bfloat16
        return core.VarDesc.VarType.BF16
    elif dtype == np.uint8:
        return core.VarDesc.VarType.UINT8
    elif dtype == np.int8:
        return core.VarDesc.VarType.INT8
    elif dtype == np.complex64:
        return core.VarDesc.VarType.COMPLEX64
    elif dtype == np.complex128:
        return core.VarDesc.VarType.COMPLEX128
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


def _varbase_creator(type=core.VarDesc.VarType.LOD_TENSOR,
                     name=None,
                     shape=None,
                     dtype=None,
                     persistable=None,
                     **kwargs):
    if dtype is not None:
        if not isinstance(dtype, core.VarDesc.VarType):
            dtype = convert_np_dtype_to_dtype_(dtype)

    return core.VarBase(dtype if dtype else core.VarDesc.VarType.FP32,
                        list(shape) if shape else [], name, type
                        if type else core.VarDesc.VarType.LOD_TENSOR, True
                        if persistable else False)


class VariableMetaClass(type):
    @classmethod
    def __instancecheck__(cls, instance):
        t = type(instance)
        if in_dygraph_mode():
            return issubclass(t, core.VarBase)
        else:
            return issubclass(t, Variable)


class ParameterMetaClass(VariableMetaClass):
    @classmethod
    def __instancecheck__(cls, instance):
        t = type(instance)
        if in_dygraph_mode():
            return issubclass(t, ParamBase)
        else:
            return issubclass(t, Parameter)


def _getitem_impl_(var, item):
    """
    Slice the variable.

    Args:
        item(int/slice/tuple) : the index.

    Returns:
        Sliced variable
    """

    if not isinstance(item, tuple):
        item = [item]

    decrease_axis = []
    slice_axis = []
    slice_start = []
    slice_end = []
    slice_step = []
    use_strided_slice = False
    reverse_axis = []
    target_block = default_main_program().current_block()

    def fill_constant(shape, value, force_cpu=False, out=None):
        var.block.append_op(
            type='fill_constant',
            inputs={},
            outputs={'Out': [out]},
            attrs={
                'shape': shape,
                'dtype': out.dtype,
                'value': float(value),
                'force_cpu': force_cpu
            })
        out.stop_gradient = True
        return out

    for dim, slice_item in enumerate(item):
        if isinstance(slice_item, slice):
            start = slice_item.start
            end = slice_item.stop
            step = slice_item.step

            if start is None and end is None and step is None:
                continue

            if step is None:
                step = 1

            if start is None and end is None:
                assert (step == -1)
                reverse_axis.append(dim)
                continue

            if start is None:
                start = 0

            if end is None:
                end = 10000000

            if step != 1:
                use_strided_slice = True

            slice_axis.append(dim)
            slice_start.append(start)
            slice_end.append(end)
            slice_step.append(step)
        else:
            decrease_axis.append(dim)
            slice_axis.append(dim)
            slice_start.append(slice_item)
            slice_step.append(1)
            if isinstance(slice_item, Variable):
                temp_1 = var.block.create_var(dtype=slice_item.dtype)
                fill_constant([1], 1, force_cpu=True, out=temp_1)
                temp_end = target_block.create_var(dtype=slice_item.dtype)
                target_block.append_op(
                    type='elementwise_add',
                    inputs={'X': slice_item,
                            'Y': temp_1},
                    outputs={'Out': temp_end},
                    attrs={'axis': -1})
                slice_end.append(temp_end)
            else:
                slice_end.append(slice_item + 1
                                 if slice_item != -1 else 10000000)

    def contain_var(one_list):
        for ele in one_list:
            if isinstance(ele, Variable):
                return True
        return False

    def get_new_list_tensor(old_list):
        new_list_tensor = []
        for dim in old_list:
            if isinstance(dim, Variable):
                dim.stop_gradient = True
                new_list_tensor.append(dim)
            else:
                assert (isinstance(dim, int))
                temp_out = var.block.create_var(dtype='int32')
                fill_constant([1], dim, force_cpu=True, out=temp_out)
                new_list_tensor.append(temp_out)
        return new_list_tensor

    inputs = {'Input': [var]}
    attrs = {
        'axes': slice_axis,
        'starts': [],
        'ends': [],
        'decrease_axis': decrease_axis
    }
    if (use_strided_slice == True):
        attrs['strides'] = []
    infer_flags = list(1 for i in range(len(slice_axis)))

    # starts
    if contain_var(slice_start):
        inputs['StartsTensorList'] = get_new_list_tensor(slice_start)
        for i, dim in enumerate(slice_start):
            if isinstance(dim, Variable):
                attrs['starts'].append(-1)
                infer_flags[i] = -1
            else:
                attrs['starts'].append(dim)
    else:
        attrs['starts'] = slice_start

    # ends
    if contain_var(slice_end):
        inputs['EndsTensorList'] = get_new_list_tensor(slice_end)
        for i, dim in enumerate(slice_end):
            if isinstance(dim, Variable):
                attrs['ends'].append(-1)
                infer_flags[i] = -1
            else:
                attrs['ends'].append(dim)
    else:
        attrs['ends'] = slice_end

    # strides
    if use_strided_slice == True:
        if contain_var(slice_step):
            inputs['StridesTensorList'] = get_new_list_tensor(slice_step)
            for i, dim in enumerate(slice_step):
                if isinstance(dim, Variable):
                    attrs['strides'].append(-1)
                    infer_flags[i] = -1
                else:
                    attrs['strides'].append(dim)
        else:
            attrs['strides'] = slice_step
    # infer_flags
    attrs['infer_flags'] = infer_flags

    out = var
    if use_strided_slice == False and len(slice_axis) > 0:
        # append slice_op here
        slice_out_var = target_block.create_var(
            name=unique_name.generate_with_ignorable_key(var.name + "_slice"),
            dtype=var.dtype)

        target_block.append_op(
            type="slice",
            inputs=inputs,
            outputs={'Out': [slice_out_var]},
            attrs=attrs)

        out = slice_out_var
    elif use_strided_slice == True and len(slice_axis) > 0:
        strided_slice_out_var = target_block.create_var(
            name=unique_name.generate_with_ignorable_key(var.name +
                                                         "_strided_slice"),
            dtype=var.dtype)
        target_block.append_op(
            type="strided_slice",
            inputs=inputs,
            outputs={'Out': [strided_slice_out_var]},
            attrs=attrs)

        out = strided_slice_out_var

    if len(reverse_axis) > 0:
        reverse_out_var = target_block.create_var(
            name=unique_name.generate_with_ignorable_key(var.name +
                                                         "_slice_reverse"),
            dtype=var.dtype)
        target_block.append_op(
            type="reverse",
            inputs={'X': out},
            outputs={'Out': [reverse_out_var]},
            attrs={'axis': reverse_axis})

        out = reverse_out_var

    return out


@six.add_metaclass(VariableMetaClass)
class Variable(object):
    """
    **Notes**:
        **The constructor of Variable should not be invoked directly.**

        **In Static Graph Mode: Please use** `Block.create_var` **to create a Static variable which has no data until being feed.**

        **In Dygraph Mode: Please use** :ref:`api_fluid_dygraph_to_variable` **to create a dygraph variable with real data**

    In Fluid, every input and output of an OP is a variable. In most
    cases, variables are used for holding different kinds of data or training
    labels. A variable belongs to a :ref:`api_guide_Block_en` . All variable has its own name and
    two variables in different :ref:`api_guide_Block_en` could have the same name.

    There are many kinds of variables. Each kind of them has its own attributes
    and usages. Please refer to the `framework.proto <https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/framework/framework.proto>`_ for details.

    Most of a Variable's member variables can be set to be None. It mean
    it is not available or will be specified later.

    Examples:
        In Static Graph Mode:

        .. code-block:: python

            import paddle.fluid as fluid
            cur_program = fluid.Program()
            cur_block = cur_program.current_block()
            new_variable = cur_block.create_var(name="X",
                                                shape=[-1, 23, 48],
                                                dtype='float32')
        In `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_  Mode:

        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np

            with fluid.dygraph.guard():
                new_variable = fluid.dygraph.to_variable(np.arange(10))

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
                 need_check_feed=False,
                 belong_to_optimizer=False,
                 **kwargs):
        self.block = block
        if name is None:
            name = unique_name.generate('_generated_var')

        if dtype is not None:
            if not isinstance(dtype, core.VarDesc.VarType):
                dtype = convert_np_dtype_to_dtype_(dtype)

        self.belong_to_optimizer = belong_to_optimizer

        self.error_clip = error_clip

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

        if need_check_feed and is_new_var:
            self.desc.set_need_check_feed(need_check_feed)

        if capacity is not None:
            if is_new_var:
                self.desc.set_capacity(capacity)
            else:
                # TODO(abhinavarora) : Compare with set capacity once,
                # get_capacity is implemented
                pass

        self.block.vars[name] = self
        self.op = None
        self._stop_gradient = stop_gradient
        self.is_data = is_data

    @fake_interface_only
    def detach(self):
        """
        **Notes**:
            **This API is ONLY available in Dygraph mode**

        Returns a new Variable, detached from the current graph.

        Returns:
             ( :ref:`api_guide_Variable_en` | dtype is same as current Variable): The detached Variable.


        Examples:
            .. code-block:: python

                import paddle.fluid as fluid
                from paddle.fluid.dygraph.base import to_variable
                from paddle.fluid.dygraph import Linear
                import numpy as np

                data = np.random.uniform(-1, 1, [30, 10, 32]).astype('float32')
                with fluid.dygraph.guard():
                    linear = Linear(32, 64)
                    data = to_variable(data)
                    x = linear(data)
                    y = x.detach()

        """
        pass

    @fake_interface_only
    def numpy(self):
        """
        **Notes**:
            **This API is ONLY available in Dygraph mode**

        Returns a numpy array shows the value of current :ref:`api_guide_Variable_en`

        Returns:
            ndarray: The numpy value of current Variable.

        Returns type:
            ndarray: dtype is same as current Variable

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid
                from paddle.fluid.dygraph.base import to_variable
                from paddle.fluid.dygraph import Linear
                import numpy as np

                data = np.random.uniform(-1, 1, [30, 10, 32]).astype('float32')
                with fluid.dygraph.guard():
                    linear = Linear(32, 64)
                    data = to_variable(data)
                    x = linear(data)
                    print(x.numpy())

        """
        pass

    @fake_interface_only
    def set_value(self, value):
        """
        **Notes**:
            **This API is ONLY available in Dygraph mode**

        Set a new value for this Variable.

        Args:
            value (Variable|np.ndarray): the new value.

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid
                from paddle.fluid.dygraph.base import to_variable
                from paddle.fluid.dygraph import Linear
                import numpy as np

                data = np.ones([3, 1024], dtype='float32')
                with fluid.dygraph.guard():
                    linear = fluid.dygraph.Linear(1024, 4)
                    t = to_variable(data)
                    linear(t)  # call with default weight
                    custom_weight = np.random.randn(1024, 4).astype("float32")
                    linear.weight.set_value(custom_weight)  # change existing weight
                    out = linear(t)  # call with different weight

        """
        pass

    @fake_interface_only
    def backward(self, retain_graph=False):
        """
        **Notes**:
            **This API is ONLY available in Dygraph mode**

        Run backward of current Graph which starts from current Tensor.

        Args:
            retain_graph(bool, optional): If False, the graph used to compute grads will be freed. If you would
                like to add more ops to the built graph after calling this method( :code:`backward` ), set the parameter
                :code:`retain_graph` to True, then the grads will be retained. Thus, seting it to False is much more memory-efficient.
                Defaults to False.

        Returns:
            NoneType: None

        Examples:
            .. code-block:: python

                import numpy as np
                import paddle
                paddle.disable_static()

                x = np.ones([2, 2], np.float32)
                inputs = []
                for _ in range(10):
                    tmp = paddle.to_tensor(x)
                    # if we don't set tmp's stop_gradient as False then, all path to loss will has no gradient since
                    # there is no one need gradient on it.
                    tmp.stop_gradient=False
                    inputs.append(tmp)
                ret = paddle.add_n(inputs)
                loss = paddle.sum(ret)
                loss.backward()

        """
        pass

    @fake_interface_only
    def gradient(self):
        """
        **Notes**:
            **This API is ONLY available in Dygraph mode**

        Get the Gradient of Current Variable

        Returns:
            ndarray or tuple of ndarray: if Variable's type is LoDTensor, return numpy value of the gradient of current Variable, if Variable's type is SelectedRows, return tuple of ndarray, first element of tuple is numpy value of the gradient of current Variable, second element of tuple is numpy value of the rows of current Variable.

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid
                import numpy as np

                # example1: return ndarray
                x = np.ones([2, 2], np.float32)
                with fluid.dygraph.guard():
                    inputs2 = []
                    for _ in range(10):
                        tmp = fluid.dygraph.base.to_variable(x)
                        tmp.stop_gradient=False
                        inputs2.append(tmp)
                    ret2 = fluid.layers.sums(inputs2)
                    loss2 = fluid.layers.reduce_sum(ret2)
                    loss2.backward()
                    print(loss2.gradient())

                # example2: return tuple of ndarray
                with fluid.dygraph.guard():
                    embedding = fluid.dygraph.Embedding(
                        size=[20, 32],
                        param_attr='emb.w',
                        is_sparse=True)
                    x_data = np.arange(12).reshape(4, 3).astype('int64')
                    x_data = x_data.reshape((-1, 3, 1))
                    x = fluid.dygraph.base.to_variable(x_data)
                    out = embedding(x)
                    out.backward()
                    print(embedding.weight.gradient())

        """
        pass

    @fake_interface_only
    def clear_gradient(self):
        """
        **Notes**:
            **1. This API is ONLY available in Dygraph mode**

            **2. Use it only Variable has gradient, normally we use this for Parameters since other temporal Variable will be deleted by Python's GC**

        Clear  (set to ``0`` ) the Gradient of Current Variable

        Returns:  None

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid
                import numpy as np

                x = np.ones([2, 2], np.float32)
                with fluid.dygraph.guard():
                    inputs2 = []
                    for _ in range(10):
                        tmp = fluid.dygraph.base.to_variable(x)
                        tmp.stop_gradient=False
                        inputs2.append(tmp)
                    ret2 = fluid.layers.sums(inputs2)
                    loss2 = fluid.layers.reduce_sum(ret2)
                    loss2.backward()
                    print(loss2.gradient())
                    loss2.clear_gradient()
                    print("After clear {}".format(loss2.gradient()))

        """
        pass

    def __str__(self):
        return self._to_readable_code()

    def _to_readable_code(self):
        """
        Get readable debug string of Variable.

        .. note::
            If you want to get the debug string in protobuf format,
            please use :code:`to_string` method.

        Returns:
            string: The formatted Variable string.

        Examples:
            .. code-block:: python

                import paddle
                import paddle.static as static

                paddle.enable_static()

                cur_program = static.Program()
                cur_block = cur_program.current_block()
                new_variable = cur_block.create_var(name="X",
                                                    shape=[-1, 23, 48],
                                                    dtype='float32')
                print(new_variable._to_readable_code())
        """
        if self.type == core.VarDesc.VarType.SELECTED_ROWS or self.type == core.VarDesc.VarType.LOD_TENSOR:
            var_str = "{name} : paddle.{type}.shape{shape}.astype({dtype})". \
                format(i="{", e="}", name=self.name, type=self.type, shape=self.shape, dtype=self.dtype)
        else:
            var_str = "{name} : paddle.{type})".\
                format(i="{", e="}", name=self.name, type=self.type)

        if type(self) == Parameter:
            if self.trainable:
                var_str = "trainable param " + var_str
            else:
                var_str = "param " + var_str
        else:
            var_str = "var " + var_str

        if self.persistable:
            var_str = "persist " + var_str

        return var_str

    def to_string(self, throw_on_error, with_details=False):
        """
        Get debug string.

        Args:

            throw_on_error (bool): True if raise an exception when self is not initialized.

            with_details (bool): more details about variables and parameters (e.g. trainable, optimize_attr, ...) will be printed when with_details is True. Default value is False;

        Returns:
            str: The debug string.

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid
                import paddle

                paddle.enable_static()
                cur_program = fluid.Program()
                cur_block = cur_program.current_block()
                new_variable = cur_block.create_var(name="X",
                                                    shape=[-1, 23, 48],
                                                    dtype='float32')
                print(new_variable.to_string(True))
                print("=============with detail===============")
                print(new_variable.to_string(True, True))
        """
        assert isinstance(throw_on_error, bool) and isinstance(with_details,
                                                               bool)
        protostr = self.desc.serialize_to_string()
        proto = framework_pb2.VarDesc.FromString(six.binary_type(protostr))
        res_str = _debug_string_(proto, throw_on_error)
        if with_details:
            additional_attr = ("error_clip", "stop_gradient")
            for attr_name in additional_attr:
                res_str += "%s: %s\n" % (attr_name,
                                         cpt.to_text(getattr(self, attr_name)))

        return res_str

    __repr__ = __str__

    @property
    def stop_gradient(self):
        """
        Indicating if we stop gradient from current Variable

        **Notes: This Property has default value as** ``True`` **in** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **mode, while Parameter's default value is False. However, in Static Graph Mode all Variable's default stop_gradient value is** ``False``

        Examples:
          .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np

            with fluid.dygraph.guard():
                value0 = np.arange(26).reshape(2, 13).astype("float32")
                value1 = np.arange(6).reshape(2, 3).astype("float32")
                value2 = np.arange(10).reshape(2, 5).astype("float32")
                linear = fluid.Linear(13, 5, dtype="float32")
                linear2 = fluid.Linear(3, 3, dtype="float32")
                a = fluid.dygraph.to_variable(value0)
                b = fluid.dygraph.to_variable(value1)
                c = fluid.dygraph.to_variable(value2)
                out1 = linear(a)
                out2 = linear2(b)
                out1.stop_gradient = True
                out = fluid.layers.concat(input=[out1, out2, c], axis=1)
                out.backward()

                assert linear.weight.gradient() is None
                assert (out1.gradient() == 0).all()
        """
        return self._stop_gradient

    @stop_gradient.setter
    def stop_gradient(self, s):
        self._stop_gradient = s

    @property
    def persistable(self):
        """
        Indicating if we current Variable should be long-term alive


        **Notes: This Property will be deprecated and this API is just to help user understand concept**

            **1. All Variable's persistable is** ``False`` **except Parameters.**

            **2. In** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **mode, this property should not be changed**

        Examples:
          .. code-block:: python

            import paddle.fluid as fluid
            cur_program = fluid.Program()
            cur_block = cur_program.current_block()
            new_variable = cur_block.create_var(name="X",
                                                shape=[-1, 23, 48],
                                                dtype='float32')
            print("persistable of current Var is: {}".format(new_variable.persistable))
        """
        return self.desc.persistable()

    @persistable.setter
    def persistable(self, p):
        self.desc.set_persistable(p)

    @property
    def name(self):
        """
        Indicating name of current Variable

        **Notes: If it has two or more Varaible share the same name in the same** :ref:`api_guide_Block_en` **, it means these Variable will share content in no-** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **mode. This is how we achieve Parameter sharing**

        Examples:
          .. code-block:: python

            import paddle.fluid as fluid
            cur_program = fluid.Program()
            cur_block = cur_program.current_block()
            new_variable = cur_block.create_var(name="X",
                                                shape=[-1, 23, 48],
                                                dtype='float32')
            print("name of current Var is: {}".format(new_variable.name))
        """
        return cpt.to_text(self.desc.name())

    @property
    def grad_name(self):
        """
        Indicating name of the gradient Variable of current Variable.

        **Notes: This is a read-only property. It simply returns name of
          gradient Variable from a naming convention but doesn't guarantee
          the gradient exists.**

        Examples:
          .. code-block:: python

          import paddle.fluid as fluid

          x = fluid.data(name="x", shape=[-1, 23, 48], dtype='float32')
          print(x.grad_name) # output is "x@GRAD"

        """
        return self.name + "@GRAD"

    @name.setter
    def name(self, new_name):
        self.desc.set_name(new_name)

    @property
    def shape(self):
        """
        Indicating shape of current Variable

        **Notes: This is a read-only property**

        Examples:
          .. code-block:: python

            import paddle.fluid as fluid
            cur_program = fluid.Program()
            cur_block = cur_program.current_block()
            new_variable = cur_block.create_var(name="X",
                                                shape=[-1, 23, 48],
                                                dtype='float32')
            print("shape of current Var is: {}".format(new_variable.shape))

        """
        # convert to tuple, make it as same as numpy API.
        return tuple(self.desc.shape())

    @property
    def dtype(self):
        """
        Indicating data type of current Variable

        **Notes: This is a read-only property**

        Examples:
          .. code-block:: python

            import paddle.fluid as fluid
            cur_program = fluid.Program()
            cur_block = cur_program.current_block()
            new_variable = cur_block.create_var(name="X",
                                                shape=[-1, 23, 48],
                                                dtype='float32')
            print("Dtype of current Var is: {}".format(new_variable.dtype))
        """
        return self.desc.dtype()

    @property
    def lod_level(self):
        """
        Indicating ``LoD`` info of current Variable, please refer to  :ref:`api_fluid_LoDTensor_en` to check the meaning
        of ``LoD``

        **Notes**:

            **1. This is a read-only property**

            **2. Don't support this property in** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **mode, it's value should be** ``0(int)``

        Examples:
          .. code-block:: python

            import paddle.fluid as fluid
            cur_program = fluid.Program()
            cur_block = cur_program.current_block()
            new_variable = cur_block.create_var(name="X",
                                                shape=[-1, 23, 48],
                                                dtype='float32')
            print("LoD Level of current Var is: {}".format(new_variable.lod_level))
        """
        if self.type == core.VarDesc.VarType.SELECTED_ROWS:
            raise Exception("SelectedRows DO NOT supprt lod")

        return self.desc.lod_level()

    @property
    def type(self):
        """
        Indicating Type of current Variable

        **Notes: This is a read-only property**

        Examples:
          .. code-block:: python

            import paddle.fluid as fluid
            cur_program = fluid.Program()
            cur_block = cur_program.current_block()
            new_variable = cur_block.create_var(name="X",
                                                shape=[-1, 23, 48],
                                                dtype='float32')
            print("Type of current Var is: {}".format(new_variable.type))
        """
        return self.desc.type()

    def clone(self):
        """
        Returns a new static Variable, which is the clone of the original static
        Variable. It remains in the current graph, that is, the cloned Variable 
        provides gradient propagation. Calling ``out = tensor.clone()`` is same
        as ``out = assign(tensor)`` .

        Returns:
            Variable: The cloned Variable.

        Examples:
            .. code-block:: python

                import paddle

                paddle.enable_static()

                # create a static Variable
                x = paddle.static.data(name='x', shape=[3, 2, 1])
                # create a cloned Variable
                y = x.clone()

        """
        output = self.block.create_var(
            name=unique_name.generate_with_ignorable_key(self.name + "_clone"),
            dtype=self.dtype,
            type=self.type,
            persistable=self.persistable,
            stop_gradient=self.stop_gradient)

        self.block.append_op(
            type='assign', inputs={'X': [self]}, outputs={'Out': [output]})
        return output

    def _set_error_clip(self, error_clip):
        """
        Set the error_clip.

        Args:
            error_clip(BaseErrorClipAttr) : The new error_clip.

        Returns:
            None
        """
        self.error_clip = error_clip

    def _set_info(self, key, value):
        """
        Set key-value information for this variable.

        Args:
            key(str): Key for this information.
            value(object): The value associated to the key.

        Returns: 
            None
        """
        if not hasattr(self, "_info"):
            self._info = {}
        self._info[key] = value

    def _get_info(self, key):
        """
        Get the information of this variable corresponding to key.

        Args:
            key(str): Key for this information.

        Returns: 
            object
        """
        if hasattr(self, "_info") and key in self._info:
            return self._info[key]
        return None

    def _slice_indices(self, slice, length):
        """
        Reference implementation for the slice.indices method.
        """
        # Compute step and length as integers.
        step = 1 if slice.step is None else slice.step

        # Raise ValueError for negative length or zero step.
        if length < 0:
            raise ValueError("length should not be negative")
        if step == 0:
            raise ValueError("slice step can not be zero")

        # Find lower and upper bounds for start and stop.
        lower = -1 if step < 0 else 0
        upper = length - 1 if step < 0 else length

        # Compute start.
        if slice.start is None:
            start = upper if step < 0 else lower
        else:
            start = slice.start
            start = max(start + length, lower) if start < 0 else min(start,
                                                                     upper)

        # Compute stop.
        if slice.stop is None:
            stop = lower if step < 0 else upper
        else:
            stop = slice.stop
            stop = max(stop + length, lower) if stop < 0 else min(stop, upper)

        return start, stop, step

    def _detectEllipsis(self, item):
        has_ellipsis = False
        start = 0
        end = len(self.shape)
        for index, o in enumerate(item):
            if o is Ellipsis:
                if has_ellipsis:
                    raise ValueError("Index can have one ellipsis only.")
                has_ellipsis = True
                start = index
            else:
                if has_ellipsis:
                    end = index
        return has_ellipsis, start, end

    def _reconstructSliceinfo(self, item):
        has_ellipsis, start, end = self._detectEllipsis(item)
        if has_ellipsis:
            newitem = []
            for i in range(start):
                newitem.append(item[i])
            for i in range(start, end):
                newitem.append(slice(None, None, None))
            for i in range(end, len(item)):
                newitem.append(item[i])
            return newitem
        else:
            return None

    def _detectContinuesSlice(self, item):
        starts = []
        ends = []
        for index, o in enumerate(item):
            if isinstance(o, int):
                start = int(o)
                if (index > 0 and index >= self.shape[index]) \
                        or (index < 0 and (index + self.shape[index]) < 0):
                    raise IndexError("invalid index")
                start = max(start + self.shape[index], 0) if start < 0 else min(
                    start, self.shape[index])
                starts.append(start)
                ends.append(start + 1)
            elif isinstance(o, slice):
                start, stop, step = self._slice_indices(o, self.shape[index])
                if step == 1 or step == -1:
                    starts.append(start)
                    ends.append(stop)
                else:
                    return False, None
            else:
                raise IndexError("Valid index accept int or slice or ellipsis")
        return True, [starts, ends]

    def _cloneVar(self, copy=False):
        if not copy:
            return self.block.create_var(
                name=unique_name.generate_with_ignorable_key(self.name),
                dtype=self.dtype)
        else:
            return self

    def _sliceVar(self, axes, starts, ends):
        new_var = self._cloneVar()
        self.block.append_op(
            type="slice",
            inputs={'Input': [self]},
            outputs={'Out': [new_var]},
            attrs={'axes': axes,
                   'starts': starts,
                   'ends': ends})
        return new_var

    def _concatVar(self, inputs, axis):
        new_var = self._cloneVar()
        self.block.append_op(
            type="concat",
            inputs={'X': inputs},
            outputs={'Out': [new_var]},
            attrs={'axis': axis, })
        return new_var

    def _sliceAndConcatVar(self, item, axis):
        if isinstance(item, slice):
            if self.shape[axis] < 0:
                return self._cloneVar(True)
            start, stop, step = self._slice_indices(item, self.shape[axis])
            if step == 1:
                return self._sliceVar([axis], [start], [stop])
            else:
                vars = []
                if step > 0:
                    while start < stop:
                        vars.append(
                            self._sliceVar([axis], [start], [start + 1]))
                        start += step
                else:
                    while start > stop:
                        vars.append(
                            self._sliceVar([axis], [start], [start + 1]))
                        start += step
                return self._concatVar(vars, axis)
        elif isinstance(item, int):
            if self.shape[axis] < 0:
                return self._cloneVar(True)
            index = int(item)
            if (index > 0 and index >= self.shape[axis]) \
                    or (index < 0 and (index + self.shape[axis]) < 0):
                raise IndexError("invalid index")
            return self._sliceVar([axis], [index], [index + 1])
        else:
            raise IndexError("Valid index accept int or slice or tuple")

    def __getitem__(self, item):
        return _getitem_impl_(self, item)

    def __setitem__(self, item, value):
        inputs = {'Input': self}

        # 1. Parse item
        if not isinstance(item, tuple):
            item = [item]

        axes = []
        starts = []
        ends = []
        max_integer = sys.maxsize
        for dim, slice_item in enumerate(item):
            if isinstance(slice_item, slice):
                start = slice_item.start
                end = slice_item.stop
                step = slice_item.step

                if start is None and end is None and step is None:
                    continue

                start = 0 if start is None else start
                step = 1 if step is None else step

                # TODO: support cases when step != 1
                if step != 1:
                    raise ValueError(
                        "When assign a value to a paddle.Tensor, only support step is 1, "
                        "but received step is {}.".format(step))
                end = max_integer if end is None else end
            else:
                start = slice_item
                end = slice_item + 1 if slice_item != -1 else max_integer
            axes.append(dim)
            starts.append(start)
            ends.append(end)

        attrs = {'axes': axes, 'starts': starts, 'ends': ends}

        # 2. Parse value
        dtype = self.dtype
        attrs['dtype'] = dtype

        from .data_feeder import convert_dtype
        #  2.1 value is an integer of float
        if isinstance(value, (int, float)):
            value = np.array([value]).astype(convert_dtype(dtype))

        #  2.2 value is a np.ndarray
        if isinstance(value, np.ndarray):
            shape = list(value.shape)
            if dtype == core.VarDesc.VarType.BOOL:
                value_name = "bool_values"
                values = [bool(v) for v in value.flat]
            elif dtype == core.VarDesc.VarType.FP32:
                value_name = "fp32_values"
                values = [float(v) for v in value.flat]
            elif dtype == core.VarDesc.VarType.FP64:
                value_name = "fp64_values"
                values = [float(v) for v in value.flat]
            elif dtype == core.VarDesc.VarType.INT32:
                value_name = "int32_values"
                values = [int(v) for v in value.flat]
            elif dtype == core.VarDesc.VarType.INT64:
                value_name = "int64_values"
                values = [int(v) for v in value.flat]
            else:
                raise TypeError(
                    "When assign a numpy.ndarray, integer or float to a paddle.Tensor, "
                    "the data type of the paddle.Tensor must be bool, float32, int32 or int64, but "
                    "received %s." % convert_dtype(dtype))
            attrs[value_name] = values
            attrs["shape"] = shape

        elif isinstance(value, Variable):
            inputs["ValueTensor"] = value
        else:
            raise TypeError(
                "Only support to assign an integer, float, numpy.ndarray or "
                "paddle.Tensor to a paddle.Tensor, but received {}".format(
                    type(value)))

        self.block.append_op(
            type="set_value", inputs=inputs, outputs={'Out': self}, attrs=attrs)
        return self


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

    def update_op_proto(self):
        op_protos = get_all_op_protos()
        for proto in op_protos:
            if proto.type not in self.op_proto_map:
                self.op_proto_map[proto.type] = proto

    @staticmethod
    def generated_op_attr_names():
        return {
            core.op_proto_and_checker_maker.kOpRoleAttrName(),
            core.op_proto_and_checker_maker.kOpRoleVarAttrName(),
            core.op_proto_and_checker_maker.kOpNameScopeAttrName(),
            core.op_proto_and_checker_maker.kOpCreationCallstackAttrName(),
            core.op_proto_and_checker_maker.kOpDeviceAttrName()
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

            import paddle.fluid as fluid
            cur_program = fluid.Program()
            cur_block = cur_program.current_block()
            # var1 += var2 + var3
            cur_block.append_op(type="sum",
                                inputs={"X": [var1, var2, var3]},
                                outputs={"Out": [var1]})
    """
    OP_WITHOUT_KERNEL_SET = {
        'feed', 'fetch', 'recurrent', 'go', 'rnn_memory_helper_grad',
        'conditional_block', 'while', 'send', 'recv', 'listen_and_serv',
        'fl_listen_and_serv', 'ncclInit', 'select', 'checkpoint_notify',
        'gen_nccl_id', 'c_gen_nccl_id', 'c_comm_init', 'c_sync_calc_stream',
        'c_sync_comm_stream', 'queue_generator', 'dequeue', 'enqueue',
        'heter_listen_and_serv'
    }

    def __init__(self,
                 block,
                 desc,
                 type=None,
                 inputs=None,
                 outputs=None,
                 attrs=None):
        if in_dygraph_mode():
            if type is None:
                raise ValueError(
                    "`type` to initialized an Operator can not be None.")
            self._type = type
            self.attrs = attrs if attrs else {}
        else:
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
                op_attrs[op_maker.kOpRoleAttrName(
                )] = self.block.program._op_role

            role_var_name = op_maker.kOpRoleVarAttrName()
            if len(self.block.program.
                   _op_role_var) != 0 and role_var_name not in op_attrs:
                op_attrs[role_var_name] = self.block.program._op_role_var

            if role_var_name in op_attrs and len(op_attrs[role_var_name]) == 0:
                del op_attrs[role_var_name]

            if len(self.desc.type()) != 0:
                return
            if type is None:
                raise ValueError(
                    "`type` to initialized an Operator can not be None.")
            else:
                callstack_var_name = op_maker.kOpCreationCallstackAttrName()
                op_attrs[callstack_var_name] = []
                for frame in traceback.extract_stack():
                    op_attrs[callstack_var_name].append(
                        '  File "{}", line {}, in {}'.format(frame[0], frame[1],
                                                             frame[2]))
                    op_attrs[callstack_var_name].append('    {}'.format(frame[
                        3]))

            self.desc.set_type(type)
            proto = OpProtoHolder.instance().get_op_proto(type)

            namescope_var_name = op_maker.kOpNameScopeAttrName()
            op_attrs[namescope_var_name] = _full_name_scope()

            # set device for op with kernels, give warning for op without kernels
            # when force_cpu and device_guard are used at the same time, a warning will be given.
            # TODO(zhangting2020): when force_cpu is removed, clear warning below.
            if _current_device is not None:
                if self._has_kernel(type):
                    op_device = op_maker.kOpDeviceAttrName()
                    op_attrs[op_device] = _current_device
                else:
                    warnings.warn("The Op(%s) is not support to set device." %
                                  type)
                if 'force_cpu' in op_attrs:
                    if (type is 'less_than' and op_attrs['force_cpu'] != None
                        ) or op_attrs['force_cpu'] != False:
                        warnings.warn(
                            "The Attr(force_cpu) of Op(%s) will be deprecated in the future, "
                            "please use 'device_guard' instead. 'device_guard' has higher priority when they are "
                            "used at the same time." % type)

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
                        if not isinstance(in_args, (list, tuple)):
                            in_args = [in_args]
                        if not in_proto.duplicable and len(in_args) > 1:
                            raise ValueError(
                                "Input %s expects only one input, but %d are given."
                                % (in_proto.name, len(in_args)))
                        in_arg_names = []
                        for index, arg in enumerate(in_args):
                            if isinstance(arg, six.string_types):
                                in_arg_names.append(arg)
                            elif isinstance(arg, six.binary_type):
                                in_arg_names.append(arg.decode())
                            elif isinstance(arg, (Variable, core.VarBase)):
                                in_arg_names.append(cpt.to_text(arg.name))
                            else:
                                raise TypeError(
                                    "The type of '%s' in operator %s should be "
                                    "one of [basestring(), str, Varibale] in python2, "
                                    "or one of [str, bytes, Variable] in python3."
                                    "but received : %s" %
                                    (in_proto.name, type, arg))
                        self.desc.set_input(in_proto.name, in_arg_names)
                    else:
                        self.desc.set_input(in_proto.name, [])

            if outputs is not None:
                for m in proto.outputs:
                    if (m.name not in outputs) and m.dispensable:
                        continue
                    if not ((m.name in outputs) or m.dispensable):
                        raise ValueError(("Incorrect setting for output(s) of "
                                          "operator \"%s\", should set: [%s].")
                                         % (type, m.name))
                for out_proto in proto.outputs:
                    if out_proto.name not in outputs:
                        continue
                    out_args = outputs[out_proto.name]
                    if not isinstance(out_args, list):
                        out_args = [out_args]
                    if not out_proto.duplicable and len(out_args) > 1:
                        raise ValueError(
                            "Output %s expects only one output, but %d are given."
                            % (out_proto.name, len(out_args)))
                    out_arg_names = []
                    for arg in out_args:
                        if isinstance(arg, six.string_types):
                            out_arg_names.append(arg)
                        else:
                            out_arg_names.append(cpt.to_text(arg.name))
                        # TODO(minqiyang): could we remove variable's op in static mode?
                        if not in_dygraph_mode():
                            if isinstance(arg, six.string_types):
                                block.var(arg).op = self
                            else:
                                arg.op = self
                    self.desc.set_output(out_proto.name, out_arg_names)

            if op_attrs is not None:
                if not isinstance(op_attrs, dict):
                    raise TypeError("'attrs' should be a dict.")
                for attr in proto.attrs:
                    attr_name = attr.name
                    if (attr_name not in op_attrs) or (
                            op_attrs[attr_name] is None):
                        continue
                    attr_val = op_attrs[attr_name]
                    self._update_desc_attr(attr_name, attr_val)

            self.desc.check_attrs()
            if self._has_kernel(type):
                self.desc.infer_var_type(self.block.desc)
                self.desc.infer_shape(self.block.desc)

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

    def _to_readable_code(self, skip_op_callstack=True):
        """
        Get readable debug string of Operator.

        .. note::
            If you want to get the debug string in protobuf format,
            please use :code:`to_string` method.

        Args:
            skip_op_callstack(bool): whether to skip parsing Operator's attribute
                op_callstack, default value is True

        Returns:
            string: The formatted Operator string.

        Examples:
            .. code-block:: python

            import paddle.fluid as fluid

            cur_program = fluid.Program()
            cur_block = cur_program.current_block()
            var = cur_block.create_var(name="X",
                                       shape=[-1, 23, 48],
                                       dtype='float32')
            new_op = cur_block.append_op(type="abs",
                                inputs={"X": [var]},
                                outputs={"Out": [var]})
            print(new_op._to_readable_code())
        """
        assert isinstance(
            skip_op_callstack, bool
        ), "skip_op_callstack parameter's type is error, expect bool, received %s".format(
            type(skip_op_callstack))
        outputs_str = "{"
        for i in range(0, len(self.output_names)):
            outputs_str += "{name}=".format(name=self.output_names[i])
            o = self.output(self.output_names[i])
            outputs_str += "{value}".format(value=o)
            if i != len(self.output_names) - 1:
                outputs_str += ", "
        outputs_str += "}"

        inputs_str = "{"
        for i in range(0, len(self.input_names)):
            inputs_str += "{name}=".format(name=self.input_names[i])
            o = self.input(self.input_names[i])
            inputs_str += "{value}".format(value=o)

            if i != len(self.input_names) - 1:
                inputs_str += ", "
        inputs_str += "}"

        attr_names = sorted(self.attr_names)
        attrs_str = ""
        for i in range(0, len(attr_names)):
            name = attr_names[i]
            if skip_op_callstack and name == "op_callstack":
                continue

            attr_type = self.desc.attr_type(name)
            if attr_type == core.AttrType.BLOCK:
                a = "{name} = block[{value}]".format(
                    name=name, type=attr_type, value=self._block_attr_id(name))
                attrs_str += a
                if i != len(attr_names) - 1:
                    attrs_str += ", "
                continue

            if attr_type == core.AttrType.BLOCKS:
                a = "{name} = blocks{value}".format(
                    name=name,
                    type=attr_type,
                    value=self._blocks_attr_ids(name))
                attrs_str += a
                if i != len(attr_names) - 1:
                    attrs_str += ", "
                continue

            a = "{name} = {value}".format(
                name=name, type=attr_type, value=self.desc.attr(name))
            attrs_str += a
            if i != len(attr_names) - 1:
                attrs_str += ", "

        if outputs_str != "{}":
            op_str = "{outputs} = {op_type}(inputs={inputs}, {attrs})".\
                format(outputs=outputs_str, op_type=self.type,
                       inputs=inputs_str, attrs=attrs_str)
        else:
            op_str = "{op_type}(inputs={inputs}, {attrs})".\
                format(op_type=self.type, inputs=inputs_str, attrs=attrs_str)
        return op_str

    def __str__(self):
        return self._to_readable_code()

    __repr__ = __str__

    @property
    def type(self):
        return self.desc.type()

    def input(self, name):
        r"""
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
        r"""
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

    def _remove_attr(self, name):
        self.desc.remove_attr(name)

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

    def _is_optimize_op(self):
        op_maker = core.op_proto_and_checker_maker
        OPTIMIZE = core.op_proto_and_checker_maker.OpRole.Optimize

        if not self.desc.has_attr(op_maker.kOpRoleAttrName()):
            return False

        op_role = self.desc.attr(op_maker.kOpRoleAttrName())
        if op_role & int(OPTIMIZE):
            return True

        return False

    def _is_backward_op(self):
        op_maker = core.op_proto_and_checker_maker
        BACKWARD = core.op_proto_and_checker_maker.OpRole.Backward

        if not self.desc.has_attr(op_maker.kOpRoleAttrName()):
            return False

        op_role = self.desc.attr(op_maker.kOpRoleAttrName())
        if op_role & int(BACKWARD):
            return True

        return False


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

            import paddle.fluid as fluid

            cur_program = fluid.Program()
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
        return self._to_readable_code()

    def _to_readable_code(self, skip_op_callstack=True):
        """
        Get readable debug string of Block.

        .. note::
            If you want to get the debug string in protobuf format,
            please use :code:`to_string` method.

        Args:
            skip_op_callstack(bool): whether to skip parsing Operator's attribute
                op_callstack, default value is True

        Returns:
            string: The formatted Block string.

        Examples:
            .. code-block:: python

            import paddle.fluid as fluid

            cur_program = fluid.Program()
            cur_block = cur_program.current_block()
            new_var = cur_block.create_var(name="X",
                                           shape=[-1, 23, 48],
                                           dtype='float32')
            new_op = cur_block.append_op(type="abs",
                                inputs={"X": [new_var]},
                                outputs={"Out": [new_var]})
            print(cur_block._to_readable_code())
        """
        assert isinstance(
            skip_op_callstack, bool
        ), "skip_op_callstack parameter's type is error, expect bool, received %s".format(
            type(skip_op_callstack))
        block_str = "{ // block "
        block_str += "{}\n".format(self.idx)
        for var in list(self.vars.values()):
            block_str += "    {}\n".format(var._to_readable_code())
        block_str += "\n"
        for op in self.ops:
            block_str += "    {}\n".format(
                op._to_readable_code(skip_op_callstack))
        block_str += "}"
        return block_str

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
    def backward_block_idx(self):
        cur_block_idx = self.idx
        for block in self.program.blocks:
            if block.forward_block_idx == cur_block_idx:
                return block.idx
        return -1

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
        if in_dygraph_mode():
            var = _varbase_creator(*args, **kwargs)
        else:
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
            if in_dygraph_mode():
                var = ParamBase(
                    d.shape(),
                    d.dtype(),
                    type=orig_var_type,
                    name=new_name,
                    stop_gradient=stop_gradient,
                    trainable=trainable,
                    optimize_attr=optimize_attr,
                    regularizer=regularizer,
                    error_clip=error_clip)
            else:
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

    def _remove_var(self, name, sync=True):
        if sync == True:
            self._sync_with_cpp()
        self.desc._remove_var(cpt.to_bytes(name))
        del self.vars[name]

    def create_parameter(self, *args, **kwargs):
        global_block = self.program.global_block()
        param = None
        if in_dygraph_mode():
            param = ParamBase(*args, **kwargs)
        else:
            param = Parameter(global_block, *args, **kwargs)
            # NOTE: Why only set stop_gradient=False in static mode
            # Because in dygraph mode, the `stop_gradient` and `trainable`
            # are related, and `trainable` default vallue is `True` or
            # it is specified by users, there is no need to set
            # `stop_gradient` for ParamBase here.
            param.stop_gradient = False
        if 'initializer' in kwargs:

            def _is_inited_by(block, var):
                init_ops = []
                for op in block.ops:
                    if var.name in op.output_arg_names:
                        # In startup_program, "c_broadcast" and "c_sync_comm_stream"
                        # are treated as initialization ops that cause error.
                        # Think of "c_broadcast" and "c_sync_comm_stream" as a special case here.
                        if op.type in ["c_broadcast", "c_sync_comm_stream"]:
                            continue
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
                # TODO already inited, do nothing, should log a warning
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
        if in_dygraph_mode():
            attrs = kwargs.get("attrs", {})
            type = kwargs.get("type", None)
            op = Operator(
                block=self,
                desc=None,
                type=type,
                inputs=None,
                outputs=None,
                attrs=attrs)

            # record ops in tracer rather than blocks
            #
            # TODO(minqiyang): add op stop_gradient support in static mode too.
            # currently, we only support stop_gradient in dygraph mode.

            _dygraph_tracer().trace_op(type,
                                       kwargs.get("inputs", {}),
                                       kwargs.get("outputs", {}), attrs
                                       if attrs else {},
                                       kwargs.get("stop_gradient", False))
        else:
            op_desc = self.desc.append_op()
            op = Operator(
                block=self,
                desc=op_desc,
                type=kwargs.get("type", None),
                inputs=kwargs.get("inputs", None),
                outputs=kwargs.get("outputs", None),
                attrs=kwargs.get("attrs", None))

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

    def _insert_op_without_sync(self, index, *args, **kwargs):
        """
        Insert an Operator according to the giving arguments, 
        without sync_with_cpp to meke the compilation faster.

        Args:
            index(int): the place that the operator to insert.

        Returns:
            Operator: the insert Operator.
        """
        op_desc = self.desc._insert_op(index)
        op = Operator(block=self, desc=op_desc, *args, **kwargs)
        self.ops.insert(index, op)
        return op

    def _remove_op(self, index, sync=True):
        """
        Remove the specific position operator.

        Args:
            index(int): the position that the operator to insert.

        Returns:
            None
        """
        if sync == True:
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
        if in_dygraph_mode():
            type = kwargs.get("type", None)
            attrs = kwargs.get("attrs", {})
            op = Operator(
                self, None, type=type, inputs=None, outputs=None, attrs=attrs)

            _dygraph_tracer().trace_op(type,
                                       kwargs.get("inputs", {}),
                                       kwargs.get("outputs", {}), attrs
                                       if attrs else {},
                                       kwargs.get("stop_gradient", False))
        else:
            op_desc = self.desc._prepend_op()
            op = Operator(
                self,
                op_desc,
                type=kwargs.get("type", None),
                inputs=kwargs.get("inputs", None),
                outputs=kwargs.get("outputs", None),
                attrs=kwargs.get("attrs", None))
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
                # if the Parameter is pruned, v may be None
                continue
            assert isinstance(v, Variable)
            new_p = None
            if in_dygraph_mode():
                new_p = ParamBase(
                    shape=v.shape,
                    dtype=v.dtype,
                    type=v.type,
                    lod_level=v.lod_level,
                    stop_gradient=p.stop_gradient,
                    trainable=p.trainable,
                    optimize_attr=p.optimize_attr,
                    regularizer=p.regularizer,
                    error_clip=p.error_clip,
                    name=v.name)
            else:
                new_p = Parameter(
                    block=self,
                    shape=v.shape,
                    dtype=v.dtype,
                    type=v.type,
                    lod_level=v.lod_level
                    if v.type == core.VarDesc.VarType.LOD_TENSOR else None,
                    stop_gradient=p.stop_gradient,
                    trainable=p.trainable,
                    optimize_attr=p.optimize_attr,
                    regularizer=p.regularizer,
                    error_clip=p.error_clip,
                    name=v.name)
            self.vars[new_p.name] = new_p

    def _clone_variable(self, var, force_persistable=True):
        """
        Clone a variable into current block.

        Args:
            var: the variable to be cloned.
            force_persistable(bool): True means setting the result variable to being persistable.
                                     False means setting the persistable the same with that of input var.
                                     default: True.

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
                persistable=True if force_persistable else var.persistable,
                is_data=var.is_data,
                need_check_feed=var.desc.need_check_feed())
        else:
            ret_var = self.create_var(
                name=var.name,
                shape=var.shape,
                dtype=var.dtype,
                type=var.type,
                lod_level=var.lod_level,
                persistable=True if force_persistable else var.persistable,
                is_data=var.is_data,
                need_check_feed=var.desc.need_check_feed())
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
            "The node variable description can not be None."
        self.node.var().set_shape(shape)

    def persistable(self):
        """
        If the variable node is a persistable variable, then return true.

        Returns:
            bool: indicate whether the variable is persistable.
        """
        assert self.node.var() is not None, \
            "The node variable description can not be None."
        return self.node.var().persistable()

    def type(self):
        """
        Return the variable type.

        Returns:
            core.VarDesc.VarType: the variable type.
        """
        assert self.node.var() is not None, \
            "The node variable description can not be None."
        return self.node.var().type()

    def dtype(self):
        """
        Return the variable data type.

        Returns:
            core.VarDesc.VarType: the variable data type.
        """
        assert self.node.var() is not None, \
            "The node variable description can not be None."
        return self.node.var().dtype()

    def shape(self):
        """
        Return the variable shape.

        Returns:
            list: the variable shape.
        """
        assert self.node.var() is not None, \
            "The node variable description can not be None."
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
            "The node operator description can not be None."
        self.node.op()._rename_input(old_input_name, new_input_name)

    def rename_output(self, old_output_name, new_output_name):
        """
        Rename the output of this node.

        Args:
            old_output_name(str): the old output name.
            new_output_name(str): the new output name.
        """
        assert self.node.op() is not None, \
            "The node operator description can not be None."
        self.node.op()._rename_output(old_output_name, new_output_name)

    def input(self, name):
        """
        Get the argument name list by the parameter name for input.

        Args:
            name(str): the parameter name.

        Returns:
            list(str): the argument name list.
        """
        assert self.node.op() is not None, \
            "The node operator description can not be None."
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
            "The node operator description can not be None."
        return self.node.op().output(name)

    def set_type(self, new_type):
        """
        Change the operator type into new type.

        Args:
            new_type(str): new operator type to be set.
        """
        assert self.node.op() is not None, \
            "The node operator description can not be None."
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
            "The node operator description can not be None."
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

    def input_arg_names(self):
        """
        Return input arguments' names of this op node.

        Returns:
            list(str): input arguments' names of this op node.
        """
        assert self.node.op() is not None, \
            "The node operator description can not be None."
        return self.node.op().input_arg_names()

    def output_arg_names(self):
        """
        Return output arguments' names of this op node.

        Returns:
            list(str): output arguments' names of this op node.
        """
        assert self.node.op() is not None, \
            "The node operator description can not be None."
        return self.node.op().output_arg_names()

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

    def clone(self):
        """
        Create a new and duplicated IrGraph.

        Warns:
            The method only clones the graph structure, not its attributes.

        Returns:
            IrGraph: A new and duplicated graph.
        """
        g = self.graph.clone()
        return IrGraph(g, self._for_test)

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

    def create_control_dep_var(self):
        """
        create a control var
        """
        return IrVarNode(self.graph.create_control_dep_var())

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
            outputs(dict): the outputs of the operator node.

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

    def update_output_link(self, old_output_node, new_output_node, op_node):
        """
        Update the output's link of an operator node.

        Args:
            old_output_node(IrNode): the old output node of the giving op_node.
            new_output_node(IrNode): the new output node of the giving op_node.
            op_node(IrOpNode): the operator node that is needed to update input's link.
        """
        assert old_output_node.node in self.graph.nodes() and new_output_node.node in \
            self.graph.nodes() and op_node.node in self.graph.nodes(), \
            'The three arguments(old_output_node &new_output_node &op_node) must be in the graph nodes.'
        old_output_node.remove_input(op_node)
        op_node.remove_output(old_output_node)
        new_output_node.append_input(op_node)
        op_node.append_output(new_output_node)
        op_node.rename_output(old_output_node.name(), new_output_node.name())

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

    def resolve_hazard(self):
        ordered_nodes = core.topology_sort(self.graph)
        var_nodes = dict()
        for node in ordered_nodes:
            if node.is_op() and node.op() is not None:
                for each_var_name in node.op().input_arg_names():
                    if each_var_name not in var_nodes:
                        var_nodes[each_var_name] = [
                            self._find_node_by_name(node.inputs, each_var_name)
                        ]
                for each_var_name in node.op().output_arg_names():
                    if each_var_name not in var_nodes:
                        var_nodes[each_var_name] = [
                            self._find_node_by_name(node.outputs, each_var_name)
                        ]
                    else:
                        var_nodes[each_var_name].append(
                            self._find_node_by_name(node.outputs,
                                                    each_var_name))
        self.graph.resolve_hazard(var_nodes)

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

        Notes: the `graph` can not contain a circle.

        Returns:
            list(IrNode): nodes in topology order.
        """
        ordered_nodes = core.topology_sort(self.graph)
        return [IrNode(n) for n in ordered_nodes]

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
            exited_code = subprocess.call(
                'dot -Tpdf ' + dot_file_path + ' -o ' + pdf_save_path,
                shell=True)
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
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        viz_dot_path = os.path.join(save_path, name) + '.dot'
        viz_pass = core.get_pass('graph_viz_pass')
        viz_pass.set('graph_viz_path', viz_dot_path)
        viz_pass.apply(self.graph)
        _convert_to_pdf(viz_dot_path)

    def to_program(self):
        """
        Convert the graph into a Program.

        WARN: When the graph includes backward operator nodes, the
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

    def _find_node_by_name(self, nodes, node_name):
        """
        Find a node in the giving nodes set by the name.
        """
        target_node = None
        for n in nodes:
            if n.name() == node_name:
                target_node = n
        assert target_node is not None, "Cannot find the target node in the giving set."
        return target_node

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
    Create Python Program.  It has at least one :ref:`api_guide_Block_en`, when the
    control flow op like conditional_block, while :ref:`api_paddle_fluid_layers_While` is included,
    it will contain nested block.

    Please reference the
    `framework.proto <https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/framework/framework.proto>`_
    for details.

    A set of Program usually contains startup program and main program.
    A startup program is set to contain some initial work, eg. initialize the ``Parameter``, and the main
    program will contain the network structure and vars for train.

    A set of Program can be used for test or train, in train program ,
    Paddle will contain all content to build a train network,  in test
    program Paddle will prune some content which is irrelevant to test, eg.
    backward ops and vars.

    **Notes**:
        **we have** :ref:`api_paddle_fluid_framework_default_startup_program` **and** :ref:`api_paddle_fluid_framework_default_main_program`
        **by default, a pair of them will shared the parameters. The** :ref:`api_paddle_fluid_framework_default_startup_program` **only run once to initialize parameters,**
        :ref:`api_paddle_fluid_framework_default_main_program` **run in every mini batch and adjust the weights.**

    Returns:
        Program: An empty Program.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.static as static

            paddle.enable_static()

            main_program = static.Program()
            startup_program = static.Program()
            with static.program_guard(main_program=main_program, startup_program=startup_program):
                x = static.data(name="x", shape=[-1, 784], dtype='float32')
                y = static.data(name="y", shape=[-1, 1], dtype='int32')
                z = static.nn.fc(name="fc", x=x, size=10, activation="relu")

            print("main program is: {}".format(main_program))
            print("start up program is: {}".format(startup_program))

    """

    def __init__(self):
        self.desc = core.ProgramDesc()
        self.blocks = [Block(self, 0)]
        self.current_block_idx = 0
        global global_prog_seed
        self._seed = global_prog_seed
        self._current_role = core.op_proto_and_checker_maker.OpRole.Forward
        self.__op_role_var = []

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

        # use Deep gradient comrepssion or not
        self._enable_dgc = False
        self._use_lamb = False

        self._nccl_comm_num = 1
        self._use_hierarchical_allreduce = False
        self._hierarchical_allreduce_inter_nranks = 0

        # if this program has been optimized by distributed optimizer
        # fleet_opt will be given a value
        self._fleet_opt = None
        self._program_config = None

        # assigned if this program has been parsed by a pipeline optimizer
        self._pipeline_opt = None

        # appending gradients times
        self._appending_grad_times = 0

        # identifier for auto checkpoint
        self._auto_checkpoint_name = unique_name.generate(
            "__auto_checkpoint_program__")

        # compiled program, i.e. Graph
        self._graph = None

    def global_seed(self, seed=0):
        """
        Set global seed for Program

        Returns:
            None.

        Examples:
            .. code-block:: python

                import paddle
                import paddle.static as static

                paddle.enable_static()

                prog = static.default_main_program()
                print(prog.random_seed)
                ## 0
                ## the default random seed is 0

                prog.global_seed(102)
                prog1 = static.default_main_program()
                print(prog1.random_seed)
                ## 102
                ## the random seed is 102
        """
        global global_prog_seed
        global_prog_seed = seed
        self._seed = global_prog_seed

    @property
    def _op_role(self):
        """
        The operator role. In a enum {Forward, Backward, Optimize}.

        Notes: this is a low level API. It is used only for ParallelExecutor to
        duplicate or schedule operator to devices.

        For example, the forward operator should be executed on every device.
        The backward operator should be executed on every device and the
        parameter gradient of backward (use :code:`_op_role_var` to get this
        variable) operator should be merged to one device. The optimization
        operators should be executed on only one device and broadcast the
        optimization result, i.e., the new parameter, to every other device.
        """
        return self._current_role

    @_op_role.setter
    def _op_role(self, role):
        self._current_role = role

    @property
    def _op_role_var(self):
        """
        The auxiliary variables for :code:`_op_role` property.

        See Also: :code:`Program._op_role`'s documentation for details.

        Notes: This is a very low-level API. Users should not use it directly.
        """
        return self.__op_role_var

    @signature_safe_contextmanager
    def _backward_role_guard(self):
        tmp_role = self._current_role

        OpRole = core.op_proto_and_checker_maker.OpRole
        self._current_role = OpRole.Backward
        try:
            yield
        finally:
            self._current_role = tmp_role

    @signature_safe_contextmanager
    def _optimized_guard(self, param_and_grads):
        """
        A with guard to set :code:`Optimization` :code:`OpRole` and
        :code:`OpRoleVar` automatically.

        Notes: This is a very low level API. Users should not use it directly.

        Args:
            param_and_grads(list): The variables (names) to be optimized.

        Examples:

            >>> import paddle.fluid as fluid
            >>> p, g = backward(...)
            >>> with program._optimized_guard([p,g]):
            >>>     p = p - 0.001 * g
        """
        tmp_role = self._current_role
        tmp_var = self.__op_role_var

        OpRole = core.op_proto_and_checker_maker.OpRole
        self._current_role = OpRole.Optimize
        self.__op_role_var = [
            var.name if isinstance(var, Variable) else var
            for var in param_and_grads
        ]
        try:
            yield
        finally:
            self.__op_role_var = tmp_var
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

            >>> import paddle.fluid as fluid
            >>> p, g = backward(...)
            >>> with program.lr_schedule_guard():
            >>>     lr = lr * decay
        """

        tmp_role = self._current_role
        tmp_var = self.__op_role_var

        OpRole = core.op_proto_and_checker_maker.OpRole
        self._current_role = OpRole.LRSched
        if is_with_opt:
            self._current_role = int(OpRole.LRSched) | int(OpRole.Optimize)
        # TODO(typhoonzero): how to set target learning rate var
        self.__op_role_var = []
        try:
            yield
        finally:
            self.__op_role_var = tmp_var
            self._current_role = tmp_role

    def __str__(self):
        """
        Get the protobuf debug string of this Program.

        Returns:
            (str): The protobuf debug string.

        Raises:
            ValueError: If any of required fields is not set.
        """
        return self._to_readable_code()

    def _to_readable_code(self, skip_op_callstack=True):
        """
        Get readable debug string of Program.

        .. note::
            If you want to get the debug string in protobuf format,
            please use :code:`to_string` method.

        Args:
            skip_op_callstack(bool): whether to skip parsing Operator's attribute
                op_callstack, default value is True

        Returns:
            string: The formatted Program string.

        Examples:
            .. code-block:: python

            import paddle
            import paddle.static as static

            paddle.enable_static()

            cur_program = static.Program()
            cur_block = cur_program.current_block()
            new_var = cur_block.create_var(name="X",
                                           shape=[-1, 23, 48],
                                           dtype='float32')
            new_op = cur_block.append_op(type="abs",
                                inputs={"X": [new_var]},
                                outputs={"Out": [new_var]})
            print(cur_program._to_readable_code())
        """
        assert isinstance(
            skip_op_callstack, bool
        ), "skip_op_callstack parameter's type is error, expect bool, received %s".format(
            type(skip_op_callstack))
        program_str = ""
        for block in self.blocks:
            program_str += block._to_readable_code(skip_op_callstack)
            program_str += '\n'
        return program_str

    def to_string(self, throw_on_error, with_details=False):
        """
        To debug string.

        Args:

            throw_on_error (bool): raise Value error when any of required fields is not set.

            with_details (bool): True if more details about variables and parameters, e.g., :code:`trainable`, :code:`optimize_attr`, need to print.

        Returns:
            str: The debug string describe current Program.

        Raises:
            ValueError: If any of required fields is not set and throw_on_error is True.

        Examples:
            .. code-block:: python

                import paddle
                import paddle.static as static

                paddle.enable_static()

                prog = static.default_main_program()
                x = static.data(name="X", shape=[2,3], dtype="float32")
                pred = static.nn.fc(x, size=3)
                prog_string = prog.to_string(throw_on_error=True, with_details=False)
                prog_string_with_details = prog.to_string(throw_on_error=False, with_details=True)
                print("program string without detail: {}".format(prog_string))
                print("program string with detail: {}".format(prog_string_with_details))
        """
        assert isinstance(
            throw_on_error, bool
        ), "The type of throw_on_error parameter is wrong, expected bool, but received {}.".format(
            type(throw_on_error))
        assert isinstance(
            with_details, bool
        ), "The type of with_details parameter is wrong, expected bool, but received {}.".format(
            type(with_details))

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
        .. note:::
            1. :code:`Program.clone()` method DOES NOT clone :ref:`api_paddle_io_DataLoader` . 
            2. Recommend you to use :code:`clone` before using :code:`Opimizer.minimize` . 
            3. This API has no effect in Dygraph Mode.

        Create a new Program with forward content of original one when ``for_test=True``.
        Create a new Program as same as the original one when ``for_test=False``.

        Some operators, e.g., :ref:`api_paddle_fluid_layers_batch_norm` , behave differently between
        training and testing. They have an attribute, :code:`is_test`, to
        control this behaviour. This method will change the :code:`is_test`
        attribute of them to :code:`True` when :code:`for_test=True`.

        * Set for_test to False when you want to clone the program for training.
        * Set for_test to True when you want to clone the program for testing.
          We will prune the backward and optimize part of the program when you
          use :code:`clone` after :code:`Opimizer.minimize`, but we still
          recommend you to use :code:`clone` before using :code:`Opimizer.minimize`.

        For Example:
          ::

            import paddle
            import paddle.static as static

            paddle.enable_static()

            img = static.data(name='image', shape=[None, 784])
            pred = static.nn.fc(x=img, size=10, actvation='relu')
            loss = paddle.mean(pred)
            # Here we use clone before Momentum
            test_program = static.default_main_program().clone(for_test=True)
            optimizer = paddle.optimizer.Momentum(learning_rate=0.01, momentum=0.9)
            optimizer.minimize(loss)

        Args:

            for_test (bool): True if change the :code:`is_test` attribute of operators to :code:`True`
                and prune the backward and optimize part of the program. The default value is :code:`False` .

        Returns:
            Program: A new Program with forward content of original one when ``for_test=True``.  A new Program as same as the original one when ``for_test=False``


        Examples:

            .. note::
                The Program's order maybe different after :code:`clone` and
                this will not affect your training or testing progress. In the following
                example we give you an simple method :code:`print_prog(program)` to
                print Program Descs inorder to make sure you have same print result
                after :code:`clone`:

            .. code-block:: python

                import six

                def print_prog(prog):
                    for name, value in sorted(six.iteritems(prog.block(0).vars)):
                        print(value)
                    for op in prog.block(0).ops:
                        print("op type is {}".format(op.type))
                        print("op inputs are {}".format(op.input_arg_names))
                        print("op outputs are {}".format(op.output_arg_names))
                        for key, value in sorted(six.iteritems(op.all_attrs())):
                            if key not in ['op_callstack', 'op_role_var']:
                                print(" [ attrs: {}:   {} ]".format(key, value))


            1. To clone a test program, the sample code is:
                .. code-block:: python

                    import six
                    import paddle
                    import paddle.static as static
                    import paddle.utils as utils
                    import paddle.nn.functional as F

                    paddle.enable_static()

                    def print_prog(prog):
                        for name, value in sorted(six.iteritems(prog.block(0).vars)):
                            print(value)
                        for op in prog.block(0).ops:
                            print("op type is {}".format(op.type))
                            print("op inputs are {}".format(op.input_arg_names))
                            print("op outputs are {}".format(op.output_arg_names))
                            for key, value in sorted(six.iteritems(op.all_attrs())):
                                if key not in ['op_callstack', 'op_role_var']:
                                    print(" [ attrs: {}:   {} ]".format(key, value))

                    train_program = static.Program()
                    startup_program = static.Program()

                    # startup_program is used to do some parameter init work,
                    # and main program is used to hold the network
                    with static.program_guard(train_program, startup_program):
                        with utils.unique_name.guard():
                            img = static.data(name='image', shape=[None, 784])
                            hidden = static.nn.fc(x=img, size=200, activation='relu')
                            hidden = F.dropout(hidden, p=0.5)
                            loss = F.cross_entropy(
                                input=static.nn.fc(x=hidden, size=10, activation='softmax'),
                                label=static.data(name='label', shape=[1], dtype='int64'))
                            avg_loss = paddle.mean(loss)
                            test_program = train_program.clone(for_test=True)
                    print_prog(test_program)

                    # Due to parameter sharing usage for train and test, so we need to use startup program of train
                    # instead of using test startup program, while nothing is in test's startup program

                    # In Paddle we will share weights by using the same Tensor name. In train and test program
                    # all parameters will have the same name and this can make train and test program sharing parameters,
                    # that's why we need to use startup program of train. And for startup program of test, it has nothing,
                    # since it is a new program.

                    with static.program_guard(train_program, startup_program):
                        with utils.unique_name.guard():
                            sgd = paddle.optimizer.SGD(learning_rate=1e-3)
                            sgd.minimize(avg_loss)


            2. The clone method can be avoid if you create program for training and program for testing individually.
                .. code-block:: python

                    import six
                    import paddle
                    import paddle.static as static
                    import paddle.utils as utils
                    import paddle.nn.functional as F

                    paddle.enable_static()

                    def print_prog(prog):
                        for name, value in sorted(six.iteritems(prog.block(0).vars)):
                            print(value)
                        for op in prog.block(0).ops:
                            print("op type is {}".format(op.type))
                            print("op inputs are {}".format(op.input_arg_names))
                            print("op outputs are {}".format(op.output_arg_names))
                            for key, value in sorted(six.iteritems(op.all_attrs())):
                                if key not in ['op_callstack', 'op_role_var']:
                                    print(" [ attrs: {}:   {} ]".format(key, value))

                    def network():
                        img = static.data(name='image', shape=[None, 784])
                        hidden = static.nn.fc(x=img, size=200, activation='relu')
                        hidden = F.dropout(hidden, p=0.5)
                        loss = F.cross_entropy(
                            input=static.nn.fc(x=hidden, size=10, activation='softmax'),
                            label=static.data(name='label', shape=[1], dtype='int64'))
                        avg_loss = paddle.mean(loss)
                        return avg_loss

                    train_program_2 = static.Program()
                    startup_program_2 = static.Program()
                    test_program_2 = static.Program()
                    with static.program_guard(train_program_2, startup_program_2):
                        with utils.unique_name.guard():
                            avg_loss = network()
                            sgd = paddle.optimizer.SGD(learning_rate=1e-3)
                            sgd.minimize(avg_loss)
                    # the test startup program is not used.
                    with static.program_guard(test_program_2, startup_program_2):
                        with utils.unique_name.guard():
                            avg_loss = network()
                    print_prog(test_program_2)

            The two code snippets above will generate and print same programs.
        """

        # NOTE(zhiqiu): we sync the original program first, since its program may diff with
        # its desc due to modifying desc in c++ space. E.g. save op will add kLookupTablePath in desc.
        self._sync_with_cpp()

        pruned_origin_block_id_map = None
        if for_test:
            forward_prog = Program()
            forward_prog.desc, pruned_origin_block_id_map = core.prune_backward(
                self.desc)
            forward_prog.blocks = [
                Block(forward_prog, i)
                for i in six.moves.range(forward_prog.desc.num_blocks())
            ]
            forward_prog._sync_with_cpp()
            p = forward_prog._inference_optimize(prune_read_op=False)
        else:
            p = Program()
            p.current_block_idx = self.current_block_idx
            p._seed = self._seed
            p.desc = core.ProgramDesc(self.desc)
            p.blocks = [
                Block(p, i) for i in six.moves.range(self.desc.num_blocks())
            ]

            p._current_role = self._current_role
            p.__op_role_var = self.__op_role_var
            p._appending_grad_times = self._appending_grad_times
            if hasattr(self, 'lr_sheduler'):
                p.lr_sheduler = self.lr_sheduler

            # NOTE(zhiqiu): we sync the cloned program, to update its program by
            # its desc.
            p._sync_with_cpp()

        p._copy_param_info_from(self)
        p._copy_data_info_from(self, pruned_origin_block_id_map)
        p._copy_dist_param_info_from(self)
        return p

    def _prune(self, targets):
        """
        Prune operators and variables which are not needed to generate
        :code:`targets`.

        Notes: This is a very low level API. Users should not use this API
        directly. This API is in flux and not stable.

        Args:
            targets(list|Variable|Operator): A list of variables, operators, or variable names
                need to be pruned

        Returns:
            Program:  A new, pruned program.
        """
        return self._prune_with_input([], targets)

    def _prune_with_input(self, feeded_var_names, targets):
        """
        Prune operators and variables which are not needed to generate
        :code:`targets`. Prune operators and variables which are needed 
        to generate feeded_var 

        Notes: This is a very low level API. Users should not use this API
        directly. This API is in flux and not stable.

        Args:
            feeded_var_names(list|str): A list of variable names from where
                pruning start. If it is set as [], this API works just like _prune()
            targets(list|Variable|Operator): A list of variables, operators, or variable names
                need to be pruned

        Returns:
            Program:  A new, pruned program.
        """

        # NOTE(zhiqiu): we sync the original program first, since its program may diff with
        # its desc due to modifying desc in c++ space. E.g. save op will add kLookupTablePath in desc.
        self._sync_with_cpp()

        if not isinstance(feeded_var_names, list):
            feeded_var_names = [feeded_var_names]
        if not isinstance(targets, list):
            targets = [targets]

        for var in feeded_var_names:
            if not isinstance(var, six.string_types):
                raise ValueError(
                    "All feeded_var_names of Program._prune_with_input() can only be "
                    "str, but received %s." % type(var))

        targets_idx = []
        for t in targets:
            if not isinstance(t, Operator):
                if isinstance(t, Variable):
                    name = t.name
                elif isinstance(t, six.string_types):
                    name = str(t)
                else:
                    raise ValueError(
                        "All targets of Program._prune_with_input() can only be "
                        "Variable or Operator, but received %s." % type(t))

                # NOTEZ(zhiqiu): For variable to be fed in fetch_list, there two cases:
                # (1) the variable is leaf, it has no op that generates it;
                # (2) the variable is not leaf, and we need to prune the op that generates it.
                # In both cases, wo can just skip target_op of that it.
                if name in feeded_var_names:
                    continue

                # After transpiler processing, the op that output this
                # variable maybe has been changed, so t.op is not reliable
                # and we need to find the current op that generate this
                # variable here.
                target_op = None
                global_block = self.global_block()
                for idx, op in enumerate(global_block.ops):
                    if name in op.output_arg_names:
                        # NOTE(zhiqiu): Find op that generate target name.
                        # Skip optimize op except for optimize op in targets,
                        # since optimize op generates parameters.
                        if op._is_optimize_op() and op not in targets:
                            continue
                        else:
                            target_op = op
                            break
                if target_op is None:
                    raise ValueError(
                        "The target variable used for pruning should have an "
                        "associated operator that generates it.")
                else:
                    targets_idx.append([target_op.block.idx, target_op.idx])
            else:
                targets_idx.append([t.block.idx, t.idx])

        res = Program()
        res.desc, pruned_origin_block_id_map = core.prune(self.desc,
                                                          set(feeded_var_names),
                                                          targets_idx)
        res.blocks = [
            Block(res, i) for i in six.moves.range(res.desc.num_blocks())
        ]
        res._sync_with_cpp()

        res._copy_param_info_from(self)
        res._copy_data_info_from(self, pruned_origin_block_id_map)
        res._copy_dist_param_info_from(self)

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
        .. note::
            1. All information about parameters will be lost after serialization; 
            2. This API has no effect in Dygraph mode.

        Deserialize a Program from  `protobuf <https://en.wikipedia.org/wiki/Protocol_Buffers>`_  binary string.
        This method always use to save and load model

        Args:

            binary_str_type (str): the binary prootbuf string.

        Returns:
            Program: A deserialized Program.

        Examples:
            .. code-block:: python

                import paddle
                import paddle.static as static

                paddle.enable_static()

                startup_prog = static.Program()
                main_prog = static.Program()
                with static.program_guard(startup_prog, main_prog):
                    x = static.data(name='X', shape=[1000, 784], dtype='float32')

                    y = static.data(name='Y', shape=[784, 100], dtype='float32')

                    z = paddle.matmul(x=x, y=y)

                    binary_str = static.default_main_program().desc.serialize_to_string()
                    prog_restored = static.default_main_program().parse_from_string(binary_str)

                    print(static.default_main_program())
                    print(prog_restored)
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
        The default random seed for random operators in Program. ``0`` means get
        the random seed from random device.

        .. note:: 
            It must be set before the operators have been added.

        Returns:
            int64: Random seed in current Program


        Examples:
            .. code-block:: python

                import paddle
                import paddle.static as static
                import paddle.nn.functional as F

                paddle.enable_static()

                prog = static.default_main_program()
                random_seed = prog.random_seed
                x_var = static.data(name="X", shape=[3,3], dtype="float32")
                print(random_seed)
                ## 0
                ## the default random seed is 0

                # Here we need to set random seed before we use paddle.nn.functional.dropout
                prog.random_seed = 1
                z_var = F.dropout(x_var, 0.7)

                print(prog.random_seed)
                ## 1
                ## the random seed is change to 1
        """
        return self._seed

    @property
    def num_blocks(self):
        """
        The number of :ref:`api_guide_Block_en`  in this Program.

        .. note:: 
            This API has no effect in Dygraph mode.

        Returns:
            int(Platform-dependent size): num of :ref:`api_guide_Block_en`  in current Program


        Examples:
            .. code-block:: python

                import paddle
                import paddle.static as static

                paddle.enable_static()

                prog = static.default_main_program()
                num_blocks = prog.num_blocks
                print(num_blocks)

                # print result:
                # 1
        """
        return self.desc.num_blocks()

    @random_seed.setter
    def random_seed(self, seed):
        if not isinstance(seed, int):
            raise ValueError(
                "Program.random_seed's input seed must be an integer, but received %s."
                % type(seed))
        self._seed = seed

    def __repr__(self):
        return self.__str__()

    def global_block(self):
        """
        .. note::
            This API has no effect in Dygraph mode.

        Get the first :ref:`api_guide_Block_en` of this Program.

        Returns:
            :ref:`api_guide_Block_en`: The first :ref:`api_guide_Block_en`  of this Program.


        Examples:
            .. code-block:: python

                import paddle
                import paddle.static as static

                paddle.enable_static()

                prog = static.default_main_program()
                gb_block = prog.global_block()
                print(gb_block)

        """
        return self.blocks[0]

    def block(self, index):
        """
        .. note::
            This API has no effect in Dygraph mode.

        Get the :code:`index`  :ref:`api_guide_Block_en`  of this Program

        Args:
            index (int) - The index of  :ref:`api_guide_Block_en`  to get

        Returns:
            :ref:`api_guide_Block_en`: The :code:`index` block

        Examples:
            .. code-block:: python

                import paddle
                import paddle.static as static

                paddle.enable_static()

                prog = static.default_main_program()
                block_0 = prog.block(0)
                print(block_0)
        """
        return self.blocks[index]

    def current_block(self):
        """
        .. note::
            This API has no effect in Dygraph mode.

        Get the current  :ref:`api_guide_Block_en` . The :code:`current`  :ref:`api_guide_Block_en`
        is the  :ref:`api_guide_Block_en`  to append operators.

        Returns:
             :ref:`api_guide_Block_en`: The :code:`index`  :ref:`api_guide_Block_en`

        Examples:
            .. code-block:: python

                import paddle
                import paddle.static as static

                paddle.enable_static()

                prog = static.default_main_program()
                current_blk = prog.current_block()
                print(current_blk)
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
            raise TypeError(
                "Function Program._copy_param_info_from() needs to pass in a source Program, but received %s"
                % type(other))

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
            raise TypeError(
                "Function Program._copy_param_info_from() needs to pass in a source Program, but received %s"
                % type(other))
        self._is_distributed = other._is_distributed
        self._is_chief = other._is_chief
        self._parameters_on_pservers = other._parameters_on_pservers
        self._endpoints = other._endpoints
        self._ps_endpoint = other._ps_endpoint
        self._distributed_lookup_table = other._distributed_lookup_table

    def _copy_data_info_from(self, other, pruned_origin_block_id_map=None):
        """
        Copy the information of data variables from other program.

        Notes: This is a very low level API. Users should not invoke it
        directly.

        Args:
            other(Program): Other program
            pruned_origin_block_id_map(dict{int:int}): A dict which maps the block id in program
            self to the block id in program other. For example, {0:0, 1:1, 2:3} means block 0 in self is 
            cloned from block 0 in other, etc. Default is None, which means default mapped, 
            {0:0, 1:1,..., n:n}.

        Returns:
            None
        """
        if not isinstance(other, Program):
            raise TypeError(
                "Function Program._copy_param_info_from() needs to pass in a source Program, but received %s"
                % type(other))

        if not pruned_origin_block_id_map:
            pruned_origin_block_id_map = {
                i: i
                for i in six.moves.range(self.desc.num_blocks())
            }

        # NOTE(zhiqiu): All vars in cloned program exist in original program.
        # The reverse is not true, due to backward pruning.
        for i, block in enumerate(self.blocks):
            other_block = other.blocks[pruned_origin_block_id_map[i]]
            for var in list(block.vars.values()):
                other_var = other_block.var(var.name)
                if other_var.is_data:
                    var.is_data = True
                if other_var.desc.need_check_feed():
                    var.desc.set_need_check_feed(True)
                if other_var.stop_gradient:
                    var.stop_gradient = True

    def list_vars(self):
        """
        Get all Tensors from this Program. A iterable object is returned.

        Returns:
            iterable Tensors: The Generator will yield every Tensor in this program.

        Examples:
            .. code-block:: python

                import paddle
                import paddle.static as static

                paddle.enable_static()

                prog = static.default_main_program()
                img = static.data(name='img', shape=[None, 1,28,28], dtype='float32')
                label = static.data(name='label', shape=[None,1], dtype='int64')
                for var in prog.list_vars():
                    print(var)

                # var img : paddle.VarType.LOD_TENSOR.shape(-1, 1, 28, 28).astype(VarType.FP32)
                # var label : paddle.VarType.LOD_TENSOR.shape(-1, 1).astype(VarType.INT64)
        """
        for each_block in self.blocks:
            for each_var in list(each_block.vars.values()):
                yield each_var

    def all_parameters(self):
        """
        Get all :ref:`api_guide_parameter_en` from this Program. A list object is returned.

        Returns:
            list[ :ref:`api_guide_parameter_en` ]: The list contians all parameters in this program.

        Examples:
            .. code-block:: python

                import paddle
                import paddle.static as static

                paddle.enable_static()

                program = static.default_main_program()
                data = static.data(name='x', shape=[None, 13], dtype='float32')
                hidden = static.nn.fc(x=data, size=10)
                loss = paddle.mean(hidden)
                paddle.optimizer.SGD(learning_rate=0.01).minimize(loss)

                for param in program.all_parameters():
                    print(param)

                # Here will print all parameters in current program, in this example,
                # the result is like:
                #
                # persist trainable param fc_0.w_0 : paddle.VarType.LOD_TENSOR.shape(13, 10).astype(VarType.FP32)
                # persist trainable param fc_0.b_0 : paddle.VarType.LOD_TENSOR.shape(10,).astype(VarType.FP32)
                #
                # Here print(param) will print out all the properties of a parameter,
                # including name, type and persistable, you can access to specific
                # property of a parameter, such as param.name, param.type
        """
        parameters = []
        for each_block in self.blocks:
            parameters.extend(each_block.all_parameters())
        return parameters


@six.add_metaclass(ParameterMetaClass)
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
        do_model_average(bool): True if the model average strategy will
            be applied on this parameter.
        need_clip (bool): Whether the parameter gradient need to be cliped 
            in optimizer. Default is True.
    """

    def __init__(self,
                 block,
                 shape,
                 dtype,
                 type=core.VarDesc.VarType.LOD_TENSOR,
                 **kwargs):
        if shape is None:
            raise ValueError("The shape of Parameter should not be None")
        if dtype is None:
            raise ValueError("The dtype of Parameter should not be None")

        if len(shape) == 0:
            raise ValueError(
                "The dimensions of shape for Parameter must be greater than 0")

        for each in shape:
            if each < 0:
                raise ValueError(
                    "Each dimension of shape for Parameter must be greater than 0, but received %s"
                    % list(shape))

        Variable.__init__(
            self,
            block,
            persistable=True,
            shape=shape,
            dtype=dtype,
            type=type,
            **kwargs)
        self.trainable = kwargs.get('trainable', True)

        self.optimize_attr = kwargs.get('optimize_attr', {'learning_rate': 1.0})

        self.regularizer = kwargs.get('regularizer', None)

        self.do_model_average = kwargs.get('do_model_average', None)

        self.need_clip = kwargs.get('need_clip', True)

        self.is_distributed = False

    def __str__(self):
        return self._to_readable_code()

    def to_string(self, throw_on_error, with_details=False):
        """
        To debug string.

        Args:
            throw_on_error(bool): raise exception when self is not initialized
                when throw_on_error is True
            with_details(bool): more details about variables and parameters
                (e.g. trainable, optimize_attr, ...) will be printed when with_details is True

        Returns(str): The debug string.

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid

                prog = fluid.default_main_program()
                rlt = fluid.layers.data("fake_data", shape=[1,1], dtype='float32')
                debug_str = prog.to_string(throw_on_error=True, with_details=False)
                print(debug_str)
        """
        assert isinstance(throw_on_error, bool) and isinstance(with_details,
                                                               bool)
        if with_details:
            res_str = Variable.to_string(self, throw_on_error, True)
            additional_attr = ("trainable", "optimize_attr", "regularizer",
                               "do_model_average", "need_clip")
            for attr_name in additional_attr:
                res_str += "%s: %s\n" % (attr_name,
                                         cpt.to_text(getattr(self, attr_name)))
        else:
            res_str = Variable.to_string(self, throw_on_error, False)
        return res_str

    __repr__ = __str__


class ParamBase(core.VarBase):
    """
    ParamBase is derived from Tensor( Which is the concept in Dygraph Mode). 
    A ParamBase is a persistable Tensor, and will be updated by optimizers 
    after each iteration.
    The training of a neural network is essentially the updating of
    its ParamBase.

    Relative to a general Tensor, a ParamBase has several its own
    member variables:

    Args:
        trainable(bool): True if the ParamBase need to be updated after
            iterations.
        optimize_attr(map): ParamBase attributes related with optimizing.
            Currently, it only contains 'learning_rate'.
            Default: {'learning_rate': 1.0}
        regularizer(WeightDecayRegularizer): The Regularizer which will
            be applied on the ParamBase. Default: None
        do_model_average(bool): True if the model average strategy will
            be applied on this ParamBase.
        need_clip (bool): Whether the parameter gradient need to be cliped 
            in optimizer. Default is True.
    """

    @dygraph_only
    def __init__(self, shape, dtype, **kwargs):
        if shape is None:
            raise ValueError("The shape of Parameter should not be None")
        if dtype is None:
            raise ValueError("The dtype of Parameter should not be None")

        if len(shape) == 0:
            raise ValueError(
                "The dimensions of shape for Parameter must be greater than 0")

        for each in shape:
            if each < 0:
                raise ValueError(
                    "Each dimension of shape for Parameter must be greater than 0, but received %s"
                    % list(shape))

        if dtype is not None:
            if not isinstance(dtype, core.VarDesc.VarType):
                dtype = convert_np_dtype_to_dtype_(dtype)

        name = kwargs.get('name', unique_name.generate('_param_base'))

        super(ParamBase, self).__init__(dtype
                                        if dtype else core.VarDesc.VarType.FP32,
                                        list(shape) if shape else [], name,
                                        core.VarDesc.VarType.LOD_TENSOR, True)

        trainable = kwargs.get('trainable', True)
        self.stop_gradient = not trainable

        self.optimize_attr = kwargs.get('optimize_attr', {'learning_rate': 1.0})

        self.regularizer = kwargs.get('regularizer', None)

        self.do_model_average = kwargs.get('do_model_average', None)

        self.need_clip = kwargs.get('need_clip', True)

        self.is_distributed = False
        # self.block = default_main_program().global_block()

    @property
    def trainable(self):
        return not self.stop_gradient

    @trainable.setter
    def trainable(self, trainable):
        if isinstance(trainable, bool):
            self.stop_gradient = not trainable
        else:
            raise ValueError(
                "The type of trainable MUST be bool, but the type is ",
                type(trainable))

    def __str__(self):
        """
        Convert a ParamBase object to a readable string.

        Returns(str): A readable string.

        Examples:
            .. code-block:: python

                import paddle
                linear = paddle.nn.Linear(3, 3)
                print(linear.weight)
                # Parameter containing:
                # Tensor(shape=[3, 3], dtype=float32, place=CUDAPlace(0), stop_gradient=False,
                #        [[ 0.48948765,  0.05829060, -0.25524026],
                #         [-0.70368278,  0.52986908, -0.68742192],
                #         [-0.54217887,  0.48439729,  0.34082305]])
        """
        return "Parameter containing:\n{tensor}".format(
            tensor=super(ParamBase, self).__str__())

    def __deepcopy__(self, memo):
        """
        Deep copy parameter, it will always performs Tensor copy.

        Examples:
            .. code-block:: python

                import paddle
                import copy
                linear = paddle.nn.Linear(1, 3)
                linear_copy = copy.deepcopy(linear)
                
                print(linear.weight)
                # Parameter containing:
                # Tensor(shape=[1, 3], dtype=float32, place=CPUPlace, stop_gradient=False,
                #     [[-0.30929261, -0.90929240, -1.07851017]])

                print(linear_copy.weight)
                # Parameter containing:
                # Tensor(shape=[1, 3], dtype=float32, place=CPUPlace, stop_gradient=False,
                #     [[-0.30929261, -0.90929240, -1.07851017]])

        """
        state = copy.deepcopy(self.__dict__, memo)
        state["name"] = self.name + unique_name.generate("_deepcopy")
        new_param = ParamBase(self.shape, self.dtype, **state)
        memo[id(self)] = new_param
        new_param.copy_(self, True)
        return new_param

    __repr__ = __str__


# program is a global instance.
_main_program_ = Program()
_startup_program_ = Program()


def default_startup_program():
    """
    Get default/global startup program.

    The :code:`paddle.nn` function will append the initialization operators into startup program.
    The :code:`startup_program` will initialize the parameters by the OPs. 

    This method will return the default or the current startup program. Users can use
    :ref:`api_paddle_fluid_framework_program_guard`  to switch :ref:`api_paddle_fluid_framework_Program` .

    Returns:
        Program: current default startup program.

    Returns type: 

    Examples:
        .. code-block:: python

            import paddle

            paddle.enable_static()
            x = paddle.static.data(name="x", shape=[-1, 784], dtype='float32')
            out = paddle.static.nn.fc(name="fc", x=x, size=10, activation="relu")
            print("main program is: {}".format(paddle.static.default_main_program()))
            print("start up program is: {}".format(paddle.static.default_startup_program()))
    """
    return _startup_program_


def default_main_program():
    """
    This API can be used to get ``default main program`` which store the 
    descriptions of Ops and tensors.

    For example ``z = paddle.add(x, y)`` will create a new ``add`` 
    Op and a new ``z`` tensor, and they will be recorded in ``default main program`` . 

    The ``default main program`` is the default value for ``Program`` parameter in 
    a lot of APIs. For example, the :code:`Executor.run()` will execute the
    :code:`default_main_program` when the program is not specified.

    If you want to switch the ``default main program``, you can use :ref:`api_paddle_fluid_framework_program_guard` .

    Returns:
        Program: A ``Program`` which holding the descriptions of OPs and tensors in the network.

    Examples:
        ..  code-block:: python

            import paddle

            paddle.enable_static()
            # Sample Network:
            x = paddle.static.data(name='x', shape=[100, 100], dtype='float32')
            y = paddle.static.data(name='x', shape=[100, 100], dtype='float32')
            out = paddle.add(x, y)

            #print the number of blocks in the program, 1 in this case
            print(paddle.static.default_main_program().num_blocks) # 1
            #print the default_main_program
            print(paddle.static.default_main_program())
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
    :api_attr: Static Graph

    Change the global main program and startup program with ``with`` statement.
    Layer functions in the Python ``with`` block will append operators and
    Tensors to the new main programs.

    Args:
        main_program(Program): New main program inside ``with`` statement.
        startup_program(Program, optional): New startup program inside ``with`` 
            statement. :code:`None` means not changing startup program, 
            default_startup_program is still used.
            Default: None.

    Examples:
       .. code-block:: python

          import paddle

          paddle.enable_static()
          main_program = paddle.static.Program()
          startup_program = paddle.static.Program()
          with paddle.static.program_guard(main_program, startup_program):
              data = paddle.static.data(name='image', shape=[None, 784, 784], dtype='float32')
              hidden = paddle.static.nn.fc(x=data, size=10, activation='relu')

    Notes: The temporary :code:`Program` can be used if the user does not need
    to construct either of startup program or main program.

    Examples:
       .. code-block:: python

          import paddle

          paddle.enable_static()
          main_program = paddle.static.Program()
          # does not care about startup program. Just pass a temporary value.
          with paddle.static.program_guard(main_program, paddle.static.Program()):
              data = paddle.static.data(name='image', shape=[None, 784, 784], dtype='float32')

    """
    from .data_feeder import check_type
    check_type(main_program, 'main_program', Program,
               'paddle.static.program_guard')
    main_program = switch_main_program(main_program)
    if startup_program is not None:
        check_type(startup_program, 'startup_program', Program,
                   'paddle.static.program_guard')
        startup_program = switch_startup_program(startup_program)
    try:
        yield
    finally:
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
def _dygraph_guard(tracer):
    global _dygraph_tracer_
    tmp_tracer = _dygraph_tracer_
    _dygraph_tracer_ = tracer
    core._switch_tracer(tracer)

    try:
        yield
    finally:
        core._switch_tracer(tmp_tracer)
        _dygraph_tracer_ = tmp_tracer


@signature_safe_contextmanager
def _dygraph_place_guard(place):
    global _global_expected_place_
    tmp_place = _global_expected_place_
    _global_expected_place_ = place

    _set_dygraph_tracer_expected_place(place)

    try:
        yield
    finally:
        _global_expected_place_ = tmp_place
        _set_dygraph_tracer_expected_place(tmp_place)


def load_op_library(lib_filename):
    """
    :api_attr: Static Graph

    Load a dynamic library, including custom operators and kernels.
    When library is loaded, ops and kernels registered in the library
    will be available in PaddlePaddle main process.
    Please note, the type of custom operators can't have the same type
    with the existing operators in the framework.

    Args:
        lib_filename (str): name of dynamic library.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            #fluid.load_op_library('custom_op.so')

    """
    core.load_op_library(lib_filename)
    OpProtoHolder.instance().update_op_proto()


def switch_device(device):
    global _current_device
    pre_device = _current_device
    _current_device = device
    return pre_device


@signature_safe_contextmanager
def device_guard(device=None):
    """
    **Notes**:
        **The API only supports static mode.**

    A context manager that specifies the device on which the OP will be placed.

    Args:
        device(str|None): Specify the device to use in the context. It should be 'cpu' or 'gpu',
            When it is set to 'cpu' or 'gpu', all OPs created in the context will be
            placed on CPUPlace or CUDAPlace. When 'gpu' is set and the program runs on
            single-card, the device index will be the same as the device on which the
            executor runs. Default: None, OPs in this context will be automatically
            assigned devices.

    Examples:
        .. code-block:: python

            import paddle

            paddle.enable_static()
            support_gpu = paddle.is_compiled_with_cuda()
            place = paddle.CPUPlace()
            if support_gpu:
                place = paddle.CUDAPlace(0)

            # if GPU is supported, the three OPs below will be automatically assigned to CUDAPlace(0)
            data1 = paddle.full(shape=[1, 3, 8, 8], fill_value=0.5, dtype='float32')
            data2 = paddle.full(shape=[1, 3, 64], fill_value=0.5, dtype='float32')
            shape = paddle.shape(data2)

            with paddle.static.device_guard("cpu"):
                # Ops created here will be placed on CPUPlace
                shape = paddle.slice(shape, axes=[0], starts=[0], ends=[4])
            with paddle.static.device_guard('gpu'):
                # if GPU is supported, OPs created here will be placed on CUDAPlace(0), otherwise on CPUPlace
                out = paddle.reshape(data1, shape=shape)

            exe = paddle.static.Executor(place)
            exe.run(paddle.static.default_startup_program())
            result = exe.run(fetch_list=[out])
    """

    index = None
    if device and ':' in device:
        device, index = device.split(':')
        if device == 'cpu':
            raise ValueError("Should not set device id for cpu.")
    if device not in ['cpu', 'gpu', '', None]:
        raise ValueError(
            "The Attr(device) should be 'cpu' or 'gpu', and it can also be empty string or None "
            "when there is no need to specify device. But received %s" % device)
    if index:
        device = ":".join([device, index])
    pre_device = switch_device(device)
    try:
        yield
    finally:
        switch_device(pre_device)


def set_flags(flags):
    """
    This function sets the GFlags value in Paddle.

    Args:
        flags (dict): A dict contains flags and its value.

    Examples:
            .. code-block:: python

                import paddle.fluid as fluid
                fluid.set_flags({'FLAGS_eager_delete_tensor_gb': 1.0})
    """
    if not isinstance(flags, dict):
        raise TypeError('flags in set_flags should be a dict')
    for key, value in flags.items():
        if core.globals().is_public(key):
            core.globals()[key] = value
        else:
            raise ValueError(
                "Flag %s cannot set its value through this function." % (key))


def get_flags(flags):
    """
    This function gets the GFlags value in Paddle.

    Args:
        flags(list|tuple|str): A list/tuple of string or a string which is the flag's name.

    Returns:
        flag's value in Paddle.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            flags = ['FLAGS_eager_delete_tensor_gb', 'FLAGS_check_nan_inf']
            res = fluid.get_flags(flags)
            print(res)
            # {'FLAGS_eager_delete_tensor_gb': 0.0, 'FLAGS_check_nan_inf': False}
    """
    flags_value = {}
    if isinstance(flags, (list, tuple)):
        for key in flags:
            if (core.globals().is_public(key)):
                value = core.globals()[key]
                temp = {key: value}
                flags_value.update(temp)
            else:
                raise ValueError(
                    'Flag %s cannot get its value through this function.' %
                    (key))
    elif isinstance(flags, str):
        if (core.globals().is_public(flags)):
            value = core.globals()[flags]
            temp = {flags: value}
            flags_value.update(temp)
        else:
            raise ValueError(
                'Flag %s cannot get its value through this function.' % (flags))
    else:
        raise TypeError('Flags in get_flags should be a list, tuple or string.')
    return flags_value


def _get_paddle_place(place):
    "convert the string to paddle Place"
    if place is None:
        return place
    if isinstance(place, (core.Place, core.XPUPlace, core.CPUPlace,
                          core.CUDAPinnedPlace, core.CUDAPlace)):
        return place

    if not isinstance(place, str):
        raise ValueError(
            "place only support string which is 'Place' and so on.")

    place = place.lower()
    if (place == "cpu"):
        return core.CPUPlace()
    if (place == "device"):
        return core.Place()

    avaliable_gpu_place = re.match(r'gpu:\d+', place)
    if place == "gpu_pinned" or place == "gpu" or avaliable_gpu_place:
        if not core.is_compiled_with_cuda():
            raise ValueError(
                "The device should not be {}, since PaddlePaddle is " \
                "not compiled with CUDA".format(avaliable_gpu_place))
        if place == "gpu_pinned":
            return core.CUDAPinnedPlace()
        elif place == "gpu":
            return core.CUDAPlace(0)
        else:
            place_info_list = place.split(':', 1)
            device_id = place_info_list[1]
            device_id = int(device_id)
            return core.CUDAPlace(device_id)
    avaliable_xpu_place = re.match(r'xpu:\d+', place)
    if avaliable_xpu_place:
        if not core.is_compiled_with_xpu():
            raise ValueError(
                "The device should not be {}, since PaddlePaddle is " \
                "not compiled with XPU".format(avaliable_xpu_place))
        place_info_list = place.split(':', 1)
        device_id = place_info_list[1]
        device_id = int(device_id)
        return core.XPUPlace(device_id)
    raise ValueError(
        "paddle support CPUPlace, CUDAPlace,CUDAPinnedPlace and XPUPlace, Please check your Place Input"
    )


def _get_paddle_place_list(places):

    if not isinstance(places, (list, tuple)):
        raise TypeError("places must to be List or Tuple")

    ret = []
    for p in places:
        p = _get_paddle_place(p)
        ret.append(p)

    return ret
