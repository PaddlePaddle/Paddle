#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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


import numpy as np

from paddle.base.core import Place, VarDesc
from paddle.base.libpaddle import DataType
from paddle.base.libpaddle.pir import (
    Program,
    get_current_insertion_point,
    reset_insertion_point_to_start,
    set_insertion_point,
    set_insertion_point_to_block_end,
)

from .._pir_ops import data, parameter, set_parameter, set_persistable_value
from ..base import unique_name
from ..base.core import set_static_op_arg_pre_cast_hook
from ..base.wrapped_decorator import signature_safe_contextmanager

vartype_to_datatype = {
    VarDesc.VarType.FP32: DataType.FLOAT32,
    VarDesc.VarType.FP64: DataType.FLOAT64,
    VarDesc.VarType.FP16: DataType.FLOAT16,
    VarDesc.VarType.BF16: DataType.BFLOAT16,
    VarDesc.VarType.INT32: DataType.INT32,
    VarDesc.VarType.INT16: DataType.INT16,
    VarDesc.VarType.INT64: DataType.INT64,
    VarDesc.VarType.BOOL: DataType.BOOL,
    VarDesc.VarType.UINT8: DataType.UINT8,
    VarDesc.VarType.INT8: DataType.INT8,
    VarDesc.VarType.COMPLEX64: DataType.COMPLEX64,
    VarDesc.VarType.COMPLEX128: DataType.COMPLEX128,
}

datatype_to_vartype = {v: k for k, v in vartype_to_datatype.items()}

np_type_to_paddle_type = {
    np.dtype("float32"): DataType.FLOAT32,
    np.dtype("float64"): DataType.FLOAT64,
    np.dtype("float16"): DataType.FLOAT16,
    np.dtype("int32"): DataType.INT32,
    np.dtype("int16"): DataType.INT16,
    np.dtype("int64"): DataType.INT64,
    np.dtype("bool_"): DataType.BOOL,
    np.dtype("uint16"): DataType.BFLOAT16,
    np.dtype("uint8"): DataType.UINT8,
    np.dtype("int8"): DataType.INT8,
    np.dtype("complex64"): DataType.COMPLEX64,
    np.dtype("complex128"): DataType.COMPLEX128,
    np.float16: DataType.FLOAT16,
    np.float32: DataType.FLOAT32,
    np.float64: DataType.FLOAT64,
    np.int32: DataType.INT32,
    np.int16: DataType.INT16,
    np.int64: DataType.INT64,
    np.bool_: DataType.BOOL,
    np.uint16: DataType.BFLOAT16,
    np.uint8: DataType.UINT8,
    np.int8: DataType.INT8,
    np.complex64: DataType.COMPLEX64,
    np.complex128: DataType.COMPLEX128,
}

_PADDLE_PIR_DTYPE_2_NUMPY_DTYPE = {
    DataType.BOOL: 'bool',
    DataType.FLOAT16: 'float16',
    DataType.BFLOAT16: 'uint16',
    DataType.FLOAT32: 'float32',
    DataType.FLOAT64: 'float64',
    DataType.INT8: 'int8',
    DataType.INT16: 'int16',
    DataType.INT32: 'int32',
    DataType.INT64: 'int64',
    DataType.UINT8: 'uint8',
    DataType.COMPLEX64: 'complex64',
    DataType.COMPLEX128: 'complex128',
}


def convert_np_dtype_to_dtype_(np_dtype):
    """
    Convert the data type in numpy to the data type in Paddle.

    Args:
        np_dtype (np.dtype|str): The data type in numpy or valid data type
            string.

    Returns:
        core.DataType : The data type in Paddle.

    """
    # Convert the data type string to numpy data type.
    if isinstance(np_dtype, str) and np_dtype == "bfloat16":
        # since there is still no support for bfloat16 in NumPy,
        # uint16 is used for casting bfloat16
        dtype = np.dtype("uint16")
    else:
        dtype = np.dtype(np_dtype)

    if dtype in np_type_to_paddle_type.keys():
        return np_type_to_paddle_type[dtype]
    else:
        raise ValueError("Not supported numpy dtype %s" % dtype)


# program is a global instance.
_main_program_ = Program()
# set the global program for c++ and this program will be used to build ops in c++
set_insertion_point_to_block_end(_main_program_.global_block())

_startup_program_ = Program()


def default_startup_program():
    """
    Get default/global startup program.

    The :code:`paddle.nn` function will append the initialization operators into startup program.
    The :code:`startup_program` will initialize the parameters by the OPs.

    This method will return the default or the current startup program. Users can use
    :ref:`api_paddle_ir_core_program_guard`  to switch :ref:`api_paddle_ir_Program` .

    Returns:
        Program: current default startup program.

    Returns type:

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> paddle.enable_static()
            >>> x = paddle.static.data(name="x", shape=[-1, 784], dtype='float32')
            >>> out = paddle.static.nn.fc(name="fc", x=x, size=10, activation="relu")
            >>> print("main program is: {}".format(paddle.static.default_main_program()))
            >>> print("start up program is: {}".format(paddle.static.default_startup_program()))
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

    If you want to switch the ``default main program``, you can use :ref:`api_paddle_ir_core_program_guard` .

    Returns:
        Program: A ``Program`` which holding the descriptions of OPs and tensors in the network.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> paddle.enable_static()
            >>> # Sample Network:
            >>> x = paddle.static.data(name='x', shape=[100, 100], dtype='float32')
            >>> y = paddle.static.data(name='y', shape=[100, 100], dtype='float32')
            >>> out = paddle.add(x, y)

            >>> print the number of blocks in the program, 1 in this case
            >>> print(paddle.static.default_main_program().num_blocks) # 1
            >>> print the default_main_program
            >>> print(paddle.static.default_main_program())
    """
    return _main_program_


def switch_main_program(program, insertion_point=None):
    """
    Switch the main program to a new program.

    Args:
        program(Program): The new main program

    Returns:
        Program: The previous main program
    """
    global _main_program_
    prev_program = _main_program_
    prev_insertion_point = get_current_insertion_point()
    _main_program_ = program
    if insertion_point is None:
        set_insertion_point_to_block_end(_main_program_.global_block())
    else:
        set_insertion_point(insertion_point)
    return prev_program, prev_insertion_point


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
            :name: code-example-1

            >>> import paddle

            >>> paddle.enable_static()
            >>> main_program = paddle.static.Program()
            >>> startup_program = paddle.static.Program()
            >>> with paddle.static.program_guard(main_program, startup_program):
            ...     data = paddle.static.data(name='image', shape=[None, 784, 784], dtype='float32')
            ...     hidden = paddle.static.nn.fc(x=data, size=10, activation='relu')

    Notes: The temporary :code:`Program` can be used if the user does not need
    to construct either of startup program or main program.

    Examples:
        .. code-block:: python
            :name: code-example-2

            >>> import paddle

            >>> paddle.enable_static()
            >>> main_program = paddle.static.Program()
            >>> # does not care about startup program. Just pass a temporary value.
            >>> with paddle.static.program_guard(main_program, paddle.static.Program()):
            ...     data = paddle.static.data(name='image', shape=[None, 784, 784], dtype='float32')
    """
    from ..base.data_feeder import check_type

    check_type(
        main_program, 'main_program', Program, 'paddle.static.program_guard'
    )
    main_program, prev_insertion_point = switch_main_program(main_program)
    if startup_program is not None:
        check_type(
            startup_program,
            'startup_program',
            Program,
            'paddle.static.program_guard',
        )
        startup_program = switch_startup_program(startup_program)
    try:
        yield
    finally:
        switch_main_program(main_program, prev_insertion_point)
        if startup_program is not None:
            switch_startup_program(startup_program)


class ParameterMeta:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype


def create_parameter(
    dtype,
    shape,
    name=None,
    **kwargs,
):
    if 'initializer' not in kwargs:
        raise ValueError(
            "initializer is None, if you want to create parameter, please pass its initializer."
        )
    if dtype is not None:
        if not isinstance(dtype, DataType):
            dtype = convert_np_dtype_to_dtype_(dtype)
    value_name = name
    if not value_name:
        value_name = unique_name.generate('parameter')
    startup_program = default_startup_program()
    main_program = default_main_program()
    parameter_meta = ParameterMeta(shape, dtype)

    with program_guard(startup_program):
        initializer = kwargs['initializer']
        init_result = initializer(
            parameter_meta, startup_program.global_block()
        )
        init_result.persistable = True
        set_parameter(init_result, value_name)

    main_program.set_parameters_from(startup_program)
    with program_guard(default_main_program()):
        reset_insertion_point_to_start()
        param = parameter(value_name)
        param.persistable = True

    param.trainable = kwargs.get('trainable', True)
    param.stop_gradient = not param.trainable
    param.optimize_attr = kwargs.get('optimize_attr', {'learning_rate': 1.0})
    param.regularizer = kwargs.get('regularizer', None)
    param.do_model_average = kwargs.get('do_model_average', None)
    param.need_clip = kwargs.get('need_clip', True)
    param.is_distributed = False
    param.is_parameter = True
    return param


def create_persistable_value(dtype, shape, name=None, **kwargs):
    """
    Create Value that is persistable in startup program and main program. The Value is initilized in startup program and
    used in main program.

    Returns:
        Value: The created Value from main program
    """
    if 'initializer' not in kwargs:
        raise ValueError(
            "initializer is None, if you want to create parameter, please pass its initializer."
        )
    if dtype is not None:
        if not isinstance(dtype, DataType):
            dtype = convert_np_dtype_to_dtype_(dtype)
    value_name = name
    if not value_name:
        value_name = unique_name.generate('persistable_value')
    startup_program = default_startup_program()
    main_program = default_main_program()

    with program_guard(startup_program):
        initializer = kwargs['initializer']
        parameter_meta = ParameterMeta(shape, dtype)
        init_result = initializer(
            parameter_meta, startup_program.global_block()
        )
        init_result.persistable = True
        set_persistable_value(init_result, value_name)

    with program_guard(default_main_program()):
        reset_insertion_point_to_start()
        persist_value = data(value_name, shape, dtype, Place())
        persist_value.persistable = True

    return persist_value


def _get_persistable_value(target_program, value_info):
    """
    Get a persistable value from a target program by using value that is in other program.
    """
    with program_guard(target_program):
        target_value = data(
            value_info.name, value_info.shape, value_info.dtype, Place()
        )
        target_value.persistable = True
    return target_value


def _get_parameter(target_program, param_info):
    """
    Get a parameter from a target program by using parameter that is in other program.
    """
    target_program.set_parameters_from(default_startup_program())
    with program_guard(target_program):
        target_param = parameter(param_info.name)
        target_param.persistable = True
        target_param.stop_gradient = param_info.stop_gradient

    if hasattr(param_info, 'regularizer'):
        target_param.regularizer = param_info.regularizer
    if hasattr(param_info, 'need_clip'):
        target_param.need_clip = param_info.need_clip
    return target_param


def _convert_into_value(tensor):
    """
    Convert Tensor into Value.
    """
    import paddle
    from paddle.jit.pir_dy2static.parameter_recorder import (
        _global_parameter_recorder,
    )

    if isinstance(tensor, paddle.Tensor):
        value = _global_parameter_recorder.get(
            paddle.pir.core.default_main_program(), tensor
        )
        NON_PERSISTABLE_VAR_NAME_SUFFIX = "__non_persistable"
        if tensor.name.endswith(NON_PERSISTABLE_VAR_NAME_SUFFIX):
            value.persistable = False
        return value

    return tensor


@signature_safe_contextmanager
def static_op_arg_cast_guard(hook):
    """
    Set a hook function to cast the arguments of static op.
    """

    original_callback = set_static_op_arg_pre_cast_hook(hook)
    try:
        yield
    finally:
        set_static_op_arg_pre_cast_hook(original_callback)
