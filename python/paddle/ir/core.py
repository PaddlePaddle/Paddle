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

from paddle.base.libpaddle import DataType
from paddle.base.libpaddle.ir import Program, set_global_program

from ..base.wrapped_decorator import signature_safe_contextmanager

np_type_to_paddle_type = {
    np.dtype("float32"): DataType.FLOAT32,
    np.dtype("float64"): DataType.FLOAT64,
    np.dtype("float16"): DataType.FLOAT16,
    np.dtype("int32"): DataType.INT32,
    np.dtype("int16"): DataType.INT16,
    np.dtype("int64"): DataType.INT64,
    np.dtype("bool_"): DataType.BOOL,
    np.dtype("uint16"): DataType.UINT16,
    np.dtype("uint8"): DataType.UINT8,
    np.dtype("int8"): DataType.INT8,
    np.dtype("complex64"): DataType.COMPLEX64,
    np.dtype("complex128"): DataType.COMPLEX128,
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
        dtype = np.uint16
    else:
        dtype = np.dtype(np_dtype)

    if dtype in np_type_to_paddle_type.keys():
        return np_type_to_paddle_type[dtype]
    else:
        raise ValueError("Not supported numpy dtype %s" % dtype)


def _use_new_ir_api():
    """
    This API checks whether paddle use new ir api.

    Returns:
        bool: Whether paddle use new ir api.

    """
    # TODO(YuanRisheng): need move import to the top of this file after break import circle
    import paddle

    if paddle.framework.get_flags("FLAGS_enable_new_ir_api")[
        'FLAGS_enable_new_ir_api'
    ]:
        return True
    else:
        return False


# program is a global instance.
_main_program_ = Program()
# set the global program for c++ and this program will be used to build ops in c++
set_global_program(_main_program_)

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

    If you want to switch the ``default main program``, you can use :ref:`api_paddle_ir_core_program_guard` .

    Returns:
        Program: A ``Program`` which holding the descriptions of OPs and tensors in the network.

    Examples:
        ..  code-block:: python

            import paddle

            paddle.enable_static()
            # Sample Network:
            x = paddle.static.data(name='x', shape=[100, 100], dtype='float32')
            y = paddle.static.data(name='y', shape=[100, 100], dtype='float32')
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
    set_global_program(_main_program_)
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
            :name: code-example-1

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
            :name: code-example-2

            import paddle

            paddle.enable_static()
            main_program = paddle.static.Program()
            # does not care about startup program. Just pass a temporary value.
            with paddle.static.program_guard(main_program, paddle.static.Program()):
                data = paddle.static.data(name='image', shape=[None, 784, 784], dtype='float32')

    """
    from ..base.data_feeder import check_type

    check_type(
        main_program, 'main_program', Program, 'paddle.static.program_guard'
    )
    main_program = switch_main_program(main_program)
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
        switch_main_program(main_program)
        if startup_program is not None:
            switch_startup_program(startup_program)
