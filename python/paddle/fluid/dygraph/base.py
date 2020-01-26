# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from ..wrapped_decorator import signature_safe_contextmanager, wrap_decorator
import contextlib
import numpy as np
from paddle.fluid import core
from paddle.fluid import framework
from .tracer import Tracer
import logging
import objgraph

__all__ = [
    'no_grad',
    'guard',
    'to_variable',
]


def _switch_to_static_graph_(func):
    def __impl__(*args, **kwargs):
        with framework._dygraph_guard(None):
            return func(*args, **kwargs)

    return __impl__


switch_to_static_graph = wrap_decorator(_switch_to_static_graph_)


@signature_safe_contextmanager
def program_desc_tracing_guard(enable):
    tracer = framework._dygraph_tracer()
    if tracer:
        original_val = tracer._enable_program_desc_tracing
        tracer._enable_program_desc_tracing = enable
    yield
    if tracer:
        tracer._enable_program_desc_tracing = original_val


# This function should be removed in V1.6, because it can easily lead to cyclic dependencies.
def enabled():
    # Internal use only
    return framework.in_dygraph_mode()


@contextlib.contextmanager
def _switch_tracer_mode_guard_(is_train=True):
    tracer = framework._dygraph_tracer()
    if tracer:
        mode = tracer._train_mode
        tracer._train_mode = is_train
        yield
        tracer._train_mode = mode
    else:
        yield


def _no_grad_(func):
    """
    This Decorator will avoid the func being decorated creating backward network in dygraph mode

    Parameter:
        - **func** (python func): the func don't need grad

    Examples:

     .. code-block:: python

        import numpy as np
        import paddle.fluid as fluid

        @fluid.dygraph.no_grad
        def test_layer():
            with fluid.dygraph.guard():
                inp = np.ones([3, 1024], dtype='float32')
                t = fluid.dygraph.base.to_variable(inp)
                linear1 = fluid.Linear(1024, 4, bias_attr=False)
                linear2 = fluid.Linear(4, 4)
                ret = linear1(t)
                dy_ret = linear2(ret)

        test_layer()

    """

    def __impl__(*args, **kwargs):
        with _switch_tracer_mode_guard_(is_train=False):
            return func(*args, **kwargs)

    return __impl__


no_grad = wrap_decorator(_no_grad_)
# for fluidDoc
no_grad.__doc__ = _no_grad_.__doc__


@signature_safe_contextmanager
def guard(place=None):
    """
    This context will create a dygraph context for dygraph to run, using python ``with`` statement.

    Parameters:
        place(fluid.CPUPlace or fluid.CUDAPlace, optional): Place to execute dygraph. 
            If None, the running place will be determined according to the way of paddle compilation. Default: None

    return:
        None

    Examples:

     .. code-block:: python

        import numpy as np
        import paddle.fluid as fluid

        with fluid.dygraph.guard():
            inp = np.ones([3, 1024], dtype='float32')
            t = fluid.dygraph.base.to_variable(inp)
            linear1 = fluid.Linear(1024, 4, bias_attr=False)
            linear2 = fluid.Linear(4, 4)
            ret = linear1(t)
            dy_ret = linear2(ret)

    """
    train = framework.Program()
    startup = framework.Program()
    tracer = Tracer()
    VarBase = core.VarBase

    if place is None:
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()
    tracer._expected_place = place

    with framework.program_guard(train, startup):
        with framework.unique_name.guard():
            with framework._dygraph_guard(tracer):
                with framework._dygraph_place_guard(place):
                    yield


def _print_debug_msg(parameter_list, limit=5, is_test=False):
    if not core._is_dygraph_debug_enabled():
        logging.warn(
            'Debug mode is not enabled. Please set FLAGS_dygraph_debug=1 to enable debug'
        )
        return
    unique_name_size = len(framework.unique_name.generator.ids)
    tracer_var_size = len(parameter_list)
    alive_cpp_var_size = len(core.VarBase._alive_vars())
    if not is_test:
        logging.warn(
            'unique_name num: {}, tracer vars num: {}, alive cpp vars num: {}'
            .format(unique_name_size, tracer_var_size, alive_cpp_var_size))
        objgraph.show_growth(limit=limit)
    else:
        return unique_name_size, tracer_var_size, alive_cpp_var_size


@framework.dygraph_only
def to_variable(value, name=None, zero_copy=None):
    """
    The API will create a ``Variable`` object from numpy\.ndarray or Variable object.

    Parameters:
        value(ndarray|Variable): The numpy\.ndarray or Variable object that needs to be converted, it can be multi-dimension, and the data type is one of numpy\.{float16, float32, float64, int16, int32, int64, uint8, uint16}.
        name(str, optional): The default value is None. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`
        zero_copy(bool, optional): Whether to share memory with the input numpy array. This parameter only works with CPUPlace and will be set to True when it is None. Default: None.

    Returns:
        Variable: If ``value`` is a numpy\.ndarray object, return ``Tensor`` created from the specified numpy\.ndarray object, which has same data type and shape with ``value``. If ``value`` is a Variable object, just return ``value``.


    Examples:

     .. code-block:: python

        import numpy as np
        import paddle.fluid as fluid

        with fluid.dygraph.guard(fluid.CPUPlace()):
            x = np.ones([2, 2], np.float32)
            y = fluid.dygraph.to_variable(x, zero_copy=False)
            x[0][0] = -1
            y[0][0].numpy()  # array([1.], dtype=float32)
            y = fluid.dygraph.to_variable(x)
            x[0][0] = 0
            y[0][0].numpy()  # array([0.], dtype=float32)

    """
    if isinstance(value, np.ndarray):
        assert framework.in_dygraph_mode(
        ), "to_variable could only be called in dygraph mode"
        if isinstance(framework._current_expected_place(),
                      framework.core.CPUPlace):
            if zero_copy is None:
                zero_copy = True
        else:
            assert not zero_copy, "zero_copy mode can only be used with CPUPlace"
            zero_copy = False
        py_var = core.VarBase(
            value=value,
            place=framework._current_expected_place(),
            persistable=False,
            zero_copy=zero_copy,
            name=name if name else '')
        return py_var
    elif isinstance(value, (core.VarBase, framework.Variable)):
        return value
    else:
        raise TypeError(
            "to_variable only accepts 'ndarray' and 'Variable' as value's input")
