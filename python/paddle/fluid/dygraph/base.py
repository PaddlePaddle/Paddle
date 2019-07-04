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
import os

from paddle.fluid import core
from paddle.fluid import framework
from .tracer import Tracer
import logging

__all__ = [
    'enabled',
    'no_grad',
    'not_support',
    'guard',
    'to_variable',
]


def enabled():
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


def _dygraph_not_support_(func):
    def __impl__(*args, **kwargs):
        assert not framework.in_dygraph_mode(
        ), "We don't support %s in Dygraph mode" % func.__name__
        return func(*args, **kwargs)

    return __impl__


def _no_grad_(func):
    """
    This Decorator will avoid the func being decorated creating backward network in dygraph mode

    Args:
        func: the func don't need grad

    Examples:

     .. code-block:: python

        import numpy as np
        import paddle.fluid as fluid

        @fluid.dygraph.no_grad
        def test_layer():
            with fluid.dygraph.guard():
                inp = np.ones([3, 32, 32], dtype='float32')
                t = fluid.dygraph.base.to_variable(inp)
                fc1 = fluid.FC('fc1', size=4, bias_attr=False, num_flatten_dims=1)
                fc2 = fluid.FC('fc2', size=4)
                ret = fc1(t)
                dy_ret = fc2(ret)

        test_layer()

    """

    def __impl__(*args, **kwargs):
        with _switch_tracer_mode_guard_(is_train=False):
            return func(*args, **kwargs)

    return __impl__


no_grad = wrap_decorator(_no_grad_)
not_support = wrap_decorator(_dygraph_not_support_)


@signature_safe_contextmanager
def guard(place=None):
    """
    This context will create a dygraph context for dygraph to run

    Args:
        place(fluid.CPUPlace|fluid.CUDAPlace|None): Place to run

    return:
        None

    Examples:

     .. code-block:: python

        import numpy as np
        import paddle.fluid as fluid

        with fluid.dygraph.guard():
            inp = np.ones([3, 32, 32], dtype='float32')
            t = fluid.dygraph.base.to_variable(inp)
            fc1 = fluid.FC('fc1', size=4, bias_attr=False, num_flatten_dims=1)
            fc2 = fluid.FC('fc2', size=4)
            ret = fc1(t)
            dy_ret = fc2(ret)

    """
    train = framework.Program()
    startup = framework.Program()
    tracer = Tracer(train.current_block().desc)

    if place is None:
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()

    with framework.program_guard(train, startup):
        with framework.unique_name.guard():
            with framework._dygraph_guard(tracer):
                with framework._dygraph_place_guard(place):
                    yield


def _print_debug_msg():
    if not core._is_dygraph_debug_enabled():
        logging.warn(
            'Debug mode is not enabled. Please set FLAGS_dygraph_debug=1 to enable debug'
        )
        return

    unique_name_size = len(framework.unique_name.generator.ids)
    tracer_var_size = len(framework._dygraph_tracer()._vars)
    alive_cpp_var_size = len(core.VarBase._alive_vars())
    logging.warn(
        'unique_name num: {}, tracer vars num: {}, alive cpp vars num: {}'
        .format(unique_name_size, tracer_var_size, alive_cpp_var_size))


def to_variable(value, block=None, name=None):
    """
    This function will create a variable from ndarray

    Args:
        value(ndarray): the numpy value need to be convert
        block(fluid.Block|None): which block this variable will be in
        name(str|None): Name of Varaible

    return:
        Variable: The variable created from given numpy

    Examples:

     .. code-block:: python

        import numpy as np
        import paddle.fluid as fluid

        with fluid.dygraph.guard():
            x = np.ones([2, 2], np.float32)
            y = fluid.dygraph.to_variable(x)

    """
    if isinstance(value, np.ndarray):
        assert enabled(), "to_variable could only be called in dygraph mode"

        if not block:
            block = framework.default_main_program().current_block()
        py_var = framework.Variable(
            block,
            type=core.VarDesc.VarType.LOD_TENSOR,
            name=name,
            shape=value.shape,
            dtype=value.dtype,
            stop_gradient=True)
        var = py_var._ivar.value()
        tensor = var.get_tensor()
        tensor.set(value, framework._current_expected_place())
        return py_var
    elif isinstance(value, framework.Variable):
        return value
    else:
        raise TypeError(
            "to_variable only accepts 'ndarray' and 'Variable' as value's input")
