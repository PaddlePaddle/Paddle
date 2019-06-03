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

__all__ = [
    'enabled',
    'no_grad',
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


def _no_grad_(func):
    def __impl__(*args, **kwargs):
        with _switch_tracer_mode_guard_(is_train=False):
            return func(*args, **kwargs)

    return __impl__


no_grad = wrap_decorator(_no_grad_)


@signature_safe_contextmanager
def guard(place=None):
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


def to_variable(value, block=None, name=None):
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
