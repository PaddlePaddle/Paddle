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
from ..wrapped_decorator import signature_safe_contextmanager
import numpy as np

from paddle.fluid import core
from paddle.fluid import framework
from .tracer import Tracer

__all__ = ['enabled', 'guard', 'to_variable']


def enabled():
    return framework._in_imperative_mode()


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
            with framework._imperative_guard(tracer):
                with framework._imperative_place_guard(place):
                    yield


def to_variable(value, block=None, name=None):
    if isinstance(value, np.ndarray):
        assert enabled(), "to_variable could only be called in imperative mode"

        if not block:
            block = framework.default_main_program().current_block()
        py_var = framework.Variable(
            block,
            type=core.VarDesc.VarType.LOD_TENSOR,
            name=name,
            shape=value.shape,
            dtype=value.dtype)
        var = py_var._ivar.value()
        tensor = var.get_tensor()
        tensor.set(value, framework._current_expected_place())
        return py_var
    elif isinstance(value, framework.Variable):
        return value
