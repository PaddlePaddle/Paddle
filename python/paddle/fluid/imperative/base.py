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
import contextlib
import numpy as np

from paddle.fluid import core
from paddle.fluid import framework

__all__ = ['enabled', 'guard', 'to_variable']


def enabled():
    return framework._in_imperative_mode()


@contextlib.contextmanager
def guard():
    train = framework.Program()
    startup = framework.Program()
    tracer = core.Tracer(train.current_block().desc)
    with framework.program_guard(train, startup):
        with framework.unique_name.guard():
            with framework._imperative_guard(tracer):
                yield


def to_variable(value, block=None):
    if isinstance(value, np.ndarray):
        if not block:
            block = framework.default_main_program().current_block()
        py_var = framework.Variable(
            block,
            type=core.VarDesc.VarType.LOD_TENSOR,
            name=None,
            shape=value.shape,
            dtype=value.dtype)
        var = py_var._ivar.value
        tensor = var.get_tensor()
        tensor.set(value, core.CPUPlace())
        return py_var
    elif isinstance(value, framework.Variable):
        return value
    else:
        raise ValueError("Unsupported type %s" % type(value))
