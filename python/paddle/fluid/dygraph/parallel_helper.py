# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except jin compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from ..layers import collective
from ..framework import Parameter

__parallel_ctx__clz__ = None


def _is_data_parallel_mode():
    global __parallel_ctx__clz__
<<<<<<< HEAD
    return (
        __parallel_ctx__clz__ is not None
        and int(os.getenv("PADDLE_TRAINERS_NUM", "1")) > 1
    )
=======
    return __parallel_ctx__clz__ is not None and int(
        os.getenv("PADDLE_TRAINERS_NUM", "1")) > 1
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


def _is_parallel_ctx_initialized():
    global __parallel_ctx__clz__
    return __parallel_ctx__clz__ is not None


def _set_parallel_ctx(ccl_parallel_context):
    global __parallel_ctx__clz__
<<<<<<< HEAD
    assert (
        __parallel_ctx__clz__ is None
    ), "ParallelContext can only be initialized once."
=======
    assert __parallel_ctx__clz__ is None, \
        "ParallelContext can only be initialized once."
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    __parallel_ctx__clz__ = ccl_parallel_context


def _init_parallel_ctx():
    global __parallel_ctx__clz__
<<<<<<< HEAD
    assert (
        __parallel_ctx__clz__ is not None
    ), "ParallelContext should be initialized."
=======
    assert __parallel_ctx__clz__ is not None, \
        "ParallelContext should be initialized."
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    __parallel_ctx__clz__.init()


def _broadcast_parameters(parameters):
    for param in parameters:
        # In model parallel, some parameters are split into multiple devices,
        # so we could not broadcast these parameters.
<<<<<<< HEAD
        if param.is_distributed:
            continue
=======
        if param.is_distributed: continue
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        if isinstance(param, Parameter) and param.trainable:
            collective._broadcast(param, 0, sync_mode=True)
