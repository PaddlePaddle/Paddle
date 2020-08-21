#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

# TODO: define random api
import paddle.fluid as fluid
from paddle.fluid import core

__all__ = ['manual_seed']


def manual_seed(seed):
    """

    Sets global seed for generating random numbers.

    Args:
        seed(int): The random seed to set. It is recommend to set a large int number.

    Returns:
        Generator: a generator object.

    Examples:
        .. code-block:: python

            import paddle
            paddle.manual_seed(102)

    """
    #TODO(zhiqiu): 1. remove program.random_seed when all random-related op upgrade
    # 2. support gpu generator by global device 

    seed = int(seed)

    fluid.default_main_program().random_seed = seed
    fluid.default_startup_program().random_seed = seed
    program = fluid.Program()
    program.global_seed(seed)

    core.default_cpu_generator()._is_init_py = False
    print(core.default_cpu_generator()._is_init_py)
    return core.default_cpu_generator().manual_seed(seed)
