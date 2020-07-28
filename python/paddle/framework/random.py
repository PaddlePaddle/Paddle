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

__all__ = ['manual_seed']


def manual_seed(seed):
    """
	:alias_main: paddle.manual_seed
	:alias: paddle.manual_seed,paddle.framework.random.manual_seed

    Set global manual seed for program

    Args:
        manual_seed(int): random seed for program

    Returns:
        None.

    Examples:
        .. code-block:: python

            from paddle.framework import manual_seed
            manual_seed(102)
    """
    fluid.default_main_program().random_seed = seed
    fluid.default_startup_program().random_seed = seed
    program = fluid.Program()
    program.global_seed(seed)
