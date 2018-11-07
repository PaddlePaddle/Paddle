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

from __future__ import print_function

import six

from paddle.fluid import core
import paddle


def __convert_distributedmodel():
    pass


def __convert_distributedparams():
    pass


def convert_distributed2local(model_dirname,
                              model_only=False,
                              model_filename=None,
                              params_filename=None,
                              new_model_dir=None):
    """
    Get the LoDTensor value of the given parameter.

    Args:
        model_dirname(str): The parameter to get value from.
        new_model_dir(str): The executor to run for retrieving the value.

    Returns:
        None

    Raises:
        AssertionError: If the `model_dir` is not an instance of Parameter.

    Examples:
        .. code-block:: python

            exe = fluid.Executor(fluid.CPUPlace())
            param = fluid.default_main_program().global_block().var('fc.w')
            p = fluid.io.get_parameter_value(param, exe)

    """
    pass
