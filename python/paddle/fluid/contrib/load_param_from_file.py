#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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

import contextlib
import numpy as np
from .. import core
from .. import executor
from .. import framework
from ..framework import Program
from .. import io
import paddle.fluid as fluid

__all__ = ['load_param_from_file']


@contextlib.contextmanager
def scope_prog_guard(main_prog, startup_prog):
    scope = core.Scope()
    with framework.unique_name.guard():
        with executor.scope_guard(scope):
            with framework.program_guard(main_prog, startup_prog):
                yield


def load_param_from_file(vars_name, param_dir, dtype="float32"):
    """
    Load parameters from files.

    Args:
        vars_name(list): the parameter file that need be loaded.
        param_dir(str): the parameter path direction.
        dtype(str): the dtype of the elements which return from
            load_param_from_file.

    Returns:
        var_tensor(dict): the dictionary of parameter and tensor.

    Examples:

        >>> import paddle.fluid as fluid
        >>> param_tensor = \
        >>>     fluid.contrib.load_param_from_file(
        >>>         vars_name = ['conv2d_0.w_0', 'batch_norm_0.b_0'],
        >>>         param_dir = "/image_classification_vgg.inference.model/")

    """
    var_tensor = {}

    with scope_prog_guard(main_prog=Program(), startup_prog=Program()):
        main_prog = framework.default_main_program()
        for var in vars_name:
            main_prog.global_block().create_var(
                name=var, shape=[1], dtype=dtype)

        exe = executor.Executor(core.CPUPlace())
        io.load_vars(exe, param_dir)

        for var in vars_name:
            var_tensor[var] = \
                np.array(fluid.global_scope().var(var).get_tensor())
    return var_tensor
