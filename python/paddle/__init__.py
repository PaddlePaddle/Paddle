# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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

import os
from paddle.check_import_scipy import check_import_scipy

check_import_scipy(os.name)

try:
    from paddle.version import full_version as __version__
    from paddle.version import commit as __git_commit__

except ImportError:
    import sys
    sys.stderr.write('''Warning with import paddle: you should not
     import paddle from the source directory; please install paddlepaddle*.whl firstly.'''
                     )

import paddle.reader
import paddle.dataset
import paddle.batch
import paddle.compat
import paddle.distributed
batch = batch.batch
import paddle.sysconfig
import paddle.nn
import paddle.framework
import paddle.imperative
import paddle.complex

# from .framework.framework import set_default_dtype   #DEFINE_ALIAS
# from .framework.framework import get_default_dtype   #DEFINE_ALIAS
from .framework.random import manual_seed  #DEFINE_ALIAS
# from .framework import append_backward   #DEFINE_ALIAS
# from .framework import gradients   #DEFINE_ALIAS
# from .framework import Executor   #DEFINE_ALIAS
# from .framework import global_scope   #DEFINE_ALIAS
# from .framework import scope_guard   #DEFINE_ALIAS
# from .framework import BuildStrategy   #DEFINE_ALIAS
# from .framework import CompiledProgram   #DEFINE_ALIAS
# from .framework import default_main_program   #DEFINE_ALIAS
# from .framework import default_startup_program   #DEFINE_ALIAS
# from .framework import create_global_var   #DEFINE_ALIAS
# from .framework import create_parameter   #DEFINE_ALIAS
# from .framework import create_py_reader_by_data   #DEFINE_ALIAS
# from .framework import Print   #DEFINE_ALIAS
# from .framework import py_func   #DEFINE_ALIAS
# from .framework import ExecutionStrategy   #DEFINE_ALIAS
# from .framework import in_dygraph_mode   #DEFINE_ALIAS
# from .framework import name_scope   #DEFINE_ALIAS
# from .framework import ParallelExecutor   #DEFINE_ALIAS
# from .framework import ParamAttr   #DEFINE_ALIAS
# from .framework import Program   #DEFINE_ALIAS
# from .framework import program_guard   #DEFINE_ALIAS
# from .framework import Variable   #DEFINE_ALIAS
# from .framework import WeightNormParamAttr   #DEFINE_ALIAS
# from .framework import Model   #DEFINE_ALIAS
# from .framework import Sequential   #DEFINE_ALIAS
