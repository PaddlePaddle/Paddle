#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import errno
import warnings
import logging
import pickle
import contextlib
from functools import reduce
import sys
from io import BytesIO

import numpy as np
import math
import paddle
from paddle.fluid import layers
from paddle.fluid.executor import Executor, global_scope
from paddle.fluid.framework import (
    Program,
    Parameter,
    default_main_program,
    default_startup_program,
    Variable,
    program_guard,
    dygraph_not_support,
    static_only,
)
from paddle.reader import (
    cache,
    map_readers,
    buffered,
    compose,
    chain,
    shuffle,
    ComposeNotAligned,
    firstn,
    xmap_readers,
    multiprocess_reader,
)
from .wrapped_decorator import signature_safe_contextmanager
from paddle.fluid.compiler import CompiledProgram
from paddle.fluid.log_helper import get_logger
from . import reader
from . import unique_name
from .reader import *
from . import core
from paddle.utils import deprecated
from paddle.fluid.framework import static_only

__all__ = reader.__all__


_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s'
)


def prepend_feed_ops(
    inference_program, feed_target_names, feed_holder_name='feed'
):
    if len(feed_target_names) == 0:
        return

    global_block = inference_program.global_block()
    feed_var = global_block.create_var(
        name=feed_holder_name,
        type=core.VarDesc.VarType.FEED_MINIBATCH,
        persistable=True,
    )

    for i, name in enumerate(feed_target_names):
        if not global_block.has_var(name):
            raise ValueError(
                "The feeded_var_names[{i}]: '{name}' doesn't exist in pruned inference program. "
                "Please check whether '{name}' is a valid feed_var name, or remove it from feeded_var_names "
                "if '{name}' is not involved in the target_vars calculation.".format(
                    i=i, name=name
                )
            )
        out = global_block.var(name)
        global_block._prepend_op(
            type='feed',
            inputs={'X': [feed_var]},
            outputs={'Out': [out]},
            attrs={'col': i},
        )


def append_fetch_ops(
    inference_program, fetch_target_names, fetch_holder_name='fetch'
):
    global_block = inference_program.global_block()
    fetch_var = global_block.create_var(
        name=fetch_holder_name,
        type=core.VarDesc.VarType.FETCH_LIST,
        persistable=True,
    )

    for i, name in enumerate(fetch_target_names):
        global_block.append_op(
            type='fetch',
            inputs={'X': [name]},
            outputs={'Out': [fetch_var]},
            attrs={'col': i},
        )
