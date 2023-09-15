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
from paddle.base import layers
from paddle.base.executor import Executor, global_scope
from paddle.base.framework import (
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
from paddle.base.compiler import CompiledProgram
from paddle.base.log_helper import get_logger
from . import reader
from . import unique_name
from .reader import *
from . import core
from paddle.utils import deprecated
from paddle.base.framework import static_only

__all__ = reader.__all__


_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s'
)
