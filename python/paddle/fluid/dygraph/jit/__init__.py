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

from __future__ import print_function

import logging
from paddle.fluid.framework import in_dygraph_mode
from paddle.fluid.wrapped_decorator import wrap_decorator

from . import traced_layer
from .traced_layer import *

from paddle.fluid.dygraph.jit.program_translator import ProgramTranslator

from . import converter

from . import transformer

logger = logging.getLogger("fluid")

__all__ = traced_layer.__all__ + [
    "ProgramTranslator"
    "declarative", "dygraph_to_static_func"
]


def _dygraph_to_static_func_(dygraph_func):
    """
    Converts imperative dygraph APIs into declarative function APIs. Decorator
    @dygraph_to_static_func only converts imperative dygraph APIs into
    declarative net-building APIs, which means it doesn't return immediate
    digital result as imperative mode. Users should handle Program and Executor
    by themselves.

    Note:
    This decorator is NOT our recommended way to transform imperative function
    to declarative function. We will remove this decorator after we finalize
    cleaning up code.

    Args:
        dygraph_func (callable): callable imperative function.

    Returns:
        Callable: converting imperative dygraph APIs into declarative
        net-building APIs.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          import numpy as np
          from paddle.fluid.dygraph.jit import dygraph_to_static_func

          @dygraph_to_static_func
          def func(x):
              if fluid.layers.mean(x) < 0:
                  x_v = x - 1
              else:
                  x_v = x + 1

               return x_v

          x = fluid.layers.fill_constant(shape=[3, 3], value=0, dtype='float64')

          x_v = func(x)
          exe = fluid.Executor(fluid.CPUPlace())
          out = exe.run(fetch_list=[x_v])
          print(out[0])
          # [[1. 1. 1.]
          #  [1. 1. 1.]
          #  [1. 1. 1.]]

    """

    # TODO: remove this decorator after we finalize training API
    def __impl__(*args, **kwargs):
        program_translator = ProgramTranslator()
        if in_dygraph_mode() or not program_translator.enable_declarative:
            logger.info(
                "The decorator 'dygraph_to_static_func' doesn't work in "
                "dygraph mode or set ProgramTranslator.enable to False. "
                "We will just return dygraph output.")
            return dygraph_func(*args, **kwargs)
        static_func = program_translator.get_func(dygraph_func)
        return static_func(*args, **kwargs)

    return __impl__


dygraph_to_static_func = wrap_decorator(_dygraph_to_static_func_)


def _declarative_(dygraph_func):
    """
    Converts imperative dygraph APIs into declarative function APIs. Decorator
    @declarative handles the Program and Executor of static mode and returns
    the result as a dygraph VarBase.

    Args:
        dygraph_func (callable): callable imperative function.

    Returns:
        VarBase: containing the numerical result.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          import numpy as np
          from paddle.fluid.dygraph.jit import declarative


          @declarative
          def func(x):
              x = fluid.dygraph.to_variable(x)
              if fluid.layers.mean(x) < 0:
                  x_v = x - 1
              else:
                  x_v = x + 1
              return x_v

          x = np.ones([1, 2])
          x_v = func(x)
          print(x_v.numpy()) # [[2. 2.]]

    """

    def __impl__(*args, **kwargs):
        program_translator = ProgramTranslator()
        if not program_translator.enable_declarative:
            logger.info(
                "The decorator 'declarative' doesn't work when setting ProgramTranslator.enable=False. "
                "We will just return dygraph output.")
            return dygraph_func(*args, **kwargs)
        return program_translator.get_output(dygraph_func, *args, **kwargs)

    return __impl__


declarative = wrap_decorator(_declarative_)
