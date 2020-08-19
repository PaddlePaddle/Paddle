# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import threading

import six
from paddle.fluid import log_helper
from paddle.fluid.dygraph.dygraph_to_static.utils import ast_to_source_code

__all__ = ["TranslatorLogger", "set_verbosity", "set_code_level"]

VERBOSITY_ENV_NAME = 'TRANLATOR_VERBOSITY'
DEFAULT_VERBOSITY = 0


def synchronized(func):
    def wrapper(*args, **kwargs):
        with threading.Lock():
            return func(*args, **kwargs)

    return wrapper


class TranslatorLogger(object):
    """
    class for Logging and debugging during the tranformation from dygraph to static graph.
    The object of this class is a singleton.
    """

    @synchronized
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            cls._instance = object.__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._logger = log_helper.get_logger(
            __name__, 1, fmt='%(asctime)s-%(levelname)s: %(message)s')
        self._verbosity_level = None
        self._transformed_code_level = None

    @property
    def logger(self):
        return self._logger

    @property
    def verbosity_level(self):
        if self._verbosity_level is not None:
            return self._verbosity_level
        else:
            return int(os.getenv(VERBOSITY_ENV_NAME, DEFAULT_VERBOSITY))

    @verbosity_level.setter
    def verbosity_level(self, level):
        self.check_level(level, {})
        self._verbosity_level = level

    @property
    def transformed_code_level(self):
        if self._transformed_code_level is not None:
            return self._transformed_code_level
        else:
            return -1

    @transformed_code_level.setter
    def transformed_code_level(self, level):
        self.check_level(level, _transformer_name_to_level)
        self._transformed_code_level = level

    def check_level(self, level, name_to_level_dict):
        if isinstance(level, six.integer_types):
            rv = level
        elif str(level) == level:
            if level not in name_to_level_dict:
                raise ValueError("Unknown level: %r" % level)
            rv = name_to_level_dict[level]
        else:
            raise TypeError("Level not an integer or a valid string: %r" %
                            level)
        return rv

    def has_code_level(self, level):
        level = self.check_level(level, _transformer_name_to_level)
        return level == self.transformed_code_level

    def has_verbosity(self, level):
        level = self.check_level(level, {})
        return level >= self.verbosity_level

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    def warn(self, msg, *args, **kwargs):
        self.logger.warn(msg, *args, **kwargs)

    def log(self, level, msg, *args, **kwargs):
        if self.has_verbosity(level):
            self.logger.log(level, msg, *args, **kwargs)

    def log_transformed_code(self, level, ast_node, *args, **kwargs):
        if self.has_code_level(level):
            msg = ast_to_source_code(ast_node)
            msg = "Transformed code \n" + msg
            self.logger.info(msg, *args, **kwargs)


_TRANSLATOR_LOGGER = TranslatorLogger()


def set_verbosity(level=0):
    """
    Sets the verbosity level for dygraph to static graph.

    There are two means to set the logging verbosity:
     1. Call function `set_verbosity`
     2. Set environment variable `verbosity_level`
    NOTE: `set_verbosity` has higher priority than the environment variable

    Args:
        level(int): The verbosity level. The larger value idicates more verbosity.
            The default value is 0, which means no logging.
    Examples:
        .. code-block:: python

            import os
            import paddle.fluid as fluid

            fluid.dygraph.dygraph_to_static.set_verbosity(1)
            # The verbosity level is now 1

            os.environ['verbosity_level'] = 3
            # The verbosity level is now 3, but it has no effect
    """
    _TRANSLATOR_LOGGER.verbosity_level = level


BasicApiTransformer = 1
TensorShapeTransformer = 2
ListTransformer = 3
BreakContinueTransformer = 4
ReturnTransformer = 5
LogicalTransformer = 6
LoopTransformer = 7
IfElseTransformer = 8
AssertTransformer = 9
PrintTransformer = 10
CallTransformer = 11
CastTransformer = 12
AllTransformer = 12

_transformer_name_to_level = {
    'BasicApiTransformer': BasicApiTransformer,
    'TensorShapeTransformer': TensorShapeTransformer,
    'ListTransformer': ListTransformer,
    'BreakContinueTransformer': BreakContinueTransformer,
    'ReturnTransformer': ReturnTransformer,
    'LogicalTransformer': LogicalTransformer,
    'LoopTransformer': LoopTransformer,
    'IfElseTransformer': IfElseTransformer,
    'AssertTransformer': AssertTransformer,
    'PrintTransformer': PrintTransformer,
    'CallTransformer': CallTransformer,
    'CastTransformer': CastTransformer,
    'AllTransformer': AllTransformer
}


def set_code_level(level):
    """
    Sets the level to print code from specific Ast Transformer.

    Args:
        level(int): The level to print code.

    Examples:
        .. code-block:: python

            import os
            import paddle.fluid as fluid
            from paddle.fluid.dygraph.dygraph_to_static import logging_utils

            logging_utils.set_code_level(logging_utils.CastTransformer)
            # It will print the transformed code after CastTransformer.
        """
    _TRANSLATOR_LOGGER.transformed_code_level = level
