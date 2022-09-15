# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import logging
import sys

from paddle.distributed.utils.log_utils import get_logger

logger = get_logger("INFO", __name__)


def set_log_level(level):
    """
    Set log level

    Args:
        level (str|int): a specified level

    Example 1:
        import paddle
        import paddle.distributed.fleet as fleet
        fleet.init()
        fleet.setLogLevel("DEBUG")

    Example 2:
        import paddle
        import paddle.distributed.fleet as fleet
        fleet.init()
        fleet.setLogLevel(1)

    """
    assert isinstance(level, (str, int)), "level's type must be str or int"
    if isinstance(level, int):
        logger.setLevel(level)
    else:
        logger.setLevel(level.upper())


def get_log_level_code():
    """
    Return current log level code
    """
    return logger.getEffectiveLevel()


def get_log_level_name():
    """
    Return current log level name
    """
    return logging.getLevelName(get_log_level_code())


def layer_to_str(base, *args, **kwargs):
    name = base + "("
    if args:
        name += ", ".join(str(arg) for arg in args)
        if kwargs:
            name += ", "
    if kwargs:
        name += ", ".join("{}={}".format(key, str(value))
                          for key, value in kwargs.items())
    name += ")"
    return name
