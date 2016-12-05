# Copyright (c) 2016 Baidu, Inc. All Rights Reserved
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

from paddle.trainer.config_parser import logger
import functools

__all__ = ['deprecated', "deprecated_msg"]


def deprecated_msg(msg):
    def __impl__(func):
        def __wrapper__(*args, **kwargs):
            logger.warning("The interface %s is deprecated. %s" %
                           (func.__name__, msg))
            return func(*args, **kwargs)

        return __wrapper__

    return __impl__


def deprecated(instead):
    return deprecated_msg("Please use %s instead." % instead)
