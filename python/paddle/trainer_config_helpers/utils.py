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

from paddle.trainer.config_parser import logger
import functools

__all__ = ['deprecated']


def deprecated(instead):
    def __impl__(func):
        @functools.wraps(func)
        def __wrapper__(*args, **kwargs):
            logger.warning("The interface %s is deprecated, "
                           "will be removed soon. Please use %s instead." %
                           (func.__name__, instead))

            return func(*args, **kwargs)

        return __wrapper__

    return __impl__
