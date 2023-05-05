# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from .reductions import init_reductions
import multiprocessing
import sys

__all__ = []

from multiprocessing import *  # noqa: F403

# Only support linux for now
if sys.platform == 'darwin' or sys.platform == 'win32':
    _sharing_strategy = 'file_system'
else:
    _sharing_strategy = 'file_descriptor'

init_reductions(_sharing_strategy)


def set_sharing_strategy(sharing_strategy):
    if (
        sharing_strategy != "file_descriptor"
        and sharing_strategy != "file_system"
    ):
        raise RuntimeError(
            "We only support file_system mode and file_descriptor mode"
        )
    else:
        _sharing_strategy = sharing_strategy
        init_reductions(_sharing_strategy)


def get_sharing_strategy():
    return _sharing_strategy
