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

import sys
from .reductions import init_reductions
import multiprocessing

__all__ = []

from multiprocessing import *  # noqa: F403

__all__ += multiprocessing.__all__  # type: ignore[attr-defined]

# TODO add paddle c++ multiprocessing init, adds a Linux specific prctl(2) wrapper function
# TODO:add spawn methods

# Only support linux for now
# Only support file_system sharing strategy.
# TODO: support file_descriptor on linux

init_reductions()
