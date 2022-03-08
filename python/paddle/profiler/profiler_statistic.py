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
import collections
from enum import Enum

from paddle.fluid.core import TracerEventType


class SortedKeys(Enum):
    r"""
    Sorted keys for printing summary table.
    """
    CPUTotal = 0
    CPUAvg = 1
    CPUMax = 2
    CPUMin = 3
    GPUTotal = 4
    GPUAvg = 5
    GPUMax = 6
    GPUMin = 7
