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
    Sorted keys for printing op summary table.
    """
    OpTotal = 0
    OpAvg = 1
    OpMax = 2
    OpMin = 3
    KernelTotal = 4
    KernelAvg = 5
    KernelMax = 6
    KernelMin = 7
