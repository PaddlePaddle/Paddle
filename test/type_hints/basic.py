# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np

import paddle
from paddle._typing import (
    IntSequence,
    NestedNumbericSequence,
    Numberic,
    NumbericSequence,
    TensorOrTensors,
)

np_num: Numberic = np.array(0)[0]
tensor: Numberic = paddle.to_tensor(np_num)
num_int: Numberic = 1
num_float: Numberic = 2.0
num_complex: Numberic = complex(1, 2)

list_int_seq: IntSequence = [0, 1, 2]
tuple_int_seq: IntSequence = (0, 1, 2)
any_seq: NumbericSequence = [np_num, num_complex]

nested_any_seq: NestedNumbericSequence = [
    [np_num, tensor],
    [num_float, num_int],
]

ten = paddle.to_tensor(np_num)
tensors: TensorOrTensors = [ten]
