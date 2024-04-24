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
from paddle import Tensor
from paddle._typing.shape import DynamicShapeLike, ShapeLike, Size2, SizeN

tensor: Tensor = paddle.to_tensor(np.random.randint(1, 10, (1,)))

list_dynamic_shape: DynamicShapeLike = [tensor, 2, None]
tuple_dynamic_shape: DynamicShapeLike = (tensor, 2, None)

list_shape: ShapeLike = [8, 3, 224]
tuple_shape: ShapeLike = (8, 3, 224)

int_size: Size2 = 3
tuple_size: Size2 = (3, 3)
list_size: Size2 = [3, 3]

int_size_n: SizeN = 3
size_n: SizeN = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
