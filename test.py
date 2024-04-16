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

depth = 10
x_lod = [[4, 1, 3, 3]]
x = [np.random.randint(0, depth - 1) for i in range(sum(x_lod[0]))]
x = np.array(x).astype('int32').reshape([sum(x_lod[0]), 1])
pd_x = paddle.to_tensor(x)
print(pd_x)

zeros = paddle.to_tensor(np.zeros([11]).astype('int32').reshape([11, 1]))
print(zeros)

expanded = paddle.expand(pd_x, [11, 10])
print(expanded)
# temp_zeros = paddle.to_tensor(np.arange(0, 11, 1, dtype=np.int32).reshape([11, 1]))
# pd_x = paddle.concat([temp_zeros, pd_x], 1)

# print(pd_x)

# zeros = paddle.to_tensor(np.zeros([11, 10]))
# update = paddle.to_tensor(np.ones([11]))
# temp = paddle.scatter_nd_add(zeros, pd_x, update)
# print(temp)
