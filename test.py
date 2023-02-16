# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

paddle.enable_static()

x = paddle.static.data(name="x", shape=[2, 4], dtype=np.int64)
output = paddle.static.nn.embedding(
    x, (10, 3), param_attr=paddle.nn.initializer.Constant(value=1.0)
)
m_output = paddle.mean(output)

place = paddle.CPUPlace()
exe = paddle.static.Executor(place)
exe.run(paddle.static.default_startup_program())

x = np.array([[7, 2, 4, 5], [4, 3, 2, 9]], dtype=np.int64)

# x is a Numpy.
# x.data = [[7, 2, 4, 5], [4, 3, 2, 9]]
# x.shape = [2, 4]

(out,) = exe.run(
    paddle.static.default_main_program(), feed={'x': x}, fetch_list=[output]
)
