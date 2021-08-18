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

import paddle
import numpy as np
import paddle.fluid as fluid
'''
x = paddle.to_tensor(np.ones(3))
dlpack = paddle.utils.dlpack.to_dlpack(x)
y = paddle.utils.dlpack.from_dlpack(dlpack)

print(x == y)
'''

paddle.enable_static()
tensor = fluid.create_lod_tensor(
    np.array([[1], [2], [3], [4]]).astype('int'), [[1, 3]], fluid.CPUPlace())
x = paddle.utils.dlpack.to_dlpack(tensor)
