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

import os
import sys
import subprocess

cur_dir = os.path.dirname(os.path.abspath(__file__))
if os.name == 'nt':
    exit()
else:
    cmd = 'cd {} && {} custom_kernel_dot_setup.py build_ext --inplace'.format(
        cur_dir, sys.executable)
subprocess.check_call(cmd, shell=True, stderr=subprocess.STDOUT)

os.environ['CUSTOM_DEVICE_ROOT'] = cur_dir

print(os.environ.get('CUSTOM_DEVICE_ROOT'))

import paddle
import numpy as np

x_data = np.random.uniform(1, 5, [3]).astype(np.int8)
y_data = np.random.uniform(1, 3, [3]).astype(np.int8)

np_z = np.dot(x_data, y_data)

x = paddle.to_tensor(x_data)
y = paddle.to_tensor(y_data)

z = paddle.dot(x, y)
print(np.allclose(z.numpy(), np_z))
