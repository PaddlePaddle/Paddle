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

DEVICES = [paddle.CPUPlace()]
if paddle.is_compiled_with_cuda():
    DEVICES.append(paddle.CUDAPlace(0))

DEFAULT_DTYPE = 'float64'

TEST_CASE_NAME = 'suffix'
# All test case will use float64 for compare percision, refs:
# https://github.com/PaddlePaddle/Paddle/wiki/Upgrade-OP-Precision-to-Float64
RTOL = {
    'float32': 1e-03,
    'complex64': 1e-3,
    'float64': 1e-5,
    'complex128': 1e-5
}
ATOL = {'float32': 0.0, 'complex64': 0, 'float64': 0.0, 'complex128': 0}
