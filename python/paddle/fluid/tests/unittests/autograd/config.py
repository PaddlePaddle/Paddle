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

# The numerical tolerance of different dtype of different order different
# derivative. It's a empirical value provided by Paddle Science team.
TOLERANCE = {
    "float32": {
        "first_order_grad": {
            "rtol": 1e-3,
            "atol": 1e-3,
            "eps": 1e-4
        },
        "second_order_grad": {
            "rtol": 1e-2,
            "atol": 1e-2,
            "eps": 1e-2
        }
    },
    "float64": {
        "first_order_grad": {
            "rtol": 1e-7,
            "atol": 1e-7,
            "eps": 1e-7
        },
        "second_order_grad": {
            "rtol": 1e-5,
            "atol": 1e-5,
            "eps": 1e-5
        }
    }
}
