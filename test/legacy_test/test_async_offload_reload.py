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


import unittest

import numpy as np

import paddle
from paddle.incubate.tensor.manipulation import (
    async_offload,
    async_reload,
    create_async_load,
)


class TestSaveLoadLargeParameters(unittest.TestCase):
    def test_large_parameters_paddle_save(self):
        loader = create_async_load()
        data0 = paddle.randn([10000, 50])
        data1 = paddle.randn([50, 50])

        cpu_data, task = async_offload(data0, loader)
        res = paddle.matmul(data1, data1)
        task.wait()
        gpu_data, task = async_reload(cpu_data, loader)
        res = paddle.matmul(data1, data1)
        task.wait()

        np.testing.assert_array_equal(
            data0.numpy(),
            cpu_data.numpy(),
        )
        np.testing.assert_array_equal(
            data0.numpy(),
            gpu_data.numpy(),
        )


if __name__ == '__main__':
    unittest.main()
