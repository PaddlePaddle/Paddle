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
    async_offload_with_offset,
    async_reload,
    create_async_load,
)


class TestSaveLoadLargeParameters(unittest.TestCase):
    def offload_and_reload(self, data0):
        loader = create_async_load()
        data1 = paddle.randn([10, 10])

        cpu_data, task = async_offload(data0, loader)
        res = paddle.matmul(data1, data1)
        task.cpu_wait()
        gpu_data, task = async_reload(cpu_data, loader)
        res = paddle.matmul(data1, data1)
        task.cuda_wait()
        task.cpu_wait()

        np.testing.assert_array_equal(
            data0.numpy(),
            cpu_data.numpy(),
        )
        np.testing.assert_array_equal(
            data0.numpy(),
            gpu_data.numpy(),
        )

    def test_large_parameters_paddle_save_tensor(self):
        data0 = paddle.randn([10, 5])
        self.offload_and_reload(data0)

    def test_large_parameters_paddle_save_model_weight(self):
        model = paddle.nn.Linear(10, 5)
        data0 = model.weight
        self.offload_and_reload(data0)

    def test_offload_with_offset(self):
        loader = create_async_load()
        data1 = paddle.randn(
            [
                100,
            ]
        )
        data2 = paddle.randn(
            [
                100,
            ]
        ).cpu()
        task1 = async_offload_with_offset(
            src_tensor=data1,
            dst_tensor=data2,
            src_offset=0,
            dst_offset=0,
            offload_size=50,
            async_loader=loader,
        )
        task2 = async_offload_with_offset(
            src_tensor=data1,
            dst_tensor=data2,
            src_offset=50,
            dst_offset=50,
            offload_size=50,
            async_loader=loader,
        )
        task1.cuda_wait()
        task2.cpu_wait()
        np.testing.assert_array_equal(
            data1.numpy(),
            data2.numpy(),
        )


if __name__ == '__main__':
    unittest.main()
