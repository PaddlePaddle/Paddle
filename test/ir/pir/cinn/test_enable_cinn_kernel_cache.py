# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

# test_enable_cinn_kernel_cache.py
# test_enable_cinn_kernel_cache_2.py
# Both tests share identical logic/configuration.
# The first test executes the full compilation and writes the kernel to disk cache,
# while the second verifies the loading of the persistent cache file.

import os
import shutil
import time
import unittest

os.environ['FLAGS_prim_all'] = 'true'
os.environ['FLAGS_prim_enable_dynamic'] = 'true'
os.environ['FLAGS_use_cinn'] = '1'
os.environ['FLAGS_deny_cinn_ops'] = 'slice;'
os.environ['FLAGS_enable_cinn_kernel_cache'] = 'true'

import paddle
from paddle import nn
from paddle.static import InputSpec


class RMSNorm(nn.Layer):
    def __init__(self):
        super().__init__()
        paddle.seed(2024)
        self.hidden_size = 768
        self.drop_prob = 0.1
        self.weight = paddle.randn([self.hidden_size], dtype="float32")
        self.variance_epsilon = 1e-6

    def forward(self, hidden_states):
        variance = (hidden_states * hidden_states).sum(-1, keepdim=True) / 768
        hidden_states = (
            paddle.rsqrt(variance + self.variance_epsilon) * hidden_states
        )
        output = hidden_states * self.weight
        mask = paddle.rand(output.shape) > self.drop_prob
        output = output * mask.astype(output.dtype) / (1.0 - self.drop_prob)

        return output


class TestCase(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def run_net(self, input_data):
        net = RMSNorm()

        input_spec = [
            InputSpec(shape=[1, None, 768], dtype='float32'),
        ]
        net = paddle.jit.to_static(
            net,
            input_spec=input_spec,
            full_graph=True,
        )
        net.eval()
        out = net(input_data)
        return out

    def clear_directory(self, path: str):
        if os.path.exists(path):
            shutil.rmtree(path)
            print(f" Clearing CINN Kernel Cache Path {path}")

    def test_case(self):
        input_data = paddle.randn([1, 2048, 768], dtype="float32")

        save_path = os.environ.get(
            'FLAGS_cinn_kernel_cache_save_path', '/tmp/cinn/'
        )
        print(f" CINN Kernel Cache Path is {save_path}")

        if os.path.exists(save_path):
            start_time_2 = time.perf_counter()
            out2 = self.run_net(input_data)
            end_time_2 = time.perf_counter()
            execution_time_2 = end_time_2 - start_time_2
            print(
                f"--- Execution Time 2 (Cache Load + Run): {execution_time_2:.4f} seconds ---"
            )
            self.clear_directory(save_path)

        else:
            start_time_1 = time.perf_counter()
            out1 = self.run_net(input_data)
            end_time_1 = time.perf_counter()
            execution_time_1 = end_time_1 - start_time_1
            print(
                f"\n--- Execution Time 1 (Compile + First Run): {execution_time_1:.4f} seconds ---"
            )


if __name__ == "__main__":
    unittest.main()
