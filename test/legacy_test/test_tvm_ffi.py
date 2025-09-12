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

import unittest

import paddle


class TestTVMFFI(unittest.TestCase):
    def test_tvm_ffi_env_stream_for_gpu_tensor(self):
        if not paddle.is_compiled_with_cuda():
            return
        tensor = paddle.to_tensor([1.0, 2.0, 3.0]).cuda()
        current_raw_stream_ptr = tensor.__tvm_ffi_env_stream__()
        self.assertIsInstance(current_raw_stream_ptr, int)
        self.assertNotEqual(current_raw_stream_ptr, 0)

    def test_tvm_ffi_env_stream_for_cpu_tensor(self):
        tensor = paddle.to_tensor([1.0, 2.0, 3.0]).cpu()
        with self.assertRaisesRegex(
            RuntimeError, r"the __tvm_ffi_env_stream__ method"
        ):
            tensor.__tvm_ffi_env_stream__()


if __name__ == '__main__':
    unittest.main()
