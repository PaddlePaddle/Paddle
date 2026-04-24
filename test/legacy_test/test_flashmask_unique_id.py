# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.base import core


@unittest.skipIf(
    not core.is_compiled_with_cuda(),
    "flashmask_get_unique_id requires CUDA",
)
class TestFlashmaskGetUniqueId(unittest.TestCase):
    """paddle.nn.functional.flashmask_get_unique_id returns a uint8 CPU tensor
    This tensor can either be all-zero (WITH_NVSHMEM=OFF) or valid NVSHMEM unique_id
    """

    def test_unique_id_tensor_properties(self):
        unique_id = paddle.nn.functional.flashmask_get_unique_id()
        self.assertTrue(
            unique_id.place.is_cpu_place(), "unique_id should be on CPU"
        )
        self.assertEqual(
            unique_id.dtype, paddle.uint8, "unique_id dtype should be uint8"
        )
        self.assertEqual(
            unique_id.numel(), 128, "unique_id numel should be 128"
        )


if __name__ == "__main__":
    unittest.main()
