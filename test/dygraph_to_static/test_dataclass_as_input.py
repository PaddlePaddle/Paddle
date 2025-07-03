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
from dataclasses import dataclass

import numpy as np
from dygraph_to_static_utils import (
    Dy2StTestBase,
)

import paddle


@dataclass
class Data:
    x: paddle.Tensor


def accept_dataclass_as_input(data: Data):
    return data.x + 1


class TestDataclassAsInput(Dy2StTestBase):
    def test_dataclass_as_input(self):
        x = paddle.to_tensor(np.random.rand(3, 4).astype('float32'))
        data = Data(x)
        dy_out = accept_dataclass_as_input(data)
        static_fn = paddle.jit.to_static(accept_dataclass_as_input)
        st_out = static_fn(data)
        np.testing.assert_allclose(
            dy_out.numpy(),
            st_out.numpy(),
        )


if __name__ == "__main__":
    unittest.main()
