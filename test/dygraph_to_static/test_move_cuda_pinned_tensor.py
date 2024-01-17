# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.jit.dy2static.utils import (
    cuda_pinned_tensors_move_to_excepted_place,
)


class TestCopyCudaPinnedTensors(unittest.TestCase):
    def test_copy_cuda_pinned_tensors(self):
        if paddle.device.is_compiled_with_cuda():

            @paddle.jit.to_static
            def f(x):
                return x

            cuda_pinned_place = paddle.CUDAPinnedPlace()
            x = paddle.to_tensor(
                [1, 2, 3], place=paddle.CUDAPinnedPlace(), stop_gradient=True
            )

            cuda_pinned_tensors_move_to_excepted_place(x)
            assert not x.place._equals(cuda_pinned_place)

            y = {
                "a": [
                    paddle.to_tensor(
                        1, place=paddle.CUDAPinnedPlace(), stop_gradient=True
                    ),
                    paddle.to_tensor(
                        2, place=paddle.CUDAPinnedPlace(), stop_gradient=True
                    ),
                ],
                "b": {
                    "c": paddle.to_tensor(
                        3, place=paddle.CUDAPinnedPlace(), stop_gradient=True
                    )
                },
            }
            cuda_pinned_tensors_move_to_excepted_place(y)
            for var in paddle.utils.flatten(y):
                if isinstance(var, paddle.Tensor):
                    assert not var.place._equals(cuda_pinned_place)


if __name__ == '__main__':
    unittest.main()
