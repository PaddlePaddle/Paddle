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

import unittest
import numpy as np

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core


class TestDLPack(unittest.TestCase):
    def test_dlpack_dygraph(self):
        tensor = paddle.to_tensor(np.array([1, 2, 3, 4]).astype('int'))
        dlpack = paddle.utils.dlpack.to_dlpack(tensor)
        out_from_dlpack = paddle.utils.dlpack.from_dlpack(dlpack)
        self.assertTrue(isinstance(out_from_dlpack, paddle.Tensor))
        self.assertTrue(
            np.array_equal(
                np.array(out_from_dlpack), np.array([1, 2, 3, 4]).astype(
                    'int')))

    def test_dlpack_static(self):
        paddle.enable_static()
        tensor = fluid.create_lod_tensor(
            np.array([[1], [2], [3], [4]]).astype('int'), [[1, 3]],
            fluid.CPUPlace())
        dlpack = paddle.utils.dlpack.to_dlpack(tensor)
        out_from_dlpack = paddle.utils.dlpack.from_dlpack(dlpack)
        self.assertTrue(isinstance(out_from_dlpack, fluid.core.Tensor))
        self.assertTrue(
            np.array_equal(
                np.array(out_from_dlpack),
                np.array([[1], [2], [3], [4]]).astype('int')))

        # when build with cuda
        if core.is_compiled_with_cuda():
            gtensor = fluid.create_lod_tensor(
                np.array([[1], [2], [3], [4]]).astype('int'), [[1, 3]],
                fluid.CUDAPlace(0))
            gdlpack = paddle.utils.dlpack.to_dlpack(gtensor)
            gout_from_dlpack = paddle.utils.dlpack.from_dlpack(gdlpack)
            self.assertTrue(isinstance(gout_from_dlpack, fluid.core.Tensor))
            self.assertTrue(
                np.array_equal(
                    np.array(gout_from_dlpack),
                    np.array([[1], [2], [3], [4]]).astype('int')))


if __name__ == '__main__':
    unittest.main()
