#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function
import paddle.tensor as tensor
import paddle.fluid as fluid
import numpy as np
import unittest


class TestClampAPI(unittest.TestCase):
    def test_clamp(self):
        data_shape = [1, 9, 9, 4]
        data = np.random.random(data_shape).astype('float32')
        images = fluid.data(name='image', shape=data_shape, dtype='float32')
        min = fluid.data(name='min', shape=[1], dtype='float32')
        max = fluid.data(name='max', shape=[1], dtype='float32')

        place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        exe = fluid.Executor(place)

        out_1 = tensor.clamp(images, min=min, max=max)
        out_2 = tensor.clamp(images, min=0.2, max=0.9)
        out_3 = tensor.clamp(images, min=0.3)
        out_4 = tensor.clamp(images, max=0.7)
        out_5 = tensor.clamp(images, min=min)
        out_6 = tensor.clamp(images, max=max)

        res1, res2, res3, res4, res5, res6 = exe.run(
            fluid.default_main_program(),
            feed={
                "image": data,
                "min": np.array([0.2]).astype('float32'),
                "max": np.array([0.8]).astype('float32')
            },
            fetch_list=[out_1, out_2, out_3, out_4, out_5, out_6])

        self.assertTrue(np.allclose(res1, data.clip(0.2, 0.8)))
        self.assertTrue(np.allclose(res2, data.clip(0.2, 0.9)))
        self.assertTrue(np.allclose(res3, data.clip(min=0.3)))
        self.assertTrue(np.allclose(res4, data.clip(max=0.7)))
        self.assertTrue(np.allclose(res5, data.clip(min=0.2)))
        self.assertTrue(np.allclose(res6, data.clip(max=0.8)))


class TestClampError(unittest.TestCase):
    def test_errors(self):
        x1 = fluid.layers.data(name='x1', shape=[1], dtype="int16")
        x2 = fluid.layers.data(name='x2', shape=[1], dtype="int8")
        self.assertRaises(TypeError, tensor.clamp, x=x1, min=0.2, max=0.8)
        self.assertRaises(TypeError, tensor.clamp, x=x2, min=0.2, max=0.8)


if __name__ == '__main__':
    unittest.main()
