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

        out = tensor.clamp(images, min=min, max=max)
        clamp_out = exe.run(fluid.default_main_program(),
                            feed={
                                "image": data,
                                "min": np.array([0.2]).astype('float32'),
                                "max": np.array([0.8]).astype('float32')
                            },
                            fetch_list=[out])

        self.assertTrue(np.allclose(clamp_out, np.clip(data, 0.2, 0.8)))


if __name__ == '__main__':
    unittest.main()
