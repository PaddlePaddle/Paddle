#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import os
import copy
import unittest
import numpy as np

import paddle
import paddle.fluid.core as core
from paddle.vision.ops import mirror_normalize


def np_mirror_normalize(image, mirror, mean, std):
    image = copy.deepcopy(image)
    for i, m in enumerate(mirror):
        if m[0]:
            image[i] = image[i][:, :, -1::-1]

    mean = np.array(mean)
    std = np.array(std)
    if np.size(mean) == 1:
        mean = np.tile(mean, (3, ))
    if np.size(std) == 1:
        std = np.tile(std, (3, ))

    mean = np.array(mean[:]).reshape([1, 3, 1, 1])
    std = np.array(std[:]).reshape([1, 3, 1, 1])

    return (image - mean) / std


class TestMirrorNormalize(unittest.TestCase):
    def setUp(self):
        self.image_shape = [16, 3, 32, 32]
        self.mirror_shape = [16, 1]
        self.mean = [123.675, 116.28, 103.53]
        self.std = [58.395, 57.120, 57.375]

        self.image = np.random.randint(0, 256, self.image_shape,
                                       'int32').astype("float32")
        self.mirror = np.random.randint(0, 2, self.mirror_shape,
                                        'int32').astype("bool")

        self.result = np_mirror_normalize(self.image, self.mirror, self.mean,
                                          self.std)

    def test_check_output_dynamic(self):
        # NOTE: only supoort CUDA kernel currently
        if not core.is_compiled_with_cuda():
            return

        dy_result = mirror_normalize(
            paddle.to_tensor(self.image),
            paddle.to_tensor(self.mirror), self.mean, self.std)
        assert np.allclose(self.result, dy_result.numpy())

    def test_check_output_static(self):
        # NOTE: only supoort CUDA kernel currently
        if not core.is_compiled_with_cuda():
            return

        paddle.enable_static()

        image_data = paddle.static.data(
            shape=self.image_shape, dtype='float32', name="image")
        mirror_data = paddle.static.data(
            shape=self.mirror_shape, dtype='bool', name="mirror")
        result_data = mirror_normalize(image_data, mirror_data, self.mean,
                                       self.std)

        # NOTE: only supoort CUDA kernel currently
        places = []
        if core.is_compiled_with_cuda():
            places.append(paddle.CUDAPlace(0))

        for place in places:
            exe = paddle.static.Executor(place)
            st_result = exe.run(
                paddle.static.default_main_program(),
                feed={"image": self.image,
                      "mirror": self.mirror},
                fetch_list=[result_data])

            assert np.allclose(self.result, st_result)

        paddle.disable_static()


class TestMirrorNormalizeSingleMeanStd(TestMirrorNormalize):
    def setUp(self):
        self.image_shape = [16, 3, 32, 32]
        self.mirror_shape = [16, 1]
        self.mean = [123.675]
        self.std = [58.395]

        self.image = np.random.randint(0, 256, self.image_shape,
                                       'int32').astype("float32")
        self.mirror = np.random.randint(0, 2, self.mirror_shape,
                                        'int32').astype("bool")

        self.result = np_mirror_normalize(self.image, self.mirror, self.mean,
                                          self.std)


class TestMirrorNormalizeFloatMeanStd(TestMirrorNormalize):
    def setUp(self):
        self.image_shape = [16, 3, 32, 32]
        self.mirror_shape = [16, 1]
        self.mean = 123.675
        self.std = 58.395

        self.image = np.random.randint(0, 256, self.image_shape,
                                       'int32').astype("float32")
        self.mirror = np.random.randint(0, 2, self.mirror_shape,
                                        'int32').astype("bool")

        self.result = np_mirror_normalize(self.image, self.mirror, self.mean,
                                          self.std)


if __name__ == '__main__':
    unittest.main()
