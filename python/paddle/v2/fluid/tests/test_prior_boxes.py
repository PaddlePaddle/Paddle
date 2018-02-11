#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np
import paddle.v2.fluid as fluid
import paddle.v2.fluid.layers.detection as detection
import paddle.v2.fluid.core as core
import unittest


def prior_box_output(data_shape):
    images = fluid.layers.data(name='pixel', shape=data_shape, dtype='float32')
    conv1 = fluid.layers.conv2d(
        input=images, num_filters=3, filter_size=3, stride=2, use_cudnn=False)
    conv2 = fluid.layers.conv2d(
        input=conv1, num_filters=3, filter_size=3, stride=2, use_cudnn=False)
    conv3 = fluid.layers.conv2d(
        input=conv2, num_filters=3, filter_size=3, stride=2, use_cudnn=False)
    conv4 = fluid.layers.conv2d(
        input=conv3, num_filters=3, filter_size=3, stride=2, use_cudnn=False)
    conv5 = fluid.layers.conv2d(
        input=conv4, num_filters=3, filter_size=3, stride=2, use_cudnn=False)

    box, var = detection.prior_box(
        inputs=[conv1, conv2, conv3, conv4, conv5, conv5],
        image=images,
        min_ratio=20,
        max_ratio=90,
        # steps=[8, 16, 32, 64, 100, 300],
        aspect_ratios=[[2.], [2., 3.], [2., 3.], [2., 3.], [2.], [2.]],
        base_size=300,
        offset=0.5,
        flip=True,
        clip=True)
    return box, var


def main(use_cuda):
    if use_cuda:  # prior_box only support CPU.
        return

    data_shape = [3, 224, 224]
    box, var = prior_box_output(data_shape)

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    batch = [4]  # batch is not used in the prior_box.

    assert box.shape[1] == 4
    assert var.shape[1] == 4
    assert box.shape == var.shape
    assert len(box.shape) == 2

    for _ in range(1):
        x = np.random.random(batch + data_shape).astype("float32")
        tensor_x = core.LoDTensor()
        tensor_x.set(x, place)
        boxes, vars = exe.run(fluid.default_main_program(),
                              feed={'pixel': tensor_x},
                              fetch_list=[box, var])
        assert vars.shape == var.shape
        assert boxes.shape == box.shape


class TestFitALine(unittest.TestCase):
    def test_cpu(self):
        main(use_cuda=False)

    def test_cuda(self):
        main(use_cuda=True)


if __name__ == '__main__':
    unittest.main()
