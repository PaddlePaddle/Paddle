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
from paddle.vision.ops import roi_align, RoIAlign


class TestRoIAlign(unittest.TestCase):
    def setUp(self):
        self.data = np.random.rand(1, 256, 32, 32).astype('float32')
        boxes = np.random.rand(3, 4)
        boxes[:, 2] += boxes[:, 0] + 3
        boxes[:, 3] += boxes[:, 1] + 4
        self.boxes = boxes.astype('float32')
        self.boxes_num = np.array([3], dtype=np.int32)

    def roi_align_functional(self, output_size):
        if isinstance(output_size, int):
            output_shape = (3, 256, output_size, output_size)
        else:
            output_shape = (3, 256, output_size[0], output_size[1])

        if paddle.in_dynamic_mode():
            data = paddle.to_tensor(self.data)
            boxes = paddle.to_tensor(self.boxes)
            boxes_num = paddle.to_tensor(self.boxes_num)

            align_out = roi_align(
                data, boxes, boxes_num=boxes_num, output_size=output_size
            )
            np.testing.assert_equal(align_out.shape, output_shape)

        else:
            data = paddle.static.data(
                shape=self.data.shape, dtype=self.data.dtype, name='data'
            )
            boxes = paddle.static.data(
                shape=self.boxes.shape, dtype=self.boxes.dtype, name='boxes'
            )
            boxes_num = paddle.static.data(
                shape=self.boxes_num.shape,
                dtype=self.boxes_num.dtype,
                name='boxes_num',
            )

            align_out = roi_align(
                data, boxes, boxes_num=boxes_num, output_size=output_size
            )

            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)

            align_out = exe.run(
                paddle.static.default_main_program(),
                feed={
                    'data': self.data,
                    'boxes': self.boxes,
                    'boxes_num': self.boxes_num,
                },
                fetch_list=[align_out],
            )

            np.testing.assert_equal(align_out[0].shape, output_shape)

    def test_roi_align_functional_dynamic(self):
        self.roi_align_functional(3)
        self.roi_align_functional(output_size=(3, 4))

    def test_roi_align_functional_static(self):
        paddle.enable_static()
        self.roi_align_functional(3)
        paddle.disable_static()

    def test_RoIAlign(self):
        roi_align_c = RoIAlign(output_size=(4, 3))
        data = paddle.to_tensor(self.data)
        boxes = paddle.to_tensor(self.boxes)
        boxes_num = paddle.to_tensor(self.boxes_num)

        align_out = roi_align_c(data, boxes, boxes_num)
        np.testing.assert_equal(align_out.shape, (3, 256, 4, 3))

    def test_value(
        self,
    ):
        data = (
            np.array([i for i in range(1, 17)])
            .reshape(1, 1, 4, 4)
            .astype(np.float32)
        )
        boxes = np.array([[1.0, 1.0, 2.0, 2.0], [1.5, 1.5, 3.0, 3.0]]).astype(
            np.float32
        )
        boxes_num = np.array([2]).astype(np.int32)
        output = np.array([[[[6.0]]], [[[9.75]]]], dtype=np.float32)

        data = paddle.to_tensor(data)
        boxes = paddle.to_tensor(boxes)
        boxes_num = paddle.to_tensor(boxes_num)

        roi_align_c = RoIAlign(output_size=1)
        align_out = roi_align_c(data, boxes, boxes_num)
        np.testing.assert_almost_equal(align_out.numpy(), output)


if __name__ == '__main__':
    unittest.main()
