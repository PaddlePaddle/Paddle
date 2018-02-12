#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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
import paddle.v2.fluid.layers as layers
from paddle.v2.fluid.framework import Program, program_guard
import unittest


class TestDetection(unittest.TestCase):
    def test_detection_output(self):
        program = Program()
        with program_guard(program):
            pb = layers.data(
                name='prior_box',
                shape=[10, 4],
                append_batch_size=False,
                dtype='float32')
            pbv = layers.data(
                name='prior_box_var',
                shape=[10, 4],
                append_batch_size=False,
                dtype='float32')
            loc = layers.data(
                name='target_box',
                shape=[10, 4],
                append_batch_size=False,
                dtype='float32')
            scores = layers.data(
                name='scores',
                shape=[2, 20, 10],
                append_batch_size=False,
                dtype='float32')
            out = layers.detection_output(
                scores=scores, loc=loc, prior_box=pb, prior_box_var=pbv)
            self.assertIsNotNone(out)
            self.assertEqual(out.shape[-1], 6)
        #print(str(program))

    def test_ssd_loss(self):
        program = Program()
        with program_guard(program):
            pb = layers.data(
                name='prior_box',
                shape=[10, 4],
                append_batch_size=False,
                dtype='float32')
            pbv = layers.data(
                name='prior_box_var',
                shape=[10, 4],
                append_batch_size=False,
                dtype='float32')
            loc = layers.data(name='target_box', shape=[10, 4], dtype='float32')
            scores = layers.data(name='scores', shape=[10, 21], dtype='float32')
            gt_box = layers.data(
                name='gt_box', shape=[4], lod_level=1, dtype='float32')
            gt_label = layers.data(
                name='gt_label', shape=[1], lod_level=1, dtype='int32')
            loss = layers.ssd_loss(loc, scores, gt_box, gt_label, pb, pbv)
            self.assertIsNotNone(loss)
            self.assertEqual(loss.shape[-1], 1)
        #print(str(program))


class TestPriorBox(unittest.TestCase):
    def test_prior_box(self):
        data_shape = [3, 224, 224]
        box, var = self.prior_box_output(data_shape)

        assert len(box.shape) == 2
        assert box.shape == var.shape
        assert box.shape[1] == 4

    def prior_box_output(self, data_shape):
        images = layers.data(name='pixel', shape=data_shape, dtype='float32')
        conv1 = layers.conv2d(
            input=images,
            num_filters=3,
            filter_size=3,
            stride=2,
            use_cudnn=False)
        conv2 = layers.conv2d(
            input=conv1,
            num_filters=3,
            filter_size=3,
            stride=2,
            use_cudnn=False)
        conv3 = layers.conv2d(
            input=conv2,
            num_filters=3,
            filter_size=3,
            stride=2,
            use_cudnn=False)
        conv4 = layers.conv2d(
            input=conv3,
            num_filters=3,
            filter_size=3,
            stride=2,
            use_cudnn=False)
        conv5 = layers.conv2d(
            input=conv4,
            num_filters=3,
            filter_size=3,
            stride=2,
            use_cudnn=False)

        box, var = layers.prior_box(
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


if __name__ == '__main__':
    unittest.main()
