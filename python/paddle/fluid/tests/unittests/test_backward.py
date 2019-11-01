# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import paddle.fluid as fluid
from simple_nets import init_data


def case1_fill_grad_vars():
    x = fluid.layers.data(name='image', shape=[784], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    feature = fluid.layers.fc(input=x, size=20, act=None)
    part1, part2 = fluid.layers.split(feature, num_or_sections=[10, 10], dim=1)
    # Note that: part2 is not used.
    loss = fluid.layers.cross_entropy(input=part1, label=label)
    loss = fluid.layers.mean(loss)
    return loss


def case2_prune_no_grad_branch():
    x = fluid.layers.data(name='image', shape=[784], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    feature = fluid.layers.fc(input=x, size=10, act=None)
    label = fluid.layers.cast(label, dtype="float32")
    label = fluid.layers.cast(label, dtype='int64')
    # Note that the label is not persistable in fluid.layers.cross_entropy.
    loss = fluid.layers.cross_entropy(input=feature, label=label)
    loss = fluid.layers.mean(loss)
    return loss


def case3_prune_no_grad_branch2():
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    label = fluid.layers.cast(label, dtype="float32")
    label = fluid.layers.cast(label, dtype='int64')
    out = fluid.layers.one_hot(input=label, depth=100)
    loss = fluid.layers.mean(out)
    return loss


def case4_with_no_grad_op_maker():
    out = fluid.layers.gaussian_random(shape=[20, 30])
    loss = fluid.layers.mean(out)
    return loss


class TestBackward(unittest.TestCase):
    def check_backward(self, model, feed_dict):
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)

        main = fluid.Program()
        startup = fluid.Program()

        with fluid.program_guard(main, startup):
            loss = model()

            optimizer = fluid.optimizer.SGD(learning_rate=0.1)
            optimizer.minimize(loss)

            exe.run(fluid.default_startup_program())
            exe.run(feed=feed_dict)

    def test_backward(self):
        batch_size = 2
        img, label = init_data(batch_size, img_shape=[784], label_range=9)
        feed_dict = {'image': img, 'label': label}
        self.check_backward(case1_fill_grad_vars, feed_dict)
        self.check_backward(case2_prune_no_grad_branch, feed_dict)
        self.check_backward(case3_prune_no_grad_branch2, {'label': label})
        self.check_backward(case4_with_no_grad_op_maker, {})


if __name__ == '__main__':
    unittest.main()
