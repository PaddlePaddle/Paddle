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

import unittest
from collections import Counter

from simple_nets import init_data

import paddle
from paddle import base

paddle.enable_static()


def test_trainable():
    x = paddle.static.data(name='image', shape=[-1, 784], dtype='float32')
    label = paddle.static.data(name='label', shape=[-1, 1], dtype='int64')
    feature = paddle.static.nn.fc(
        x, size=10, weight_attr=base.ParamAttr(trainable=False)
    )
    loss = paddle.nn.functional.cross_entropy(
        input=feature, label=label, reduction='none', use_softmax=False
    )
    loss = paddle.mean(loss)
    return loss


class TestTrainable(unittest.TestCase):
    def check_trainable(
        self, model, feed_dict, op_count, optimizer=paddle.optimizer.Adam()
    ):
        place = base.CPUPlace()
        exe = base.Executor(place)

        main = base.Program()
        startup = base.Program()

        with base.program_guard(main, startup):
            loss = model()
            optimizer.minimize(loss)

            # The number of adam should be one.
            ops = Counter([op.type for op in main.global_block().ops])
            for op in op_count:
                if op_count[op] == 0:
                    assert op not in ops
                else:
                    assert ops[op] == op_count[op]

            exe.run(base.default_startup_program())
            exe.run(feed=feed_dict)

    def test_trainable(self):
        batch_size = 2
        img, label = init_data(batch_size, img_shape=[784], label_range=9)
        feed_dict = {'image': img, 'label': label}
        # Note that, because the Weight of FC is not trainable and the x is stop_gradient,
        # so the 'mul_grad' should not be appended.
        self.check_trainable(
            test_trainable,
            feed_dict,
            op_count={'adam': 1, 'scale': 0, 'mul_grad': 0},
        )
        self.check_trainable(
            test_trainable,
            feed_dict,
            op_count={'adamax': 1, 'scale': 1, 'mul_grad': 0},
            optimizer=paddle.optimizer.Adamax(learning_rate=0.2),
        )


if __name__ == '__main__':
    unittest.main()
