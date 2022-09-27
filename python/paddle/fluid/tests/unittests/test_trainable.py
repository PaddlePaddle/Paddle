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

from collections import Counter
import unittest
import paddle
import paddle.fluid as fluid
from simple_nets import init_data


def test_trainable():
    x = fluid.layers.data(name='image', shape=[784], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    feature = fluid.layers.fc(input=x,
                              size=10,
                              param_attr=fluid.ParamAttr(trainable=False))
    loss = fluid.layers.cross_entropy(input=feature, label=label)
    loss = paddle.mean(loss)
    return loss


class TestTrainable(unittest.TestCase):

    def check_trainable(self,
                        model,
                        feed_dict,
                        op_count,
                        optimizer=fluid.optimizer.Adam()):
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)

        main = fluid.Program()
        startup = fluid.Program()

        with fluid.program_guard(main, startup):
            loss = model()
            optimizer.minimize(loss)

            # The number of adam should be one.
            ops = Counter([op.type for op in main.global_block().ops])
            for op in op_count:
                if op_count[op] == 0:
                    assert op not in ops
                else:
                    assert ops[op] == op_count[op]

            exe.run(fluid.default_startup_program())
            exe.run(feed=feed_dict)

    def test_trainable(self):
        batch_size = 2
        img, label = init_data(batch_size, img_shape=[784], label_range=9)
        feed_dict = {'image': img, 'label': label}
        # Note that, because the Weight of FC is not trainable and the x is stop_gradient,
        # so the 'mul_grad' should not be appended.
        self.check_trainable(test_trainable,
                             feed_dict,
                             op_count={
                                 'adam': 1,
                                 'scale': 0,
                                 'mul_grad': 0
                             })
        self.check_trainable(
            test_trainable,
            feed_dict,
            op_count={
                'adamax': 1,
                'scale': 1,
                'mul_grad': 0
            },
            optimizer=fluid.optimizer.Adamax(learning_rate=0.2))


if __name__ == '__main__':
    unittest.main()
