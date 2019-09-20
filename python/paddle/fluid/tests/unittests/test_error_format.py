#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import numpy
import os
import paddle
import paddle.compat as cpt
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid import error_format
import unittest


def _make_enforce_not_met_exception():
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    x = fluid.layers.data(name='X', shape=[12], dtype='float32')
    y = fluid.layers.data(name='Y', shape=[1], dtype='float32')
    y_ = fluid.layers.fc(input=x, size=1, act=None)
    loss = fluid.layers.square_error_cost(input=y_, label=y)
    avg_loss = fluid.layers.mean(loss)
    fluid.optimizer.SGD(learning_rate=0.01).minimize(avg_loss)

    exe.run(fluid.default_startup_program())
    x = numpy.random.random(size=(10, 13)).astype('float32')
    y = numpy.random.random(size=(10, 1)).astype('float32')
    loss_data, = exe._run_impl(
        program=fluid.default_main_program(),
        feed={'X': x,
              'Y': y},
        fetch_list=[avg_loss.name],
        feed_var_name='feed',
        fetch_var_name='fetch',
        scope=None,
        return_numpy=True,
        use_program_cache=False)


class TestErrorMessageHintAugment(unittest.TestCase):
    def setUp(self):
        self.exception = None
        try:
            _make_enforce_not_met_exception()
        except core.EnforceNotMet as ex:
            self.exception = ex

        self.assertIsNotNone(self.exception)
        self.ex_msg = cpt.get_exception_message(self.exception)

    def augment_result_check(self, origin, augment_str):
        origin_augment = error_format.hint_augment(origin)
        origin_lines = origin.splitlines()
        origin_augment_lines = origin_augment.splitlines()
        origin_lines.reverse()
        origin_augment_lines.reverse()
        self.assertEqual(origin_augment_lines[0].strip(), augment_str)
        self.assertEqual("".join(origin_augment_lines[1:]),
                         "".join(origin_lines[1:]))

    def test_hint_augment_success(self):
        ex_msg = self.ex_msg
        augment_str = "[[operator mul execution error (defined at /usr/lib/python3.5/unittest/main.py:94)]]"
        self.augment_result_check(ex_msg, augment_str)

    def test_hint_augment_no_tag(self):
        ex_msg_lines = self.ex_msg.splitlines()
        # remove tag
        ex_msg_no_tag = "\n".join(ex_msg_lines[:-1])
        ex_msg_augment = error_format.hint_augment(ex_msg_no_tag)
        self.assertEqual(ex_msg_augment, ex_msg_no_tag)

    def test_hint_augment_no_frame(self):
        ex_msg_lines = self.ex_msg.splitlines()
        # remove frame
        ex_msg_no_frame = "\n".join(ex_msg_lines[-5:])
        augment_str = "[[operator mul error]]"
        self.augment_result_check(ex_msg_no_frame, augment_str)


if __name__ == "__main__":
    unittest.main()
