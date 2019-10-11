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

import multiprocessing
import numpy as np
import os
import paddle
import paddle.fluid as fluid
import paddle.fluid.compiler as compiler
import paddle.fluid.core as core
import six
import unittest

os.environ['CPU_NUM'] = str(4)
np.random.seed(123)


class TestFeedData(unittest.TestCase):
    '''
        Test paddle.fluid.data feeds with different shape and types.
        Note: paddle.fluid.data is not paddle.fluid.layers.data.
    '''

    def setUp(self):
        self.hidden_sizes = [25, 20, 15]
        self.base_batch_size = 10
        self.class_num = 10
        self.iterations = 5

    def _get_batch_size(self, use_cuda, use_parallel_executor):
        batch_size_times = 1
        if use_parallel_executor:
            batch_size_times = core.get_cuda_device_count(
            ) if use_cuda else int(
                os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
        return self.base_batch_size * batch_size_times

    def _simple_fc_net(self, in_size, label_size, class_num, hidden_sizes):
        in_data = fluid.data(name="data", dtype='float32', shape=in_size)
        label = fluid.data(name='label', dtype='int64', shape=label_size)

        hidden = in_data
        for hidden_size in hidden_sizes:
            hidden = fluid.layers.fc(hidden, size=hidden_size)

        predict_label = fluid.layers.fc(hidden, size=class_num, act='softmax')
        loss = fluid.layers.mean(
            fluid.layers.cross_entropy(
                input=predict_label, label=label))

        optimizer = fluid.optimizer.Adam()
        optimizer.minimize(loss)
        return in_data, label, loss

    def test(self):
        for use_cuda in [True,
                         False] if core.is_compiled_with_cuda() else [False]:
            for use_parallel_executor in [False, True]:
                print('Test Parameters:'),
                print({
                    'use_cuda': use_cuda,
                    'use_parallel_executor': use_parallel_executor,
                })
                # Test feeding without error
                self._test_feed_data_match_shape_type(use_cuda,
                                                      use_parallel_executor)
                self._test_feed_data_contains_neg_one(use_cuda,
                                                      use_parallel_executor)

                # Test exception message when feeding with error 
                batch_size = self._get_batch_size(use_cuda,
                                                  use_parallel_executor)
                if six.PY2:
                    in_shape_tuple = (long(-1), long(3), long(4), long(8))
                    feed_shape_list = [
                        long(batch_size), long(3), long(4), long(5)
                    ]
                else:
                    in_shape_tuple = (-1, 3, 4, 8)
                    feed_shape_list = [batch_size, 3, 4, 5]

                with self.assertRaises(ValueError) as shape_mismatch_err:
                    self._test_feed_data_shape_mismatch(use_cuda,
                                                        use_parallel_executor)
                self.assertEqual(
                    str(shape_mismatch_err.exception),
                    "The feeded Variable %r should have dimensions = %r, "
                    "shape = %r, but received feeded shape %r" %
                    (u'data', len(in_shape_tuple), in_shape_tuple,
                     feed_shape_list))

                with self.assertRaises(ValueError) as dtype_mismatch_err:
                    self._test_feed_data_dtype_mismatch(use_cuda,
                                                        use_parallel_executor)
                self.assertEqual(
                    str(dtype_mismatch_err.exception),
                    "The data type of feeded Variable %r must be 'int64', but "
                    "received 'float64'" % (u'label'))

    def _test_feed_data_dtype_mismatch(self, use_cuda, use_parallel_executor):
        batch_size = self._get_batch_size(use_cuda, use_parallel_executor)
        in_size = [batch_size, 3, 4, 5]
        feed_in_data = np.random.uniform(
            size=[batch_size, 3, 4, 5]).astype(np.float32)
        label_size = [batch_size, 1]
        feed_label = np.random.randint(
            low=0, high=self.class_num, size=[batch_size, 1]).astype(np.float64)
        self._feed_data_in_executor(in_size, label_size, feed_in_data,
                                    feed_label, use_cuda, use_parallel_executor)

    def _test_feed_data_shape_mismatch(self, use_cuda, use_parallel_executor):
        batch_size = self._get_batch_size(use_cuda, use_parallel_executor)
        in_size = [None, 3, 4, 8]
        feed_in_data = np.random.uniform(
            size=[batch_size, 3, 4, 5]).astype(np.float32)
        label_size = [-1, 1]
        feed_label = np.random.randint(
            low=0, high=self.class_num, size=[batch_size, 1]).astype(np.int64)
        self._feed_data_in_executor(in_size, label_size, feed_in_data,
                                    feed_label, use_cuda, use_parallel_executor)

    def _test_feed_data_contains_neg_one(self, use_cuda, use_parallel_executor):
        batch_size = self._get_batch_size(use_cuda, use_parallel_executor)
        in_size = [-1, 3, 4, 5]
        feed_in_data = np.random.uniform(
            size=[batch_size, 3, 4, 5]).astype(np.float32)
        label_size = (None, 1)
        feed_label = np.random.randint(
            low=0, high=self.class_num, size=[batch_size, 1]).astype(np.int64)
        self._feed_data_in_executor(in_size, label_size, feed_in_data,
                                    feed_label, use_cuda, use_parallel_executor)

    def _test_feed_data_match_shape_type(self, use_cuda, use_parallel_executor):
        batch_size = self._get_batch_size(use_cuda, use_parallel_executor)
        in_size = [batch_size, 3, 4, 5]
        feed_in_data = np.random.uniform(size=in_size).astype(np.float32)
        label_size = [batch_size, 1]
        feed_label = np.random.randint(
            low=0, high=self.class_num, size=label_size).astype(np.int64)
        self._feed_data_in_executor(in_size, label_size, feed_in_data,
                                    feed_label, use_cuda, use_parallel_executor)

    def _feed_data_in_executor(self, in_size, label_size, feed_in_data,
                               feed_label, use_cuda, use_parallel_executor):

        startup_program = fluid.Program()
        main_program = fluid.Program()

        with fluid.program_guard(main_program, startup_program):
            in_data, label, loss = self._simple_fc_net(
                in_size, label_size, self.class_num, self.hidden_sizes)

        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

        exe = fluid.Executor(place)
        exe.run(startup_program)

        train_program = main_program
        if use_parallel_executor:
            train_program = compiler.CompiledProgram(
                main_program).with_data_parallel(loss_name=loss.name)

        for i in range(self.iterations):
            fetches = exe.run(
                train_program,
                feed={in_data.name: feed_in_data,
                      label.name: feed_label},
                fetch_list=[loss.name])


if __name__ == '__main__':
    unittest.main()
