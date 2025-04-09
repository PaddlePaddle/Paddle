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

import multiprocessing
import os
import unittest

import numpy as np

import paddle
from paddle import base
from paddle.base import core

paddle.enable_static()

os.environ['CPU_NUM'] = str(4)
np.random.seed(123)


class TestFeedData(unittest.TestCase):
    '''
    Test paddle.static.data feeds with different shape and types.
    Note: paddle.static.data is not paddle.static.data.
    '''

    def setUp(self):
        self.hidden_sizes = [25, 20, 15]
        self.data_batch_size = 10
        self.class_num = 10
        self.iterations = 5

    def _get_device_count(self, use_cuda):
        return (
            core.get_cuda_device_count()
            if use_cuda
            else int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
        )

    def _get_feed_batch_size(self, use_cuda):
        """
        Returns actual fed data size. We should multiple the number of
        devices when it is using ParallelExecutor
        """
        return self.data_batch_size

    def _simple_fc_net(self, in_size, label_size, class_num, hidden_sizes):
        in_data = paddle.static.data(
            name="data", dtype='float32', shape=in_size
        )
        label = paddle.static.data(
            name='label', dtype='int64', shape=label_size
        )

        hidden = in_data
        for hidden_size in hidden_sizes:
            hidden = paddle.static.nn.fc(hidden, size=hidden_size)

        predict_label = paddle.static.nn.fc(
            hidden, size=class_num, activation='softmax'
        )
        loss = paddle.mean(
            paddle.nn.functional.cross_entropy(
                input=predict_label,
                label=label,
                reduction='none',
                use_softmax=False,
            )
        )

        optimizer = paddle.optimizer.Adam()
        optimizer.minimize(loss)
        return in_data, label, loss

    def test(self):
        for use_cuda in (
            [True, False] if core.is_compiled_with_cuda() else [False]
        ):
            print('Test Parameters:'),
            print(
                {
                    'use_cuda': use_cuda,
                }
            )
            # Test feeding without error
            self._test_feed_data_match_shape_type(use_cuda)
            self._test_feed_data_contains_neg_one(use_cuda)
            self._test_feed_lod_tensor(use_cuda)

            # Test exception message when feeding with error
            in_shape_tuple = (-1, 3, 4, 8)
            error_shape_list = [self.data_batch_size, 3, 4, 5]

            with self.assertRaises(ValueError) as shape_mismatch_err:
                self._test_feed_data_shape_mismatch(use_cuda)
            self.assertEqual(
                str(shape_mismatch_err.exception),
                "The fed Variable {!r} should have dimensions = {!r}, "
                "shape = {!r}, but received fed shape {!r} on each device".format(
                    'data',
                    len(in_shape_tuple),
                    in_shape_tuple,
                    error_shape_list,
                ),
            )

            with self.assertRaises(ValueError) as dtype_mismatch_err:
                self._test_feed_data_dtype_mismatch(use_cuda)
            self.assertEqual(
                str(dtype_mismatch_err.exception),
                "The data type of fed Variable {!r} must be 'int64', but "
                "received 'float64'".format('label'),
            )

    def _test_feed_data_dtype_mismatch(self, use_cuda):
        feed_batch_size = self._get_feed_batch_size(use_cuda)
        in_size = [self.data_batch_size, 3, 4, 5]
        feed_in_data = np.random.uniform(
            size=[feed_batch_size, 3, 4, 5]
        ).astype(np.float32)
        label_size = [self.data_batch_size, 1]
        feed_label = np.random.randint(
            low=0, high=self.class_num, size=[feed_batch_size, 1]
        ).astype(np.float64)
        self._feed_data_in_executor(
            in_size,
            label_size,
            feed_in_data,
            feed_label,
            use_cuda,
        )

    def _test_feed_data_shape_mismatch(self, use_cuda):
        batch_size = self._get_feed_batch_size(use_cuda)
        in_size = [None, 3, 4, 8]
        feed_in_data = np.random.uniform(size=[batch_size, 3, 4, 5]).astype(
            np.float32
        )
        label_size = [-1, 1]
        feed_label = np.random.randint(
            low=0, high=self.class_num, size=[batch_size, 1]
        ).astype(np.int64)
        self._feed_data_in_executor(
            in_size,
            label_size,
            feed_in_data,
            feed_label,
            use_cuda,
        )

    def _test_feed_data_contains_neg_one(self, use_cuda):
        batch_size = self._get_feed_batch_size(use_cuda)
        in_size = [-1, 3, 4, 5]
        feed_in_data = np.random.uniform(size=[batch_size, 3, 4, 5]).astype(
            np.float32
        )
        label_size = (None, 1)
        feed_label = np.random.randint(
            low=0, high=self.class_num, size=[batch_size, 1]
        ).astype(np.int64)
        self._feed_data_in_executor(
            in_size,
            label_size,
            feed_in_data,
            feed_label,
            use_cuda,
        )

    def _test_feed_data_match_shape_type(self, use_cuda):
        feed_batch_size = self._get_feed_batch_size(use_cuda)
        in_size = [self.data_batch_size, 3, 4, 5]
        feed_in_data = np.random.uniform(
            size=[feed_batch_size, 3, 4, 5]
        ).astype(np.float32)
        label_size = [self.data_batch_size, 1]
        feed_label = np.random.randint(
            low=0, high=self.class_num, size=[feed_batch_size, 1]
        ).astype(np.int64)
        self._feed_data_in_executor(
            in_size,
            label_size,
            feed_in_data,
            feed_label,
            use_cuda,
        )

    def _test_feed_lod_tensor(self, use_cuda):
        device_count = self._get_device_count(use_cuda)

        in_size = [device_count, 3, 4, 5]
        sequence_lengths = [range(1, device_count + 1)]
        # sum from 1 to device_count
        sum_length = int((device_count + 1) * device_count / 2)

        feed_in_data = np.random.uniform(size=[sum_length, 3, 4, 5]).astype(
            np.float32
        )
        feed_data_tensor = base.DenseTensor()
        feed_data_tensor.set(feed_in_data, base.CPUPlace())
        feed_data_tensor.set_recursive_sequence_lengths(sequence_lengths)

        label_size = [device_count, 1]
        feed_label_tensor = base.DenseTensor()
        feed_label = np.random.randint(
            low=0, high=self.class_num, size=[sum_length, 1]
        ).astype(np.int64)
        feed_label_tensor.set(feed_label, base.CPUPlace())
        feed_label_tensor.set_recursive_sequence_lengths(sequence_lengths)

        self._feed_data_in_executor(
            in_size,
            label_size,
            feed_data_tensor,
            feed_label_tensor,
            use_cuda,
        )

    def _feed_data_in_executor(
        self,
        in_size,
        label_size,
        feed_in_data,
        feed_label,
        use_cuda,
    ):
        startup_program = base.Program()
        main_program = base.Program()

        with base.program_guard(main_program, startup_program):
            in_data, label, loss = self._simple_fc_net(
                in_size, label_size, self.class_num, self.hidden_sizes
            )

        place = base.CUDAPlace(0) if use_cuda else base.CPUPlace()

        exe = base.Executor(place)
        exe.run(startup_program)

        train_program = main_program

        for i in range(self.iterations):
            fetches = exe.run(
                train_program,
                feed={in_data.name: feed_in_data, label.name: feed_label},
                fetch_list=[loss],
            )


if __name__ == '__main__':
    unittest.main()
