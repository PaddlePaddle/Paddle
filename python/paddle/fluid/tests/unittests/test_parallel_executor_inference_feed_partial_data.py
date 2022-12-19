# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid
import paddle.nn.functional as F


class TestInferencePartialFeed(unittest.TestCase):
    def setUp(self):
        self.iterations = 10
        self.size = 10

    def run_network(self, places, use_split, has_persistable):
        startup_prog = fluid.Program()
        main_prog = fluid.Program()

        with fluid.program_guard(main_prog, startup_prog):
            x = fluid.data(name='x', shape=[None, self.size], dtype='float32')
            y = fluid.data(name='y', shape=[None, self.size], dtype='float32')
            if has_persistable:
                lr = fluid.data(name='lr', shape=[1], dtype='float32')
                lr.persistable = True
            else:
                lr = fluid.data(name='lr', shape=[None], dtype='float32')

            relu_x = F.relu(x)
            relu_y = F.relu(y)
            relu_lr = F.relu(lr)

        exe = fluid.Executor(places[0])
        exe.run(startup_prog)

        prog = fluid.CompiledProgram(main_prog).with_data_parallel(
            places=places
        )

        gen_random = lambda shape: np.random.uniform(
            low=-1.0, high=1.0, size=shape
        ).astype('float32')
        assert_result = lambda feed, result: np.testing.assert_array_equal(
            np.maximum(0, feed), result
        )

        def assert_merged_unmerged(merged, unmerged):
            unmerged = np.concatenate(unmerged, axis=0)
            np.testing.assert_array_equal(merged, unmerged)

        def feed_split_test():
            for place_num in range(1, len(places) * 3):
                x_np = gen_random([place_num, self.size])
                y_np = gen_random([place_num, self.size])
                if not lr.persistable or place_num <= len(places):
                    lr_np = gen_random([place_num])
                else:
                    lr_np = gen_random([1])

                feed = {x.name: x_np, y.name: y_np, lr.name: lr_np}
                fetch_list = [relu_x, relu_y, relu_lr]

                relu_x_np, relu_y_np, relu_lr_np = exe.run(
                    prog, feed=feed, fetch_list=fetch_list, return_merged=True
                )

                (
                    relu_x_np_unmerged,
                    relu_y_np_unmerged,
                    relu_lr_np_unmerged,
                ) = exe.run(
                    prog, feed=feed, fetch_list=fetch_list, return_merged=False
                )

                assert_merged_unmerged(relu_x_np, relu_x_np_unmerged)
                assert_merged_unmerged(relu_y_np, relu_y_np_unmerged)
                assert_merged_unmerged(relu_lr_np, relu_lr_np_unmerged)

                assert_result(x_np, relu_x_np)
                assert_result(y_np, relu_y_np)
                if not lr.persistable or place_num <= len(places):
                    assert_result(lr_np, relu_lr_np)
                else:
                    expected_relu_lr_np = max(lr_np[0], 0)
                    self.assertTrue(np.all(expected_relu_lr_np == relu_lr_np))

        def feed_list_test():
            for place_num in range(1, len(places) + 1):
                x_np_list = []
                y_np_list = []
                lr_np_list = []
                feed_list = []
                for _ in range(place_num):
                    x_np = gen_random([1, self.size])
                    y_np = gen_random([1, self.size])
                    lr_np = gen_random([1])
                    x_np_list.append(x_np)
                    y_np_list.append(y_np)
                    lr_np_list.append(lr_np)

                    feed_list.append(
                        {x.name: x_np, y.name: y_np, lr.name: lr_np}
                    )

                fetch_list = [relu_x, relu_y, relu_lr]
                relu_x_np, relu_y_np, relu_lr_np = exe.run(
                    prog,
                    feed=feed_list,
                    fetch_list=fetch_list,
                    return_merged=True,
                )

                (
                    relu_x_np_unmerged,
                    relu_y_np_unmerged,
                    relu_lr_np_unmerged,
                ) = exe.run(
                    prog,
                    feed=feed_list,
                    fetch_list=fetch_list,
                    return_merged=False,
                )

                assert_merged_unmerged(relu_x_np, relu_x_np_unmerged)
                assert_merged_unmerged(relu_y_np, relu_y_np_unmerged)
                assert_merged_unmerged(relu_lr_np, relu_lr_np_unmerged)

                x_np = np.concatenate(x_np_list)
                y_np = np.concatenate(y_np_list)
                lr_np = np.concatenate(lr_np_list)

                assert_result(x_np, relu_x_np)
                assert_result(y_np, relu_y_np)
                assert_result(lr_np, relu_lr_np)

        for _ in range(self.iterations):
            if use_split:
                feed_split_test()
            else:
                feed_list_test()

    def test_main(self):
        places = [fluid.cpu_places(4)]
        if fluid.is_compiled_with_cuda():
            places.append(fluid.cuda_places())

        for p in places:
            for has_persistable in [False, True]:
                for use_split in [False, True]:
                    self.run_network(
                        p, use_split=use_split, has_persistable=has_persistable
                    )


class TestInferencePartialFeedUsingDataLoader(unittest.TestCase):
    def setUp(self):
        self.epoch_num = 3
        self.batch_num = 101  # a prime number
        self.batch_size = 32

    def create_reader(self):
        def __impl__():
            for _ in range(self.batch_num):
                yield np.random.random([self.batch_size, 1]).astype('float32'),

        return __impl__

    def run_network(self, iterable, use_cuda, drop_last):
        x = fluid.data(shape=[None, 1], name='x', dtype='float32')
        places = fluid.cuda_places() if use_cuda else fluid.cpu_places(4)
        loader = fluid.io.DataLoader.from_generator(
            feed_list=[x], capacity=16, iterable=iterable, drop_last=drop_last
        )
        y = fluid.layers.fc(x, size=10)
        loss = paddle.mean(y)

        exe = fluid.Executor(places[0])
        exe.run(fluid.default_startup_program())

        prog = fluid.CompiledProgram(
            fluid.default_main_program()
        ).with_data_parallel(places=places, loss_name=loss.name)

        loader.set_batch_generator(
            self.create_reader(), places=places if iterable else None
        )

        for _ in range(self.epoch_num):
            actual_batch_num = 0
            if loader.iterable:
                for feed_data in loader():
                    (x_data,) = exe.run(prog, feed=feed_data, fetch_list=[x])
                    self.assertEqual(x_data.shape[0] % self.batch_size, 0)
                    self.assertTrue(x_data.shape[0] != 0)
                    actual_batch_num += int(x_data.shape[0] / self.batch_size)
            else:
                loader.start()
                try:
                    while True:
                        (x_data,) = exe.run(prog, fetch_list=[x])
                        self.assertEqual(x_data.shape[0] % self.batch_size, 0)
                        self.assertTrue(x_data.shape[0] != 0)
                        actual_batch_num += int(
                            x_data.shape[0] / self.batch_size
                        )
                except fluid.core.EOFException:
                    loader.reset()

            if not drop_last or len(places) == 1:
                self.assertEqual(self.batch_num, actual_batch_num)
            else:
                self.assertGreater(self.batch_num, actual_batch_num)

    def test_main(self):
        use_cuda_list = (
            [False, True] if fluid.is_compiled_with_cuda() else [False]
        )
        iterable_list = [False, True]
        drop_last_list = [False, True]
        for iterable in iterable_list:
            for use_cuda in use_cuda_list:
                for drop_last in drop_last_list:
                    with fluid.program_guard(fluid.Program(), fluid.Program()):
                        with fluid.scope_guard(fluid.Scope()):
                            self.run_network(iterable, use_cuda, drop_last)


if __name__ == '__main__':
    unittest.main()
