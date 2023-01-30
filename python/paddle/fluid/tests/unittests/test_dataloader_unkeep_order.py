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

<<<<<<< HEAD
import os
import unittest

import numpy as np

import paddle
import paddle.fluid as fluid
=======
import paddle.fluid as fluid
import unittest
import numpy as np
import os
import six
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
from paddle.fluid.reader import keep_data_loader_order

keep_data_loader_order(False)


def create_reader(shape, batch_number):
<<<<<<< HEAD
    def __impl__():
        idx = 0
        for _ in range(batch_number):
=======

    def __impl__():
        idx = 0
        for _ in six.moves.range(batch_number):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            yield np.ones(shape).astype('float32') * idx,
            idx += 1

    return __impl__


class DataLoaderKeepOrderTestBase(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def initParameters(self):
        self.iterable = False
        self.break_num = 10000

    def setUp(self):
        self.epoch_num = 3
        self.batch_num = 40
        self.shape = [3, 4, 5]
        self.initParameters()

    def clear_visited(self):
        self.visited = set()

    def build_network(self, places):
        input_data = fluid.data(shape=self.shape, dtype='float32', name="input")
<<<<<<< HEAD
        loader = fluid.io.DataLoader.from_generator(
            capacity=16, feed_list=[input_data], iterable=self.iterable
        )

        fc = paddle.static.nn.fc(input_data, size=10)
        loss = paddle.mean(fc)

        loader.set_batch_generator(
            create_reader(self.shape, self.batch_num),
            places=places if loader.iterable else None,
        )

        return input_data, loss, loader

    def assertInputData(
        self, batch_id, input_data, dev_cnt, check_visited=True
    ):
=======
        loader = fluid.io.DataLoader.from_generator(capacity=16,
                                                    feed_list=[input_data],
                                                    iterable=self.iterable)

        fc = fluid.layers.fc(input_data, size=10)
        loss = fluid.layers.reduce_mean(fc)

        loader.set_batch_generator(create_reader(self.shape, self.batch_num),
                                   places=places if loader.iterable else None)

        return input_data, loss, loader

    def assertInputData(self,
                        batch_id,
                        input_data,
                        dev_cnt,
                        check_visited=True):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        if isinstance(input_data, list):
            self.assertTrue(len(input_data), dev_cnt)
            start_val = dev_cnt * batch_id
            for each_input_dict in input_data:
                input_tensor = np.array(each_input_dict["input"])
                self.assertEqual(self.shape, list(input_tensor.shape))

                num = input_tensor.flatten()[0]
                equal = (input_tensor == num).all()
                self.assertTrue(equal)
                if check_visited:
                    self.assertTrue(num not in self.visited)
                    self.visited.add(num)

                start_val += 1
        else:
<<<<<<< HEAD
            self.assertEqual(
                list(input_data.shape),
                [self.shape[0] * dev_cnt] + self.shape[1:],
            )
            start_val = dev_cnt * batch_id
            for idx in range(dev_cnt):
                data_part = input_data[
                    idx * self.shape[0] : (idx + 1) * self.shape[0], :
                ]
=======
            self.assertEqual(list(input_data.shape),
                             [self.shape[0] * dev_cnt] + self.shape[1:])
            start_val = dev_cnt * batch_id
            for idx in six.moves.range(dev_cnt):
                data_part = input_data[idx * self.shape[0]:(idx + 1) *
                                       self.shape[0], :]
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                num = data_part.flatten()[0]
                self.assertTrue((data_part == num).all())
                if check_visited:
                    self.assertTrue(num not in self.visited)
                    self.visited.add(num)

                start_val += 1

    def get_places(self):
        place_list = [fluid.cpu_places(1), fluid.cpu_places(4)]
        if fluid.is_compiled_with_cuda():
            if os.name == "nt":
                place_list.extend([fluid.cuda_places(0)])
            else:
                place_list.extend(
<<<<<<< HEAD
                    [fluid.cuda_places(0), fluid.cuda_places([0, 1])]
                )
=======
                    [fluid.cuda_places(0),
                     fluid.cuda_places([0, 1])])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        return place_list

    def test_main(self):
        for p in self.get_places():
            use_compiled_program_list = [True] if len(p) > 1 else [False, True]
            for use_compiled_program in use_compiled_program_list:
                self.run_main_with_place(p, use_compiled_program)

    def run_main_with_place(self, places, use_compiled_program=True):
        with fluid.scope_guard(fluid.Scope()):
            with fluid.program_guard(fluid.Program(), fluid.Program()):
                input_data, loss, loader = self.build_network(places)
                fetch_list = [input_data]

                exe = fluid.Executor(places[0])
                exe.run(fluid.default_startup_program())

                dev_cnt = len(places)
                if dev_cnt > 1:
                    self.assertTrue(use_compiled_program)

                main_program = fluid.default_main_program()
                if use_compiled_program:
                    main_program = fluid.CompiledProgram(
<<<<<<< HEAD
                        main_program
                    ).with_data_parallel(loss_name=loss.name, places=places)

                max_batch_num = min(
                    self.break_num, int(self.batch_num / dev_cnt)
                )

                if loader.iterable:
                    early_break = False
                    for epoch_id in range(self.epoch_num):
=======
                        main_program).with_data_parallel(loss_name=loss.name,
                                                         places=places)

                max_batch_num = min(self.break_num,
                                    int(self.batch_num / dev_cnt))

                if loader.iterable:
                    early_break = False
                    for epoch_id in six.moves.range(self.epoch_num):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                        early_break = False
                        self.clear_visited()
                        batch_id = 0
                        for data in loader():
                            if batch_id >= self.break_num:
                                early_break = True
                                break
<<<<<<< HEAD
                            self.assertInputData(
                                batch_id, data, dev_cnt, check_visited=False
                            )
                            (fetch_val,) = exe.run(
                                program=main_program,
                                feed=data,
                                fetch_list=fetch_list,
                            )
=======
                            self.assertInputData(batch_id,
                                                 data,
                                                 dev_cnt,
                                                 check_visited=False)
                            fetch_val, = exe.run(program=main_program,
                                                 feed=data,
                                                 fetch_list=fetch_list)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                            self.assertInputData(batch_id, fetch_val, dev_cnt)
                            batch_id += 1

                        if dev_cnt == 1:
                            self.assertEqual(batch_id, max_batch_num)
                        else:
                            self.assertLessEqual(batch_id, max_batch_num)

                    if early_break:
                        loader._reset()
                else:
<<<<<<< HEAD
                    for epoch_id in range(self.epoch_num):
=======
                    for epoch_id in six.moves.range(self.epoch_num):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                        batch_id = 0
                        self.clear_visited()
                        loader.start()
                        try:
                            while True:
                                if batch_id >= self.break_num:
                                    loader.reset()
                                    break
<<<<<<< HEAD
                                (fetch_val,) = exe.run(
                                    program=main_program, fetch_list=fetch_list
                                )
                                self.assertInputData(
                                    batch_id, fetch_val, dev_cnt
                                )
=======
                                fetch_val, = exe.run(program=main_program,
                                                     fetch_list=fetch_list)
                                self.assertInputData(batch_id, fetch_val,
                                                     dev_cnt)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                                batch_id += 1
                        except fluid.core.EOFException:
                            loader.reset()

                        if dev_cnt == 1:
                            self.assertEqual(batch_id, max_batch_num)
                        else:
                            self.assertLessEqual(batch_id, max_batch_num)


class IterableDataLoaderKeepOrderTest2(DataLoaderKeepOrderTestBase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def initParameters(self):
        self.iterable = True
        self.break_num = 10000


class IterableDataLoaderKeepOrderTest3(DataLoaderKeepOrderTestBase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def initParameters(self):
        self.iterable = False
        self.break_num = 2


class IterableDataLoaderKeepOrderTest4(DataLoaderKeepOrderTestBase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def initParameters(self):
        self.iterable = True
        self.break_num = 2


class IterableDataLoaderKeepOrderTest5(DataLoaderKeepOrderTestBase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def initParameters(self):
        self.iterable = False
        self.break_num = 0


class IterableDataLoaderKeepOrderTest6(DataLoaderKeepOrderTestBase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def initParameters(self):
        self.iterable = True
        self.break_num = 0


if __name__ == '__main__':
    unittest.main()
