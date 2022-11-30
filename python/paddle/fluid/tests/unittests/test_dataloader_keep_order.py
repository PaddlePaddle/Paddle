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

import os
import unittest

import numpy as np

import paddle
import paddle.fluid as fluid


def create_reader(shape, batch_number):
    def __impl__():
        idx = 0
        for _ in range(batch_number):
            yield np.ones(shape).astype('float32') * idx,
            idx += 1

    return __impl__


class DataLoaderKeepOrderTestBase(unittest.TestCase):
    def initParameters(self):
        self.iterable = False
        self.break_num = 100

    def setUp(self):
        self.epoch_num = 3
        self.batch_num = 40
        self.shape = [3, 4, 5]
        self.initParameters()

    def build_network(self, places):
        input_data = fluid.data(shape=self.shape, dtype='float32', name="input")
        loader = fluid.io.DataLoader.from_generator(
            capacity=16, feed_list=[input_data], iterable=self.iterable
        )

        fc = fluid.layers.fc(input_data, size=10)
        loss = paddle.mean(fc)

        loader.set_batch_generator(
            create_reader(self.shape, self.batch_num),
            places=places if loader.iterable else None,
        )

        return input_data, loss, loader

    def assertInputData(self, batch_id, input_data, dev_cnt):
        if isinstance(input_data, list):
            self.assertTrue(len(input_data), dev_cnt)
            start_val = dev_cnt * batch_id
            for each_input_dict in input_data:
                input_tensor = np.array(each_input_dict["input"])
                self.assertEqual(self.shape, list(input_tensor.shape))
                self.assertTrue((input_tensor == start_val).all())
                start_val += 1
        else:
            self.assertEqual(
                list(input_data.shape),
                [self.shape[0] * dev_cnt] + self.shape[1:],
            )
            start_val = dev_cnt * batch_id
            for idx in range(dev_cnt):
                data_part = input_data[
                    idx * self.shape[0] : (idx + 1) * self.shape[0], :
                ]
                self.assertTrue((data_part == start_val).all())
                start_val += 1

    def get_places(self):
        place_list = [fluid.cpu_places(1), fluid.cpu_places(4)]
        if fluid.is_compiled_with_cuda():
            if os.name == "nt":
                place_list.extend([fluid.cuda_places(0)])
            else:
                place_list.extend(
                    [fluid.cuda_places(0), fluid.cuda_places([0, 1])]
                )
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
                        main_program
                    ).with_data_parallel(loss_name=loss.name, places=places)

                max_batch_num = min(
                    self.break_num, int(self.batch_num / dev_cnt)
                )

                if loader.iterable:
                    early_break = False
                    for epoch_id in range(self.epoch_num):
                        early_break = False
                        batch_id = 0
                        for data in loader():
                            if batch_id >= self.break_num:
                                early_break = True
                                break
                            self.assertInputData(batch_id, data, dev_cnt)
                            (fetch_val,) = exe.run(
                                program=main_program,
                                feed=data,
                                fetch_list=fetch_list,
                            )
                            self.assertInputData(batch_id, fetch_val, dev_cnt)
                            batch_id += 1

                        self.assertEqual(batch_id, max_batch_num)

                    if early_break:
                        loader._reset()
                else:
                    for epoch_id in range(self.epoch_num):
                        batch_id = 0
                        loader.start()
                        try:
                            while True:
                                if batch_id >= self.break_num:
                                    loader.reset()
                                    break
                                (fetch_val,) = exe.run(
                                    program=main_program, fetch_list=fetch_list
                                )
                                self.assertInputData(
                                    batch_id, fetch_val, dev_cnt
                                )
                                batch_id += 1
                        except fluid.core.EOFException:
                            loader.reset()

                        self.assertEqual(batch_id, max_batch_num)


class IterableDataLoaderKeepOrderTest2(DataLoaderKeepOrderTestBase):
    def initParameters(self):
        self.iterable = True
        self.break_num = 100


class IterableDataLoaderKeepOrderTest3(DataLoaderKeepOrderTestBase):
    def initParameters(self):
        self.iterable = False
        self.break_num = 2


class IterableDataLoaderKeepOrderTest4(DataLoaderKeepOrderTestBase):
    def initParameters(self):
        self.iterable = True
        self.break_num = 2


class IterableDataLoaderKeepOrderTest5(DataLoaderKeepOrderTestBase):
    def initParameters(self):
        self.iterable = False
        self.break_num = 0


class IterableDataLoaderKeepOrderTest6(DataLoaderKeepOrderTestBase):
    def initParameters(self):
        self.iterable = True
        self.break_num = 0


if __name__ == '__main__':
    unittest.main()
