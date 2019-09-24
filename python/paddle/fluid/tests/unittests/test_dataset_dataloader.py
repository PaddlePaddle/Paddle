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

import paddle.fluid as fluid
import numpy as np
import six
import os
import unittest
from simple_nets import simple_fc_net_with_inputs

BATCH_SIZE = 32
BATCH_NUM = 10
EPOCH_NUM = 4

IMAGE_SHAPE = [2, 3]
LABEL_SHAPE = [1]

ALL_WRITTEN_FILES = set()


def get_place_string(p):
    if isinstance(p, (fluid.CPUPlace or fluid.CUDAPlace)):
        tmp = fluid.core.Place()
        tmp.set_place(p)
        p = tmp

    if p._type() == fluid.CPUPlace()._type():
        return 'CPUPlace()'
    else:
        return 'CUDAPlace()'


def remove_all_written_files():
    for filename in ALL_WRITTEN_FILES:
        os.remove(filename)


def write_reader_data_to_file(filename, reader):
    ALL_WRITTEN_FILES.add(filename)
    with open(filename, 'w') as fid:
        for instance_list in reader():
            for i, instance in enumerate(instance_list):
                instance = np.reshape(instance, [instance.size, ])
                fid.write(str(instance.size) + ' ')
                fid.write(' '.join(map(str, instance)))
                fid.write(' ')

            fid.write('\n')


def fake_reader(batch_size=BATCH_SIZE, batch_num=BATCH_NUM):
    def __reader__():
        iteration = BATCH_SIZE * BATCH_NUM
        iteration = int(iteration + BATCH_SIZE / 2)
        for _ in six.moves.range(iteration):
            image = np.random.random(size=IMAGE_SHAPE).astype('float32')
            label = np.random.random_integers(
                size=LABEL_SHAPE, low=0, high=9).astype('int64')
            yield image, label

    return __reader__


class DatasetLoaderTestBase(unittest.TestCase):
    def setUp(self):
        self.dataset_name = "QueueDataset"
        self.drop_last = False

    def tearDown(self):
        return
        remove_all_written_files()

    def build_network(self):
        main_prog = fluid.Program()
        startup_prog = fluid.Program()
        with fluid.program_guard(main_prog, startup_prog):
            image = fluid.layers.data(
                name='image', shape=IMAGE_SHAPE, dtype='float32')
            label = fluid.layers.data(
                name='label', shape=LABEL_SHAPE, dtype='int64')

            simple_fc_net_with_inputs(image, label)

        return main_prog, startup_prog, [image, label]

    def check_batch_number(self, place, randomize_batch_num=False):
        main_prog, startup_prog, feeds = self.build_network()
        dataset = fluid.DatasetFactory().create_dataset(self.dataset_name)
        dataset.set_batch_size(BATCH_SIZE)

        if isinstance(place, fluid.CPUPlace):
            file_num = 10
            os.environ['CPU_NUM'] = str(file_num)
            places = fluid.cpu_places()
            use_cuda = False
        else:
            file_num = fluid.core.get_cuda_device_count()
            places = fluid.cuda_places()
            use_cuda = True

        filelist = []
        if file_num > 1 and randomize_batch_num:
            random_delta_batch_size = np.random.random_integers(
                low=-BATCH_NUM / 2, high=BATCH_NUM / 2, size=[file_num])
            random_delta_batch_size[-1] = -int(
                np.sum(random_delta_batch_size[0:-1]))
        else:
            random_delta_batch_size = np.zeros(shape=[file_num])

        for i in six.moves.range(file_num):
            filename = 'dataset_test_{}.txt'.format(i)
            filelist.append(filename)
            write_reader_data_to_file(
                filename,
                fake_reader(batch_num=BATCH_NUM + random_delta_batch_size[i]))

        dataset.set_filelist(filelist)
        dataset.set_use_var(feeds)
        dataset.set_pipe_command("cat")
        if self.dataset_name == 'InMemoryDataset':
            dataset.load_into_memory()

        dataloader = fluid.io.DataLoader.from_dataset(
            dataset=dataset, places=places, drop_last=self.drop_last)
        prog = fluid.CompiledProgram(main_prog).with_data_parallel()
        exe = fluid.Executor(place)

        exe.run(startup_prog)

        for _ in six.moves.range(EPOCH_NUM):
            has_complete_batch = False
            for batch_id, data in enumerate(dataloader):
                self.assertEquals(len(places), len(data))
                for idx, data_on_each_device in enumerate(data):
                    image = data_on_each_device["image"]
                    label = data_on_each_device["label"]

                    if self.drop_last:
                        batch_size = BATCH_SIZE
                    else:
                        if batch_id == BATCH_NUM:
                            batch_size = BATCH_SIZE / 2
                        else:
                            batch_size = BATCH_SIZE

                    self.assertEquals(image.shape()[1:], IMAGE_SHAPE)
                    self.assertTrue(
                        image._place()._equals(places[idx]),
                        msg=get_place_string(image._place()) + ' vs ' +
                        get_place_string(places[idx]))
                    if self.drop_last:
                        self.assertEquals(image.shape()[0], BATCH_SIZE)
                    else:
                        self.assertTrue(image.shape()[0] == BATCH_SIZE or
                                        image.shape()[0] == BATCH_SIZE / 2)

                    self.assertEquals(label.shape()[1:], LABEL_SHAPE)
                    self.assertTrue(label._place()._equals(places[idx]))
                    if self.drop_last:
                        self.assertEquals(label.shape()[0], BATCH_SIZE)
                    else:
                        self.assertTrue(label.shape()[0] == BATCH_SIZE or
                                        label.shape()[0] == BATCH_SIZE / 2)

                    self.assertEquals(image.shape()[0], label.shape()[0])

                    if image.shape()[0] == BATCH_SIZE:
                        has_complete_batch = True

                exe.run(prog, feed=data)

            self.assertTrue(has_complete_batch)

    def get_all_places(self):
        p = [fluid.CPUPlace()]
        if fluid.is_compiled_with_cuda():
            p.append(fluid.CUDAPlace(0))
        return p

    def test_batch_number_with_same_length_files(self):
        for p in self.get_all_places():
            with fluid.scope_guard(fluid.Scope()):
                self.check_batch_number(place=p, randomize_batch_num=False)

    def test_batch_number_with_different_length_files(self):
        for p in self.get_all_places():
            with fluid.scope_guard(fluid.Scope()):
                self.check_batch_number(place=p, randomize_batch_num=True)


class QueueDatasetTestWithoutDropLast(DatasetLoaderTestBase):
    def setUp(self):
        self.dataset_name = "QueueDataset"
        self.drop_last = True


class InMemoryDatasetTestWithoutDropLast(DatasetLoaderTestBase):
    def setUp(self):
        self.dataset_name = "InMemoryDataset"
        self.drop_last = False


class InMemoryDatasetTestWithDropLast(DatasetLoaderTestBase):
    def setUp(self):
        self.dataset_name = "InMemoryDataset"
        self.drop_last = True


if __name__ == '__main__':
    unittest.main()
