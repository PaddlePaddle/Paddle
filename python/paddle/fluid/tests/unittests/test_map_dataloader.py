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

from __future__ import division

import os
import six
import time
import unittest
import numpy as np

import paddle.fluid as fluid
from paddle.fluid.io import Dataset, MnistDataset, BatchSampler, DataLoader
from paddle.fluid.dygraph.nn import Linear
from paddle.fluid.dygraph.base import to_variable

EPOCH_NUM = 10
BATCH_SIZE = 32
IMAGE_SIZE = 784
SAMPLE_NUM = 800
CLASS_NUM = 10


class RandomDataset(Dataset):
    def __init__(self, sample_num, class_num):
        self.sample_num = sample_num
        self.class_num = class_num

    def __getitem__(self, idx):
        np.random.seed(idx)
        image = np.random.random([IMAGE_SIZE]).astype('float32')
        label = np.random.randint(0, CLASS_NUM - 1, (1, )).astype('int64')
        return image, label

    def __len__(self):
        return self.sample_num


def simple_fc_net_static():
    startup_prog = fluid.Program()
    main_prog = fluid.Program()
    startup_prog.random_seed = 1
    main_prog.random_seed = 1

    with fluid.unique_name.guard():
        with fluid.program_guard(main_prog, startup_prog):
            image = fluid.data(
                name='image', shape=[None, IMAGE_SIZE], dtype='float32')
            label = fluid.data(name='label', shape=[None, 1], dtype='int64')
            hidden = image
            param_attr = fluid.ParamAttr(initializer=fluid.initializer.Constant(
                value=0.8))
            bias_attr = fluid.ParamAttr(initializer=fluid.initializer.Constant(
                value=0.5))
            for hidden_size in [10, 20, 30]:
                hidden = fluid.layers.fc(hidden,
                                         size=hidden_size,
                                         act='tanh',
                                         param_attr=param_attr,
                                         bias_attr=bias_attr)

            predict_label = fluid.layers.fc(hidden,
                                            size=CLASS_NUM,
                                            act='softmax',
                                            param_attr=param_attr,
                                            bias_attr=bias_attr)
            loss = fluid.layers.reduce_mean(
                fluid.layers.cross_entropy(
                    input=predict_label, label=label))

            optimizer = fluid.optimizer.Adam()
            optimizer.minimize(loss)
    return startup_prog, main_prog, image, label, loss


class SimpleFCNet(fluid.dygraph.Layer):
    def __init__(self):
        super(SimpleFCNet, self).__init__()

        param_attr = fluid.ParamAttr(initializer=fluid.initializer.Constant(
            value=0.8))
        bias_attr = fluid.ParamAttr(initializer=fluid.initializer.Constant(
            value=0.5))
        self._fcs = []
        in_channel = IMAGE_SIZE
        for hidden_size in [10, 20, 30]:
            self._fcs.append(
                Linear(
                    in_channel,
                    hidden_size,
                    act='tanh',
                    param_attr=param_attr,
                    bias_attr=bias_attr))
            in_channel = hidden_size
        self._fcs.append(
            Linear(
                in_channel,
                CLASS_NUM,
                act='softmax',
                param_attr=param_attr,
                bias_attr=bias_attr))

    def forward(self, image):
        out = image
        for fc in self._fcs:
            out = fc(out)
        return out


class TestStaticDataLoader(unittest.TestCase):
    def run_main(self, num_workers, use_buffer_reader, places,
                 with_data_parallel):
        scope = fluid.Scope()
        with fluid.scope_guard(scope):
            startup_prog, main_prog, image, label, loss = simple_fc_net_static()

            ps = places if use_buffer_reader else fluid.cpu_places(len(places))
            dataset = RandomDataset(SAMPLE_NUM, CLASS_NUM)
            dataloader = DataLoader(
                dataset,
                feed_list=[image, label],
                places=ps,
                num_workers=num_workers,
                batch_size=BATCH_SIZE,
                shuffle=use_buffer_reader,
                drop_last=True)

            exe = fluid.Executor(place=places[0])
            exe.run(startup_prog)

            prog = fluid.CompiledProgram(main_prog)
            if with_data_parallel:
                prog = prog.with_data_parallel(
                    loss_name=loss.name, places=places)

            step_list = []
            loss_list = []
            start_t = time.time()
            for _ in six.moves.range(EPOCH_NUM):
                step = 0
                for d in dataloader:
                    assert len(d) == len(places), "{} != {}".format(
                        len(d), len(places))
                    for i, item in enumerate(d):
                        image = item['image']
                        label = item['label']
                        assert image.shape() == [BATCH_SIZE, IMAGE_SIZE]
                        assert label.shape() == [BATCH_SIZE, 1]
                        assert image._place()._equals(ps[i])
                        assert label._place()._equals(ps[i])
                    L, = exe.run(program=prog,
                                 feed=d,
                                 fetch_list=[loss],
                                 use_program_cache=True)
                    loss_list.append(np.mean(L))
                    step += 1
                step_list.append(step)

        end_t = time.time()
        ret = {
            "time": end_t - start_t,
            "step": step_list,
            "loss": np.array(loss_list)
        }
        print("time cost", ret['time'], 'step_list', ret['step'])
        return ret

    def prepare_places(self, with_data_parallel, with_cpu=True, with_gpu=True):
        places = []
        if with_cpu:
            places.append([fluid.CPUPlace()])
            if with_data_parallel:
                places.append([fluid.CPUPlace()] * 2)

        if with_gpu and fluid.core.is_compiled_with_cuda():
            tmp = fluid.cuda_places()
            assert len(tmp) > 0, "no gpu detected"
            if with_data_parallel:
                places.append(tmp)
            places.append([tmp[0]])
        return places

    def test_main(self):
        for with_data_parallel in [False] if self.__class__.__name__ \
                == "TestDygraphDataLoader" else [True, False]:
            for p in self.prepare_places(with_data_parallel):
                for use_buffer_reader in [False, True]:
                    results = []
                    for num_workers in [0, 4]:
                        print(self.__class__.__name__, p, use_buffer_reader,
                              num_workers)
                        ret = self.run_main(
                            num_workers=num_workers,
                            use_buffer_reader=use_buffer_reader,
                            places=p,
                            with_data_parallel=with_data_parallel)
                        results.append(ret)
                    if not use_buffer_reader:
                        diff = np.max(
                            np.abs(results[0]['loss'] - results[1]['loss']) /
                            np.abs(results[0]['loss']))
                        self.assertLess(diff, 1e-2)


class TestDygraphDataLoader(TestStaticDataLoader):
    def run_main(self, num_workers, use_buffer_reader, places,
                 with_data_parallel):
        fluid.default_startup_program().random_seed = 1
        fluid.default_main_program().random_seed = 1
        with fluid.dygraph.guard(places[0]):
            fc_net = SimpleFCNet()
            optimizer = fluid.optimizer.Adam(parameter_list=fc_net.parameters())

            dataset = RandomDataset(SAMPLE_NUM, CLASS_NUM)
            dataloader = DataLoader(
                dataset,
                places=places[0],
                num_workers=num_workers,
                batch_size=BATCH_SIZE,
                shuffle=use_buffer_reader,
                drop_last=True)

            step_list = []
            loss_list = []
            start_t = time.time()
            for _ in six.moves.range(EPOCH_NUM):
                step = 0
                for image, label in dataloader():
                    out = fc_net(image)
                    loss = fluid.layers.cross_entropy(out, label)
                    avg_loss = fluid.layers.reduce_mean(loss)
                    avg_loss.backward()
                    optimizer.minimize(avg_loss)
                    fc_net.clear_gradients()

                    loss_list.append(np.mean(avg_loss.numpy()))
                    step += 1
                step_list.append(step)

        end_t = time.time()
        ret = {
            "time": end_t - start_t,
            "step": step_list,
            "loss": np.array(loss_list)
        }
        print("time cost", ret['time'], 'step_list', ret['step'])
        return ret


class TestDataLoaderSetXXXException(unittest.TestCase):
    def test_main(self):
        place = fluid.cpu_places()[0]
        with fluid.dygraph.guard(place):
            dataset = RandomDataset(SAMPLE_NUM, CLASS_NUM)
            dataloader = DataLoader(dataset, places=place)

            try:
                dataloader.set_sample_generator()
                self.assertTrue(False)
            except:
                pass

            try:
                dataloader.set_sample_list_generator()
                self.assertTrue(False)
            except:
                pass

            try:
                dataloader.set_batch_generator()
                self.assertTrue(False)
            except:
                pass


# -------------- Dataset unittests --------------
class TestDatasetAbstract(unittest.TestCase):
    def test_main(self):
        dataset = Dataset()
        try:
            d = dataset[0]
            self.assertTrue(False)
        except NotImplementedError:
            pass

        try:
            l = len(dataset)
            self.assertTrue(False)
        except NotImplementedError:
            pass


class TestMnistDataset(unittest.TestCase):
    def test_main(self):
        md = MnistDataset(mode='test')
        self.assertTrue(len(md) == 10000)

        for i in range(len(md)):
            image, label = md[i]
            self.assertTrue(image.shape[0] == 784)
            self.assertTrue(isinstance(label, int))


# -------------- BatchSampler unittests --------------
class TestBatchSampler(unittest.TestCase):
    def setUp(self):
        self.num_samples = 1000
        self.batch_size = 32
        self.shuffle = False
        self.drop_last = False

    def test_main(self):
        bs = BatchSampler(
            [0] * self.num_samples,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=self.drop_last)

        # length check
        bs_len = (self.num_samples + int(not self.drop_last) \
                * (self.batch_size - 1)) // self.batch_size
        self.assertTrue(bs_len == len(bs))

        # output indices check
        if not self.shuffle:
            index = 0
            for indices in bs:
                for idx in indices:
                    self.assertTrue(index == idx)
                    index += 1


class TestBatchSamplerDropLast(TestBatchSampler):
    def setUp(self):
        self.num_samples = 1000
        self.batch_size = 32
        self.shuffle = False
        self.drop_last = True


class TestBatchSamplerShuffle(TestBatchSampler):
    def setUp(self):
        self.num_samples = 1000
        self.batch_size = 32
        self.shuffle = True
        self.drop_last = True


if __name__ == '__main__':
    unittest.main()
