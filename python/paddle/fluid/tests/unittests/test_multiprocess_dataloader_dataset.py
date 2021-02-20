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

from __future__ import division

import unittest
import numpy as np

import paddle
import paddle.fluid as fluid
from paddle.io import Dataset, IterableDataset, TensorDataset, \
        ComposeDataset, ChainDataset, DataLoader
from paddle.fluid.dygraph.base import to_variable

IMAGE_SIZE = 32


class RandomDataset(Dataset):
    def __init__(self, sample_num):
        self.sample_num = sample_num

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        np.random.seed(idx)
        image = np.random.random([IMAGE_SIZE]).astype('float32')
        label = np.random.randint(0, 9, (1, )).astype('int64')
        return image, label


class RandomIterableDataset(IterableDataset):
    def __init__(self, sample_num):
        self.sample_num = sample_num

    def __iter__(self):
        for i in range(self.sample_num):
            np.random.seed(i)
            image = np.random.random([IMAGE_SIZE]).astype('float32')
            label = np.random.randint(0, 9, (1, )).astype('int64')
            yield image, label


class TestTensorDataset(unittest.TestCase):
    def run_main(self, num_workers, places):
        fluid.default_startup_program().random_seed = 1
        fluid.default_main_program().random_seed = 1
        place = fluid.CPUPlace()
        with fluid.dygraph.guard(place):
            input_np = np.random.random([16, 3, 4]).astype('float32')
            input = to_variable(input_np)
            label_np = np.random.random([16, 1]).astype('int32')
            label = to_variable(label_np)

            dataset = TensorDataset([input, label])
            assert len(dataset) == 16
            dataloader = DataLoader(
                dataset,
                places=place,
                num_workers=num_workers,
                batch_size=1,
                drop_last=True)

            for i, (input, label) in enumerate(dataloader()):
                assert len(input) == 1
                assert len(label) == 1
                assert input.shape == [1, 3, 4]
                assert label.shape == [1, 1]
                assert isinstance(input, paddle.Tensor)
                assert isinstance(label, paddle.Tensor)
                assert np.allclose(input.numpy(), input_np[i])
                assert np.allclose(label.numpy(), label_np[i])

    def test_main(self):
        places = [fluid.CPUPlace()]
        if fluid.core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.run_main(num_workers=0, places=p)


class TestComposeDataset(unittest.TestCase):
    def test_main(self):
        fluid.default_startup_program().random_seed = 1
        fluid.default_main_program().random_seed = 1

        dataset1 = RandomDataset(10)
        dataset2 = RandomDataset(10)
        dataset = ComposeDataset([dataset1, dataset2])
        assert len(dataset) == 10

        for i in range(len(dataset)):
            input1, label1, input2, label2 = dataset[i]
            input1_t, label1_t = dataset1[i]
            input2_t, label2_t = dataset2[i]
            assert np.allclose(input1, input1_t)
            assert np.allclose(label1, label1_t)
            assert np.allclose(input2, input2_t)
            assert np.allclose(label2, label2_t)


class TestChainDataset(unittest.TestCase):
    def run_main(self, num_workers, places):
        fluid.default_startup_program().random_seed = 1
        fluid.default_main_program().random_seed = 1

        dataset1 = RandomIterableDataset(10)
        dataset2 = RandomIterableDataset(10)
        dataset = ChainDataset([dataset1, dataset2])

        samples = []
        for data in iter(dataset):
            samples.append(data)
        assert len(samples) == 20

        idx = 0
        for image, label in iter(dataset1):
            assert np.allclose(image, samples[idx][0])
            assert np.allclose(label, samples[idx][1])
            idx += 1
        for image, label in iter(dataset2):
            assert np.allclose(image, samples[idx][0])
            assert np.allclose(label, samples[idx][1])
            idx += 1

    def test_main(self):
        places = [fluid.CPUPlace()]
        if fluid.core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.run_main(num_workers=0, places=p)


class NumpyMixTensorDataset(Dataset):
    def __init__(self, sample_num):
        self.sample_num = sample_num

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        np.random.seed(idx)
        image = np.random.random([IMAGE_SIZE]).astype('float32')
        label = np.random.randint(0, 9, (1, )).astype('int64')
        return paddle.to_tensor(image, place=paddle.CPUPlace()), label


class TestNumpyMixTensorDataset(TestTensorDataset):
    def run_main(self, num_workers, places):
        paddle.static.default_startup_program().random_seed = 1
        paddle.static.default_main_program().random_seed = 1
        place = paddle.CPUPlace()
        with fluid.dygraph.guard(place):
            dataset = NumpyMixTensorDataset(16)
            assert len(dataset) == 16
            dataloader = DataLoader(
                dataset,
                places=place,
                num_workers=num_workers,
                batch_size=1,
                drop_last=True)

            for i, (input, label) in enumerate(dataloader()):
                assert len(input) == 1
                assert len(label) == 1
                assert input.shape == [1, IMAGE_SIZE]
                assert label.shape == [1, 1]
                assert isinstance(input, paddle.Tensor)
                assert isinstance(label, paddle.Tensor)


if __name__ == '__main__':
    unittest.main()
