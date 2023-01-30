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
import unittest

=======
from __future__ import division

import unittest
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
import numpy as np

import paddle
import paddle.fluid as fluid
<<<<<<< HEAD
from paddle.io import (
    ChainDataset,
    ComposeDataset,
    DataLoader,
    Dataset,
    IterableDataset,
    TensorDataset,
)
=======
from paddle.io import Dataset, IterableDataset, TensorDataset, \
        ComposeDataset, ChainDataset, DataLoader, random_split, Subset
from paddle.fluid.framework import _test_eager_guard, _in_legacy_dygraph
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

IMAGE_SIZE = 32


class RandomDataset(Dataset):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def __init__(self, sample_num):
        self.sample_num = sample_num

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        np.random.seed(idx)
        image = np.random.random([IMAGE_SIZE]).astype('float32')
<<<<<<< HEAD
        label = np.random.randint(0, 9, (1,)).astype('int64')
=======
        label = np.random.randint(0, 9, (1, )).astype('int64')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        return image, label


class RandomIterableDataset(IterableDataset):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def __init__(self, sample_num):
        self.sample_num = sample_num

    def __iter__(self):
        for i in range(self.sample_num):
            np.random.seed(i)
            image = np.random.random([IMAGE_SIZE]).astype('float32')
<<<<<<< HEAD
            label = np.random.randint(0, 9, (1,)).astype('int64')
=======
            label = np.random.randint(0, 9, (1, )).astype('int64')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            yield image, label


class TestTensorDataset(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def run_main(self, num_workers, places):
        paddle.static.default_startup_program().random_seed = 1
        paddle.static.default_main_program().random_seed = 1
        place = paddle.CPUPlace()
        with fluid.dygraph.guard(place):
            input_np = np.random.random([16, 3, 4]).astype('float32')
            input = paddle.to_tensor(input_np)
            label_np = np.random.random([16, 1]).astype('int32')
            label = paddle.to_tensor(label_np)

            dataset = TensorDataset([input, label])
            assert len(dataset) == 16
<<<<<<< HEAD
            dataloader = DataLoader(
                dataset,
                places=place,
                num_workers=num_workers,
                batch_size=1,
                drop_last=True,
            )
=======
            dataloader = DataLoader(dataset,
                                    places=place,
                                    num_workers=num_workers,
                                    batch_size=1,
                                    drop_last=True)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            for i, (input, label) in enumerate(dataloader()):
                assert len(input) == 1
                assert len(label) == 1
                assert input.shape == [1, 3, 4]
                assert label.shape == [1, 1]
<<<<<<< HEAD
                assert isinstance(
                    input, (fluid.core.VarBase, fluid.core.eager.Tensor)
                )
                assert isinstance(
                    label, (fluid.core.VarBase, fluid.core.eager.Tensor)
                )
                assert np.allclose(input.numpy(), input_np[i])
                assert np.allclose(label.numpy(), label_np[i])

    def test_main(self):
=======
                assert isinstance(input,
                                  (fluid.core.VarBase, fluid.core.eager.Tensor))
                assert isinstance(label,
                                  (fluid.core.VarBase, fluid.core.eager.Tensor))
                assert np.allclose(input.numpy(), input_np[i])
                assert np.allclose(label.numpy(), label_np[i])

    def func_test_main(self):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        places = [paddle.CPUPlace()]
        if paddle.is_compiled_with_cuda():
            places.append(paddle.CUDAPlace(0))
        for p in places:
            self.run_main(num_workers=0, places=p)

<<<<<<< HEAD

class TestComposeDataset(unittest.TestCase):
    def test_main(self):
=======
    def test_main(self):
        with _test_eager_guard():
            self.func_test_main()
        self.func_test_main()


class TestComposeDataset(unittest.TestCase):

    def func_test_main(self):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        paddle.static.default_startup_program().random_seed = 1
        paddle.static.default_main_program().random_seed = 1

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

<<<<<<< HEAD

class TestRandomSplitApi(unittest.TestCase):
    def test_main(self):
=======
    def test_main(self):
        with _test_eager_guard():
            self.func_test_main()
        self.func_test_main()


class TestRandomSplitApi(unittest.TestCase):

    def func_test_main(self):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        paddle.static.default_startup_program().random_seed = 1
        paddle.static.default_main_program().random_seed = 1

        dataset1, dataset2 = paddle.io.random_split(range(5), [1, 4])

        self.assertTrue(len(dataset1) == 1)
        self.assertTrue(len(dataset2) == 4)

        elements_list = list(range(5))

        for _, val in enumerate(dataset1):
            elements_list.remove(val)

        for _, val in enumerate(dataset2):
            elements_list.remove(val)

        self.assertTrue(len(elements_list) == 0)

<<<<<<< HEAD

class TestRandomSplitError(unittest.TestCase):
    def test_errors(self):
=======
    def test_main(self):
        with _test_eager_guard():
            self.func_test_main()
        self.func_test_main()


class TestRandomSplitError(unittest.TestCase):

    def func_test_errors(self):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        paddle.static.default_startup_program().random_seed = 1
        paddle.static.default_main_program().random_seed = 1

        self.assertRaises(ValueError, paddle.io.random_split, range(5), [3, 8])
        self.assertRaises(ValueError, paddle.io.random_split, range(5), [8])
        self.assertRaises(ValueError, paddle.io.random_split, range(5), [])

<<<<<<< HEAD

class TestSubsetDataset(unittest.TestCase):
=======
    def test_errors(self):
        with _test_eager_guard():
            self.func_test_errors()
        self.func_test_errors()


class TestSubsetDataset(unittest.TestCase):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def run_main(self, num_workers, places):
        paddle.static.default_startup_program().random_seed = 1
        paddle.static.default_main_program().random_seed = 1

        input_np = np.random.random([5, 3, 4]).astype('float32')
        input = paddle.to_tensor(input_np)
        label_np = np.random.random([5, 1]).astype('int32')
        label = paddle.to_tensor(label_np)

        dataset = TensorDataset([input, label])
        even_subset = paddle.io.Subset(dataset, [0, 2, 4])
        odd_subset = paddle.io.Subset(dataset, [1, 3])

        assert len(dataset) == 5

        def prepare_dataloader(dataset):
<<<<<<< HEAD
            return DataLoader(
                dataset,
                places=places,
                num_workers=num_workers,
                batch_size=1,
                drop_last=True,
            )
=======
            return DataLoader(dataset,
                              places=places,
                              num_workers=num_workers,
                              batch_size=1,
                              drop_last=True)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        dataloader = prepare_dataloader(dataset)
        dataloader_even = prepare_dataloader(even_subset)
        dataloader_odd = prepare_dataloader(odd_subset)

        def assert_basic(input, label):
            assert len(input) == 1
            assert len(label) == 1
            assert input.shape == [1, 3, 4]
            assert label.shape == [1, 1]
<<<<<<< HEAD
            assert isinstance(
                input, (fluid.core.VarBase, fluid.core.eager.Tensor)
            )
            assert isinstance(
                label, (fluid.core.VarBase, fluid.core.eager.Tensor)
            )
=======
            assert isinstance(input,
                              (fluid.core.VarBase, fluid.core.eager.Tensor))
            assert isinstance(label,
                              (fluid.core.VarBase, fluid.core.eager.Tensor))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        elements_list = list()
        for _, (input, label) in enumerate(dataloader()):
            assert_basic(input, label)
            elements_list.append(label)

        for _, (input, label) in enumerate(dataloader_even()):
            assert_basic(input, label)
            elements_list.remove(label)

        odd_list = list()
        for _, (input, label) in enumerate(dataloader_odd()):
            assert_basic(input, label)
            odd_list.append(label)

        self.assertEqual(odd_list, elements_list)

<<<<<<< HEAD
    def test_main(self):
=======
    def func_test_main(self):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        paddle.static.default_startup_program().random_seed = 1
        paddle.static.default_main_program().random_seed = 1

        places = [paddle.CPUPlace()]
        if paddle.is_compiled_with_cuda():
            places.append(paddle.CUDAPlace(0))
        for p in places:
            self.run_main(num_workers=0, places=p)

<<<<<<< HEAD

class TestChainDataset(unittest.TestCase):
=======
    def test_main(self):
        with _test_eager_guard():
            self.func_test_main()
        self.func_test_main()


class TestChainDataset(unittest.TestCase):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def run_main(self, num_workers, places):
        paddle.static.default_startup_program().random_seed = 1
        paddle.static.default_main_program().random_seed = 1

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

<<<<<<< HEAD
    def test_main(self):
=======
    def func_test_main(self):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        places = [paddle.CPUPlace()]
        if paddle.is_compiled_with_cuda():
            places.append(paddle.CUDAPlace(0))
        for p in places:
            self.run_main(num_workers=0, places=p)

<<<<<<< HEAD

class NumpyMixTensorDataset(Dataset):
=======
    def test_main(self):
        with _test_eager_guard():
            self.func_test_main()
        self.func_test_main()


class NumpyMixTensorDataset(Dataset):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def __init__(self, sample_num):
        self.sample_num = sample_num

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        np.random.seed(idx)
        image = np.random.random([IMAGE_SIZE]).astype('float32')
<<<<<<< HEAD
        label = np.random.randint(0, 9, (1,)).astype('int64')
=======
        label = np.random.randint(0, 9, (1, )).astype('int64')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        return paddle.to_tensor(image, place=paddle.CPUPlace()), label


class TestNumpyMixTensorDataset(TestTensorDataset):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def run_main(self, num_workers, places):
        paddle.static.default_startup_program().random_seed = 1
        paddle.static.default_main_program().random_seed = 1
        place = paddle.CPUPlace()
        with fluid.dygraph.guard(place):
            dataset = NumpyMixTensorDataset(16)
            assert len(dataset) == 16
<<<<<<< HEAD
            dataloader = DataLoader(
                dataset,
                places=place,
                num_workers=num_workers,
                batch_size=1,
                drop_last=True,
            )
=======
            dataloader = DataLoader(dataset,
                                    places=place,
                                    num_workers=num_workers,
                                    batch_size=1,
                                    drop_last=True)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            for i, (input, label) in enumerate(dataloader()):
                assert len(input) == 1
                assert len(label) == 1
                assert input.shape == [1, IMAGE_SIZE]
                assert label.shape == [1, 1]
<<<<<<< HEAD
                assert isinstance(
                    input, (fluid.core.VarBase, fluid.core.eager.Tensor)
                )
                assert isinstance(
                    label, (fluid.core.VarBase, fluid.core.eager.Tensor)
                )


class ComplextDataset(Dataset):
=======
                assert isinstance(input,
                                  (fluid.core.VarBase, fluid.core.eager.Tensor))
                assert isinstance(label,
                                  (fluid.core.VarBase, fluid.core.eager.Tensor))


class ComplextDataset(Dataset):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def __init__(self, sample_num):
        self.sample_num = sample_num

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
<<<<<<< HEAD
        return (
            3.1,
            'abc',
            paddle.to_tensor(
                np.random.random([IMAGE_SIZE]).astype('float32'),
                place=paddle.CPUPlace(),
            ),
            [1, np.random.random([2]).astype('float32')],
            {'a': 2.0, 'b': np.random.random([2]).astype('float32')},
        )


class TestComplextDataset(unittest.TestCase):
=======
        return (3.1, 'abc',
                paddle.to_tensor(np.random.random([IMAGE_SIZE
                                                   ]).astype('float32'),
                                 place=paddle.CPUPlace()),
                [1, np.random.random([2]).astype('float32')], {
                    'a': 2.0,
                    'b': np.random.random([2]).astype('float32')
                })


class TestComplextDataset(unittest.TestCase):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def run_main(self, num_workers):
        paddle.static.default_startup_program().random_seed = 1
        paddle.static.default_main_program().random_seed = 1
        place = paddle.CPUPlace()
        with fluid.dygraph.guard(place):
            dataset = ComplextDataset(16)
            assert len(dataset) == 16
<<<<<<< HEAD
            dataloader = DataLoader(
                dataset,
                places=place,
                num_workers=num_workers,
                batch_size=2,
                drop_last=True,
            )
=======
            dataloader = DataLoader(dataset,
                                    places=place,
                                    num_workers=num_workers,
                                    batch_size=2,
                                    drop_last=True)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            for i, data in enumerate(dataloader()):
                assert len(data) == 5
                # data[0]: collate 3.1
                assert data[0].shape == [2]
                assert isinstance(data[1], list)
                # data[1]: collate 'abc'
                assert len(data[1]) == 2
                assert isinstance(data[1][0], str)
                assert isinstance(data[1][1], str)
                # data[2]: collate tensor
                assert data[2].shape == [2, IMAGE_SIZE]
                # data[3]: collate list
                assert isinstance(data[3], list)
                assert data[3][0].shape == [2]
                assert data[3][1].shape == [2, 2]
                # data[4]: collate dict
                assert isinstance(data[4], dict)
                assert data[4]['a'].shape == [2]
                assert data[4]['b'].shape == [2, 2]

<<<<<<< HEAD
    def test_main(self):
        for num_workers in [0, 2]:
            self.run_main(num_workers)


class SingleFieldDataset(Dataset):
=======
    def func_test_main(self):
        for num_workers in [0, 2]:
            self.run_main(num_workers)

    def test_main(self):
        with _test_eager_guard():
            self.func_test_main()
        self.func_test_main()


class SingleFieldDataset(Dataset):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def __init__(self, sample_num):
        self.sample_num = sample_num

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        return np.random.random((2, 3)).astype('float32')


class TestSingleFieldDataset(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_dataset(self):
        self.sample_num = 16
        self.dataset = SingleFieldDataset(self.sample_num)

    def run_main(self, num_workers):
        paddle.static.default_startup_program().random_seed = 1
        paddle.static.default_main_program().random_seed = 1
        place = paddle.CPUPlace()
        with fluid.dygraph.guard(place):
            self.init_dataset()
<<<<<<< HEAD
            dataloader = DataLoader(
                self.dataset,
                places=place,
                num_workers=num_workers,
                batch_size=2,
                drop_last=True,
            )

            for i, data in enumerate(dataloader()):
                assert isinstance(
                    data, (fluid.core.VarBase, fluid.core.eager.Tensor)
                )
                assert data.shape == [2, 2, 3]

    def test_main(self):
        for num_workers in [0, 2]:
            self.run_main(num_workers)


class SingleFieldIterableDataset(IterableDataset):
=======
            dataloader = DataLoader(self.dataset,
                                    places=place,
                                    num_workers=num_workers,
                                    batch_size=2,
                                    drop_last=True)

            for i, data in enumerate(dataloader()):
                assert isinstance(data,
                                  (fluid.core.VarBase, fluid.core.eager.Tensor))
                assert data.shape == [2, 2, 3]

    def func_test_main(self):
        for num_workers in [0, 2]:
            self.run_main(num_workers)

    def test_main(self):
        with _test_eager_guard():
            self.func_test_main()
        self.func_test_main()


class SingleFieldIterableDataset(IterableDataset):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def __init__(self, sample_num):
        self.sample_num = sample_num

    def __iter__(self):
        for _ in range(self.sample_num):
            yield np.random.random((2, 3)).astype('float32')


class TestSingleFieldIterableDataset(TestSingleFieldDataset):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_dataset(self):
        self.sample_num = 16
        self.dataset = SingleFieldIterableDataset(self.sample_num)


class TestDataLoaderGenerateStates(unittest.TestCase):
<<<<<<< HEAD
    def setUp(self):
        self.inputs = [(0, 1), (0, 2), (1, 3)]
        self.outputs = [
            [1835504127, 1731038949, 1320224556, 2330041505],
            [2834126987, 2358157858, 1860244682, 1437227251],
            [457190280, 2660306227, 859341110, 354512857],
        ]

    def test_main(self):
        from paddle.fluid.dataloader.worker import _generate_states

=======

    def setUp(self):
        self.inputs = [(0, 1), (0, 2), (1, 3)]
        self.outputs = [[1835504127, 1731038949, 1320224556, 2330041505],
                        [2834126987, 2358157858, 1860244682, 1437227251],
                        [457190280, 2660306227, 859341110, 354512857]]

    def func_test_main(self):
        from paddle.fluid.dataloader.worker import _generate_states
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        for inp, outp in zip(self.inputs, self.outputs):
            out = _generate_states(*inp)
            assert out == outp

<<<<<<< HEAD

class TestDatasetWithDropLast(unittest.TestCase):
    def run_main(self, dataset, num_samples, batch_size):
        for num_workers in [0, 1]:
            for drop_last in [True, False]:
                steps = (
                    num_samples + (1 - int(drop_last)) * (batch_size - 1)
                ) // batch_size
                dataloader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    drop_last=drop_last,
                    num_workers=num_workers,
                )
=======
    def test_main(self):
        with _test_eager_guard():
            self.func_test_main()
        self.func_test_main()


class TestDatasetWithDropLast(unittest.TestCase):

    def run_main(self, dataset, num_samples, batch_size):
        for num_workers in [0, 1]:
            for drop_last in [True, False]:
                steps = (num_samples + (1 - int(drop_last)) * \
                        (batch_size - 1)) // batch_size
                dataloader = DataLoader(dataset,
                                        batch_size=batch_size,
                                        drop_last=drop_last,
                                        num_workers=num_workers)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                datas = []
                for data in dataloader:
                    datas.append(data)
                assert len(datas) == steps

<<<<<<< HEAD
    def test_map_dataset(self):
        dataset = RandomDataset(10)
        self.run_main(dataset, 10, 3)

    def test_iterable_dataset(self):
        dataset = RandomIterableDataset(10)
        self.run_main(dataset, 10, 3)

=======
    def func_test_map_dataset(self):
        dataset = RandomDataset(10)
        self.run_main(dataset, 10, 3)

    def test_map_dataset(self):
        with _test_eager_guard():
            self.func_test_map_dataset()
        self.func_test_map_dataset()

    def func_test_iterable_dataset(self):
        dataset = RandomIterableDataset(10)
        self.run_main(dataset, 10, 3)

    def test_iterable_dataset(self):
        with _test_eager_guard():
            self.func_test_iterable_dataset()
        self.func_test_iterable_dataset()

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

if __name__ == '__main__':
    unittest.main()
