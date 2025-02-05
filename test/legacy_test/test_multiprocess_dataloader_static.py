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

import sys
import time
import unittest

import numpy as np

import paddle
from paddle import base
from paddle.io import DataLoader, Dataset

EPOCH_NUM = 3
BATCH_SIZE = 8
IMAGE_SIZE = 32
SAMPLE_NUM = 100
CLASS_NUM = 10


class RandomDataset(Dataset):
    def __init__(self, sample_num, class_num):
        self.sample_num = sample_num
        self.class_num = class_num

    def __getitem__(self, idx):
        np.random.seed(idx)
        image = np.random.random([IMAGE_SIZE]).astype('float32')
        label = np.random.randint(0, self.class_num - 1, (1,)).astype('int64')
        return image, label

    def __len__(self):
        return self.sample_num


def simple_fc_net_static():
    startup_prog = base.Program()
    main_prog = base.Program()
    paddle.seed(1)

    with base.unique_name.guard():
        with base.program_guard(main_prog, startup_prog):
            image = paddle.static.data(
                name='image', shape=[None, IMAGE_SIZE], dtype='float32'
            )
            label = paddle.static.data(
                name='label', shape=[None, 1], dtype='int64'
            )
            hidden = image
            param_attr = base.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.8)
            )
            bias_attr = base.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.5)
            )
            for hidden_size in [10, 20, 30]:
                hidden = paddle.static.nn.fc(
                    hidden,
                    size=hidden_size,
                    activation='tanh',
                    weight_attr=param_attr,
                    bias_attr=bias_attr,
                )

            predict_label = paddle.static.nn.fc(
                hidden,
                size=CLASS_NUM,
                activation='softmax',
                weight_attr=param_attr,
                bias_attr=bias_attr,
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
    return startup_prog, main_prog, image, label, loss


def prepare_places(with_cpu=False, with_gpu=True):
    places = []
    if with_cpu:
        places.append([base.CPUPlace()])

    if with_gpu and base.core.is_compiled_with_cuda():
        tmp = base.cuda_places()[:2]
        assert len(tmp) > 0, "no gpu detected"
        places.append([tmp[0]])
    return places


class TestStaticDataLoader(unittest.TestCase):
    def run_main(self, num_workers, places, persistent_workers):
        scope = base.Scope()
        with base.scope_guard(scope):
            startup_prog, main_prog, image, label, loss = simple_fc_net_static()

            dataset = RandomDataset(SAMPLE_NUM, CLASS_NUM)
            dataloader = DataLoader(
                dataset,
                feed_list=[image, label],
                places=places,
                num_workers=num_workers,
                batch_size=BATCH_SIZE,
                return_list=False,
                drop_last=True,
                persistent_workers=persistent_workers,
            )
            assert len(dataloader) == int(SAMPLE_NUM / BATCH_SIZE)

            exe = base.Executor(place=places[0])
            exe.run(startup_prog)

            prog = main_prog

            step_list = []
            loss_list = []
            start_t = time.time()
            for _ in range(EPOCH_NUM):
                step = 0
                for d in dataloader:
                    assert len(d) == len(places), f"{len(d)} != {len(places)}"
                    for i, item in enumerate(d):
                        image = item['image']
                        label = item['label']
                        assert image.shape() == [BATCH_SIZE, IMAGE_SIZE]
                        assert label.shape() == [BATCH_SIZE, 1]
                        assert image._place()._equals(places[i])
                        assert label._place()._equals(places[i])
                    (L,) = exe.run(
                        program=prog,
                        feed=d,
                        fetch_list=[loss],
                        use_program_cache=True,
                    )
                    loss_list.append(np.mean(L))
                    step += 1
                step_list.append(step)

        end_t = time.time()
        ret = {
            "time": end_t - start_t,
            "step": step_list,
            "loss": np.array(loss_list),
        }
        print("time cost", ret['time'], 'step_list', ret['step'])
        return ret

    def test_main(self):
        for p in prepare_places():
            for persistent_workers in [True, False]:
                results = []
                for num_workers in [0, 2]:
                    print(
                        self.__class__.__name__,
                        p,
                        num_workers,
                        persistent_workers,
                    )
                    sys.stdout.flush()
                    ret = self.run_main(
                        num_workers=num_workers,
                        places=p,
                        persistent_workers=persistent_workers,
                    )
                    results.append(ret)
                diff = np.max(
                    np.abs(results[0]['loss'] - results[1]['loss'])
                    / np.abs(results[0]['loss'])
                )
                self.assertLess(diff, 1e-2)


class TestStaticDataLoaderReturnList(unittest.TestCase):
    def run_single_place(self, num_workers):
        scope = base.Scope()
        image = paddle.static.data(
            name='image', shape=[None, IMAGE_SIZE], dtype='float32'
        )
        label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
        with base.scope_guard(scope):
            dataset = RandomDataset(SAMPLE_NUM, CLASS_NUM)
            dataloader = DataLoader(
                dataset,
                feed_list=[image, label],
                num_workers=num_workers,
                batch_size=BATCH_SIZE,
                drop_last=True,
                return_list=True,
            )

            for d in dataloader:
                assert isinstance(d, list)
                assert len(d) == 2
                assert not isinstance(d[0], list)
                assert not isinstance(d[1], list)

    def run_multi_place(self, num_workers):
        scope = base.Scope()
        image = paddle.static.data(
            name='image', shape=[None, IMAGE_SIZE], dtype='float32'
        )
        label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
        with base.scope_guard(scope):
            dataset = RandomDataset(SAMPLE_NUM, CLASS_NUM)
            dataloader = DataLoader(
                dataset,
                feed_list=[image, label],
                num_workers=num_workers,
                batch_size=BATCH_SIZE,
                places=[base.CPUPlace()] * 2,
                drop_last=True,
                return_list=True,
            )

            for d in dataloader:
                assert isinstance(d, list)
                assert len(d) == 2
                assert isinstance(d[0], list)
                assert isinstance(d[1], list)

    def test_main(self):
        paddle.enable_static()
        for num_workers in [0, 2]:
            self.run_single_place(num_workers)
            self.run_multi_place(num_workers)


class RandomBatchedDataset(Dataset):
    def __init__(self, sample_num, class_num):
        self.sample_num = int(sample_num / BATCH_SIZE)
        self.class_num = class_num

    def __getitem__(self, idx):
        np.random.seed(idx)
        images = []
        labels = []
        for _ in range(BATCH_SIZE):
            image = np.random.random([IMAGE_SIZE]).astype('float32')
            label = np.random.randint(0, self.class_num - 1, (1,)).astype(
                'int64'
            )
            images.append(image)
            labels.append(label)
        return np.stack(images, axis=0), np.stack(labels, axis=0)

    def __len__(self):
        return self.sample_num


class TestStaticDataLoaderWithBatchedDataset(TestStaticDataLoader):
    def run_main(self, num_workers, places, persistent_workers):
        scope = base.Scope()
        with base.scope_guard(scope):
            startup_prog, main_prog, image, label, loss = simple_fc_net_static()

            dataset = RandomBatchedDataset(SAMPLE_NUM, CLASS_NUM)
            dataloader = DataLoader(
                dataset,
                feed_list=[image, label],
                places=places,
                num_workers=num_workers,
                batch_size=None,
                return_list=False,
                drop_last=True,
                persistent_workers=persistent_workers,
            )
            assert len(dataloader) == int(SAMPLE_NUM / BATCH_SIZE)

            exe = base.Executor(place=places[0])
            exe.run(startup_prog)

            prog = main_prog

            step_list = []
            loss_list = []
            start_t = time.time()
            for _ in range(EPOCH_NUM):
                step = 0
                for d in dataloader:
                    assert len(d) == len(places), f"{len(d)} != {len(places)}"
                    for i, item in enumerate(d):
                        image = item['image']
                        label = item['label']
                        assert image.shape() == [BATCH_SIZE, IMAGE_SIZE]
                        assert label.shape() == [BATCH_SIZE, 1]
                        assert image._place()._equals(places[i])
                        assert label._place()._equals(places[i])
                    (L,) = exe.run(
                        program=prog,
                        feed=d,
                        fetch_list=[loss],
                        use_program_cache=True,
                    )
                    loss_list.append(np.mean(L))
                    step += 1
                step_list.append(step)

        end_t = time.time()
        ret = {
            "time": end_t - start_t,
            "step": step_list,
            "loss": np.array(loss_list),
        }
        print("time cost", ret['time'], 'step_list', ret['step'])
        return ret


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
