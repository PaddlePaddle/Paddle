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

import paddle
import paddle.fluid as fluid
import numpy as np
import time
import six
import unittest
from paddle.fluid.io import Dataset, DataLoader

EPOCH_NUM = 20
BATCH_SIZE = 32
IMAGE_SIZE = 784
SAMPLE_NUM = 80000
CLASS_NUM = 10


class RandomDataset(Dataset):
    def __init__(self, sample_num, class_num):
        self.sample_num = sample_num
        self.class_num = class_num

    def __getitem__(self, idx):
        np.random.seed(idx)
        image = np.random.random([IMAGE_SIZE]).astype('float32')
        for _ in range(100):
            image = image * (image + 0.5)
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
            for hidden_size in [10, 20, 30]:
                hidden = fluid.layers.fc(
                    hidden,
                    size=hidden_size,
                    act='tanh',
                    bias_attr=fluid.ParamAttr(
                        initializer=fluid.initializer.Constant(value=1.0)))

            predict_label = fluid.layers.fc(hidden,
                                            size=CLASS_NUM,
                                            act='softmax')
            loss = fluid.layers.mean(
                fluid.layers.cross_entropy(
                    input=predict_label, label=label))

            optimizer = fluid.optimizer.Adam()
            optimizer.minimize(loss)
    return startup_prog, main_prog, image, label, loss


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
                print("step_list", step_list)
                import sys
                sys.stdout.flush()
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
        print("time cost", ret['time'])
        return ret

    def prepare_places(self, with_data_parallel, with_cpu=True, with_gpu=True):
        places = []
        # if with_cpu:
        #     places.append([fluid.CPUPlace()])
        #     if with_data_parallel:
        #         places.append([fluid.CPUPlace()] * 2)

        if with_gpu and fluid.core.is_compiled_with_cuda():
            tmp = fluid.cuda_places()
            assert len(tmp) > 0, "no gpu detected"
            # if with_data_parallel:
            #     places.append(tmp)
            places.append([tmp[0]])
        return places

    def test_main(self):
        # for with_data_parallel in [True, False]:
        for with_data_parallel in [True]:
            for p in self.prepare_places(with_data_parallel):
                # for use_buffer_reader in [False, True]:
                for use_buffer_reader in [True]:
                    results = []
                    # for num_workers in [0, 4]:
                    for num_workers in [4]:
                        print(p, use_buffer_reader, num_workers)
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
                        self.assertLess(diff, 1e-3)


# class TestDataLoaderBaseAbstract(unittest.TestCase):
#     def test_main(self):
#         loader = DataLoaderBase()
#         try:
#             loader.__iter__()
#             self.assertTrue(False)
#         except NotImplementedError:
#             self.assertTrue(True)
#
#         try:
#             loader.__next__()
#             self.assertTrue(False)
#         except NotImplementedError:
#             self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
