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

import time
import unittest

import numpy as np

import paddle
import paddle.fluid as fluid
from paddle.fluid.reader import DataLoaderBase

EPOCH_NUM = 20
BATCH_SIZE = 32
BATCH_NUM = 20
CLASS_NUM = 10


def random_reader():
    np.random.seed(1)
    for i in range(BATCH_SIZE * BATCH_NUM):
        image = np.random.random([784])
        label = np.random.random_integers(low=0, high=CLASS_NUM - 1)
        yield image, label


def simple_fc_net(places, use_legacy_py_reader, use_double_buffer):
    paddle.seed(1)
    paddle.framework.random._manual_program_seed(1)
    startup_prog = fluid.Program()
    main_prog = fluid.Program()

    with fluid.unique_name.guard():
        with fluid.program_guard(main_prog, startup_prog):
            image = fluid.layers.data(
                name='image', shape=[784], dtype='float32'
            )
            label = fluid.layers.data(name='label', shape=[1], dtype='int64')
            py_reader = fluid.io.DataLoader.from_generator(
                feed_list=[image, label],
                capacity=4,
                iterable=not use_legacy_py_reader,
                use_double_buffer=use_double_buffer,
            )
            hidden = image
            for hidden_size in [10, 20, 30]:
                hidden = fluid.layers.fc(
                    hidden,
                    size=hidden_size,
                    act='tanh',
                    bias_attr=fluid.ParamAttr(
                        initializer=fluid.initializer.Constant(value=1.0)
                    ),
                )

            predict_label = fluid.layers.fc(
                hidden, size=CLASS_NUM, act='softmax'
            )
            loss = paddle.mean(
                paddle.nn.functional.cross_entropy(
                    input=predict_label,
                    label=label,
                    reduction='none',
                    use_softmax=False,
                )
            )

            optimizer = fluid.optimizer.Adam()
            optimizer.minimize(loss)
    return startup_prog, main_prog, py_reader, loss


class TestBase(unittest.TestCase):
    def run_main(
        self,
        use_legacy_py_reader,
        with_data_parallel,
        places,
        use_double_buffer,
    ):
        scope = fluid.Scope()
        with fluid.scope_guard(scope):
            startup_prog, main_prog, py_reader, loss = simple_fc_net(
                places, use_legacy_py_reader, use_double_buffer
            )

            reader = paddle.batch(random_reader, batch_size=BATCH_SIZE)

            ps = places if use_double_buffer else fluid.cpu_places(len(places))

            py_reader.set_sample_list_generator(
                reader, places=ps if py_reader.iterable else None
            )

            exe = fluid.Executor(place=places[0])
            exe.run(startup_prog)

            prog = fluid.CompiledProgram(main_prog)
            if with_data_parallel:
                prog = prog.with_data_parallel(
                    loss_name=loss.name, places=places
                )

            step = 0
            step_list = []
            loss_list = []
            start_t = time.time()
            if not py_reader.iterable:
                for _ in range(EPOCH_NUM):
                    step = 0
                    py_reader.start()
                    while True:
                        try:
                            (L,) = exe.run(
                                program=prog,
                                fetch_list=[loss],
                                use_program_cache=True,
                            )
                            loss_list.append(np.mean(L))
                            step += 1
                        except fluid.core.EOFException:
                            py_reader.reset()
                            break
                    step_list.append(step)
            else:
                for _ in range(EPOCH_NUM):
                    step = 0
                    for d in py_reader():
                        assert len(d) == len(places), "{} != {}".format(
                            len(d), len(places)
                        )
                        for i, item in enumerate(d):
                            image = item['image']
                            label = item['label']
                            assert image.shape() == [BATCH_SIZE, 784]
                            assert label.shape() == [BATCH_SIZE, 1]
                            assert image._place()._equals(ps[i])
                            assert label._place()._equals(ps[i])
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
        for with_data_parallel in [True, False]:
            for p in self.prepare_places(with_data_parallel):
                for use_double_buffer in [False, True]:
                    results = []
                    for use_legacy_py_reader in [False, True]:
                        print(p, use_double_buffer, use_legacy_py_reader)
                        ret = self.run_main(
                            use_legacy_py_reader=use_legacy_py_reader,
                            with_data_parallel=with_data_parallel,
                            places=p,
                            use_double_buffer=use_double_buffer,
                        )
                        results.append(ret)
                    if not use_double_buffer:
                        diff = np.max(
                            np.abs(results[0]['loss'] - results[1]['loss'])
                        )
                        self.assertLess(diff, 1e-3)


class TestDataLoaderBaseAbstract(unittest.TestCase):
    def test_main(self):
        loader = DataLoaderBase()
        try:
            loader.__iter__()
            self.assertTrue(False)
        except NotImplementedError:
            self.assertTrue(True)

        try:
            loader.__next__()
            self.assertTrue(False)
        except NotImplementedError:
            self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
