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

EPOCH_NUM = 60
BATCH_SIZE = 32
CLASS_NUM = 10


def random_reader():
    for i in range(BATCH_SIZE * 40):
        image = np.random.random([784])
        label = np.random.random_integers(low=0, high=CLASS_NUM - 1)
        yield image, label


def simple_fc_net(places, use_legacy_py_reader):
    startup_prog = fluid.Program()
    main_prog = fluid.Program()
    startup_prog.random_seed = 1
    main_prog.random_seed = 1
    reader = paddle.batch(random_reader, batch_size=BATCH_SIZE)

    with fluid.unique_name.guard():
        with fluid.program_guard(main_prog, startup_prog):
            if not use_legacy_py_reader:
                image = fluid.layers.data(
                    name='image', shape=[784], dtype='float32')
                label = fluid.layers.data(
                    name='label', shape=[1], dtype='int64')
                py_reader = fluid.io.PyReader(
                    feed_list=[image, label],
                    places=places,
                    capacity=4,
                    multi_queue=False)
                py_reader.set_numpy_reader(reader)
            else:
                py_reader = fluid.layers.py_reader(
                    capacity=4,
                    shapes=[(-1, 784), (-1, 1)],
                    dtypes=['float32', 'int64'])
                image, label = fluid.layers.read_file(py_reader)
                py_reader.decorate_paddle_reader(reader)

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
    return startup_prog, main_prog, py_reader, loss


class TestBase(unittest.TestCase):
    def run_main(self, use_legacy_py_reader, with_data_parallel, places):
        with fluid.scope_guard(fluid.Scope()):
            startup_prog, main_prog, py_reader, loss = simple_fc_net(
                places, use_legacy_py_reader)
            exe = fluid.Executor(place=places[0])
            exe.run(startup_prog)

            prog = fluid.CompiledProgram(main_prog)
            if with_data_parallel:
                prog = prog.with_data_parallel(
                    loss_name=loss.name, places=places)

            step = 0
            start_t = time.time()
            if use_legacy_py_reader:
                for _ in six.moves.range(EPOCH_NUM):
                    py_reader.start()
                    while True:
                        try:
                            L, = exe.run(program=prog, fetch_list=[loss])
                            step += 1
                        except fluid.core.EOFException:
                            py_reader.reset()
                            break
            else:
                for _ in six.moves.range(EPOCH_NUM):
                    for d in py_reader():
                        '''
                        assert len(d) == len(places)
                        for i, item in enumerate(d):
                            image = item['image']
                            label = item['label']
                            assert image.shape() == [BATCH_SIZE, 784]
                            assert label.shape() == [BATCH_SIZE, 1]
                            assert image._place()._equals(places[i])
                            assert label._place()._equals(places[i])
                        '''
                        L, = exe.run(program=prog, feed=d, fetch_list=[loss])
                        step += 1
            end_t = time.time()
            return {"time": end_t - start_t, "step": step}

    def prepare_places(self, with_data_parallel):
        places = [[fluid.CPUPlace()], ]
        if with_data_parallel:
            places.append([fluid.CPUPlace()] * 2)

        if fluid.core.is_compiled_with_cuda():
            tmp = fluid.cuda_places()
            assert len(tmp) > 0, "no gpu detected"
            if with_data_parallel:
                places.append(tmp)
            places.append([tmp[0]])
        return places

    def test_main(self):
        for with_data_parallel in [True, False]:
            for p in self.prepare_places(with_data_parallel):
                t = []
                for use_legacy_py_reader in [False, True]:
                    ret = self.run_main(
                        use_legacy_py_reader=use_legacy_py_reader,
                        with_data_parallel=with_data_parallel,
                        places=p)
                    ret['legacy'] = use_legacy_py_reader
                    ret['data_parallel'] = with_data_parallel
                    ret['places'] = p
                    t.append(ret)

                print(t)


if __name__ == '__main__':
    unittest.main()
