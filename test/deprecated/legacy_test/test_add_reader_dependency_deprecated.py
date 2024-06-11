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

import time
import unittest

import numpy as np

import paddle
from paddle import base
from paddle.base.layer_helper import LayerHelper

paddle.enable_static()


def inplace_add(x, bias):
    helper = LayerHelper('scale', **locals())
    helper.append_op(
        type='scale',
        inputs={'X': [x]},
        outputs={'Out': [x]},
        attrs={'bias': bias},
    )
    return x


class TestAddReaderDependency(unittest.TestCase):
    def setUp(self):
        self.batch_num = 3
        self.sleep_time = 2
        self.use_double_buffer = True

    def test_main(self):
        self.run_main(base.CPUPlace())

        if base.is_compiled_with_cuda():
            self.run_main(base.CUDAPlace(0))

    def run_main(self, place):
        with base.program_guard(base.Program(), base.Program()):
            with base.scope_guard(base.Scope()):
                tmp_in = paddle.static.data(
                    name='tmp_in', dtype='float32', shape=[1]
                )
                loader = base.io.DataLoader.from_generator(
                    feed_list=[tmp_in],
                    capacity=16,
                    iterable=False,
                    use_double_buffer=self.use_double_buffer,
                )

                def data_source():
                    for _ in range(self.batch_num):
                        time.sleep(self.sleep_time)  # sleep some times
                        yield np.random.uniform(
                            low=-1, high=1, size=[1]
                        ).astype('float32'),

                persistable_in = paddle.static.data(
                    name='persistable_in', dtype='float32', shape=[1]
                )
                persistable_in.persistable = True

                persistable_in = inplace_add(persistable_in, bias=1)
                prog = base.CompiledProgram(base.default_main_program())

                exe = base.Executor(place)

                loader.set_batch_generator(data_source)
                loader.start()
                batch_id = 0
                try:
                    while True:
                        if batch_id == 0:
                            feed = {
                                persistable_in.name: np.array([-1]).astype(
                                    'float32'
                                )
                            }
                        else:
                            feed = None

                        (ret,) = exe.run(
                            prog, feed=feed, fetch_list=[persistable_in]
                        )
                        self.assertEqual(ret.shape, (1,))
                        self.assertEqual(ret[0], batch_id)
                        batch_id += 1
                except base.core.EOFException:
                    loader.reset()

                    self.assertEqual(batch_id, self.batch_num)
                    t = (
                        base.global_scope()
                        .find_var(persistable_in.name)
                        .get_tensor()
                    )
                    t_val = np.array(t)
                    self.assertEqual(t_val.shape, (1,))
                    self.assertEqual(t_val[0] + 1, batch_id)


class TestAddReaderDependencyWithoutDoubleBuffer(TestAddReaderDependency):
    def setUp(self):
        self.batch_num = 3
        self.sleep_time = 2
        self.use_double_buffer = False


if __name__ == '__main__':
    unittest.main()
