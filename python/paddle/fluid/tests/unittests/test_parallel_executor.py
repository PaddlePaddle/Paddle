#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import paddle.fluid as fluid


class ParallelExecutor(unittest.TestCase):
    def test_main(self):
        main = fluid.Program()
        startup = fluid.Program()

        with fluid.program_guard(main, startup):
            reader = fluid.layers.open_recordio_file(
                filename='tmp',
                shapes=[[-1, 784], [-1, 1]],
                lod_levels=[0, 0],
                dtypes=['float32', 'int64'])
            img, label = fluid.layers.read_file(reader)
            hidden = fluid.layers.fc(img, size=200, act='tanh')
            prediction = fluid.layers.fc(hidden, size=10, act='softmax')
            loss = fluid.layers.cross_entropy(input=prediction, label=label)
            loss = fluid.layers.mean(loss)
            adam = fluid.optimizer.Adam()
            adam.minimize(loss)
        act_places = []
        for each in [fluid.CUDAPlace(0), fluid.CUDAPlace(1)]:
            p = fluid.core.Place()
            p.set_place(each)
            act_places.append(p)

        exe = fluid.core.ParallelExecutor(
            act_places,
            set([p.name for p in main.global_block().iter_parameters()]),
            startup.desc, main.desc, loss.name, fluid.global_scope())
        exe.run()
