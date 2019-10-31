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

from __future__ import print_function

import unittest
import paddle.fluid as fluid
import numpy as np


class TestImperativeContainerSequential(unittest.TestCase):
    def test_sequential(self):
        data = np.random.uniform(-1, 1, [5, 10]).astype('float32')
        with fluid.dygraph.guard():
            data = fluid.dygraph.to_variable(data)
            model1 = fluid.dygraph.Sequential('model1',
                                              fluid.FC('fc1', 1),
                                              fluid.FC('fc2', 2))
            res1 = model1(data)
            self.assertListEqual(res1.shape, [5, 2])
            self.assertTrue('fc1' in model1[0]._full_name)
            model1[1] = fluid.FC('fc2_new', 3)
            res1 = model1(data)
            self.assertListEqual(res1.shape, [5, 3])
            self.assertTrue('fc2_new' in name
                            for name in [p.name for p in model1.parameters()])
            loss1 = fluid.layers.reduce_mean(res1)
            loss1.backward()

            model2 = fluid.dygraph.Sequential(
                'model2', ('l1', fluid.FC('l1', 1)), ('l2', fluid.FC('l2', 3)))
            self.assertEqual(len(model2), 2)
            res2 = model2(data)
            self.assertTrue('l1' in model2.l1.full_name())
            self.assertListEqual(res2.shape, res1.shape)
            self.assertEqual(len(model1.parameters()), len(model2.parameters()))
            del model2['l2']
            self.assertEqual(len(model2), 1)
            res2 = model2(data)
            self.assertListEqual(res2.shape, [5, 1])
            model2.add_sublayer('l3', fluid.FC('l3', 3))
            model2.add_sublayer('l4', fluid.FC('l4', 4))
            self.assertEqual(len(model2), 3)
            res2 = model2(data)
            self.assertListEqual(res2.shape, [5, 4])

            loss2 = fluid.layers.reduce_mean(res2)
            loss2.backward()


if __name__ == '__main__':
    unittest.main()
