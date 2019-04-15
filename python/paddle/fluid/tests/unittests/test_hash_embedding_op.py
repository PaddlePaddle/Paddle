#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np
import paddle.fluid as fluid


class TestHashEmbeddingOp(unittest.TestCase):
    def test_hash_embedding(self):
        main_program = fluid.Program()
        start_up_program = fluid.Program()
        with fluid.program_guard(main_program, start_up_program):
            x = fluid.layers.data(
                name='x', shape=[1], dtype='int64', lod_level=1)
            emb_attr = fluid.ParamAttr(
                name="emb", initializer=fluid.initializer.Constant(value=1.0))
            imp_attr = fluid.ParamAttr(
                name="imp", initializer=fluid.initializer.Constant(value=0.5))
            hash_emb = fluid.layers.hash_embedding(
                input=x,
                param_attr=[emb_attr, imp_attr],
                padding_idx=0,
                embedding_size=[10, 3],
                importance_size=[100, 2])

            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())

            x = np.array([[0, 1, 2]]).astype("int64").reshape((-1, 1))
            xx = fluid.create_lod_tensor(x, [[0, 3]], place)
            res = exe.run(fluid.default_main_program(),
                          feed={'x': xx},
                          fetch_list=[hash_emb],
                          return_numpy=False)
            res = np.array(res[0])
            self.assertTrue(list(res.shape) == [3, 3])
            res = res.reshape(-1)
            benchmark = np.array([[0., 0., 0.], [1., 1., 1.],
                                  [1., 1., 1.]]).astype("float64").reshape(-1)
            self.assertTrue(all(res == benchmark))


class TestHashEmbedding2Op(unittest.TestCase):
    def test_hash_embedding(self):
        #multi-dimension input
        main_program = fluid.Program()
        start_up_program = fluid.Program()
        with fluid.program_guard(main_program, start_up_program):
            x = fluid.layers.data(
                name='x', shape=[1, 1], dtype='int64', lod_level=0)
            emb_attr = fluid.ParamAttr(
                name="emb", initializer=fluid.initializer.Constant(value=1.0))
            imp_attr = fluid.ParamAttr(
                name="imp", initializer=fluid.initializer.Constant(value=0.5))
            hash_emb = fluid.layers.hash_embedding(
                input=x,
                param_attr=[emb_attr, imp_attr],
                padding_idx=0,
                embedding_size=[10, 3],
                importance_size=[100, 2])

            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())

            x = np.array([[0, 1, 2, 2, 8, 9]]).astype("int64").reshape(
                (-1, 2, 1))
            res = exe.run(fluid.default_main_program(),
                          feed={'x': x},
                          fetch_list=[hash_emb],
                          return_numpy=False)
            res = np.array(res[0])
            self.assertTrue(list(res.shape) == [3, 2, 3])
            res = res.reshape(-1)

            benchmark = np.array([[0., 0., 0.], [1., 1., 1.], [1., 1., 1.],
                                  [1., 1., 1.], [1., 1., 1.],
                                  [1., 1., 1.]]).astype("float64").reshape(-1)
            self.assertTrue(all(res == benchmark))


if __name__ == '__main__':
    unittest.main()
