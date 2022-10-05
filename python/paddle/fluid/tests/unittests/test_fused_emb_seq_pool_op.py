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
import platform
import numpy as np
from op_test import OpTest, skip_check_grad_ci
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid.op import Operator
import paddle.compat as cpt
import paddle.version as ver


@skip_check_grad_ci(reason="check_grad is called when ver.mkl() == ON"
                    "and 'Linux' in platform.platform().")
class TestFusedEmbeddingSeqPoolOp(OpTest):

    def setUp(self):
        self.op_type = "fused_embedding_seq_pool"
        self.emb_size = 6
        self.table = np.random.random((17, self.emb_size)).astype("float64")
        self.ids = np.array([[[4], [3]], [[4], [3]], [[2], [1]],
                             [[16], [1]]]).astype("int64")
        ids_expand = np.expand_dims(self.ids, axis=1)
        self.lod = [[3, 1]]
        self.attrs = {'is_sparse': True}
        self.inputs = {'W': self.table, 'Ids': (ids_expand, self.lod)}
        self.outputs = {
            'Out':
            np.reshape(
                np.array([
                    self.table[[4, 3]] + self.table[[4, 3]] +
                    self.table[[2, 1]], self.table[[16, 1]]
                ]), [len(self.lod[0]), 2 * self.emb_size])
        }

    def test_check_output(self):
        # TODO(wangzhongpu): support lod in dygraph mode
        self.check_output(check_dygraph=False)

    def test_check_grad(self):
        # TODO(wangzhongpu): support lod in dygraph mode
        if ver.mkl() == "ON" and 'Linux' in platform.platform():
            self.attrs = {'is_sparse': False}
            self.check_grad(['W'],
                            'Out',
                            no_grad_set=['Ids'],
                            check_dygraph=False)


class TestLookupTableOpWithPadding(TestFusedEmbeddingSeqPoolOp):

    def test_check_output(self):
        if ver.mkl() == "ON" and 'Linux' in platform.platform():
            ids = np.squeeze(self.ids, axis=2)
            padding_idx = np.random.choice(ids.flatten(), 1)[0]
            output = list()
            index = 0
            for count in self.lod[0]:
                arr = ids[index:count + index]
                out = np.reshape(self.table[arr.flatten()],
                                 [arr.shape[0], arr.shape[1], self.emb_size])
                idx = np.argwhere(arr == padding_idx)
                for item in idx:
                    out[item[0], item[1], :] = np.zeros(self.emb_size)
                output.append(np.sum(out, 0))
                index += count
            self.outputs = {
                'Out':
                np.reshape(np.array(output),
                           [len(self.lod[0]), 2 * self.emb_size])
            }
            self.attrs = {'padding_idx': int(padding_idx)}
            # TODO(wangzhongpu): support lod in dygraph mode
            self.check_output(check_dygraph=False)

    def test_check_grad(self):
        if ver.mkl() == "ON" and 'Linux' in platform.platform():
            ids = np.squeeze(self.ids, axis=2)
            padding_idx = np.random.choice(ids.flatten(), 1)[0]
            self.attrs = {'padding_idx': int(padding_idx), 'is_sparse': False}
            # TODO(wangzhongpu): support lod in dygraph mode
            self.check_grad(['W'],
                            'Out',
                            no_grad_set=['Ids'],
                            check_dygraph=False)


class TestFusedEmbeddingSeqPoolApi(unittest.TestCase):

    def test_api(self):
        if ver.mkl() == "ON" and 'Linux' in platform.platform():
            import paddle.fluid as fluid

            dict_size = 20
            data_t = fluid.layers.data(name='word',
                                       shape=[1],
                                       dtype='int64',
                                       lod_level=1)
            padding_idx = np.random.randint(1, 10)
            out = fluid.contrib.fused_embedding_seq_pool(
                input=data_t,
                size=[dict_size, 32],
                param_attr='w',
                padding_idx=padding_idx,
                is_sparse=False)

            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            # prepare input words' idx
            x_tensor = fluid.core.LoDTensor()
            idxs = np.random.randint(1, 10, (8)).astype("int64")

            x_tensor.set(idxs, place)
            x_tensor.set_recursive_sequence_lengths([[4, 4]])
            ret = exe.run(feed={'word': x_tensor}, fetch_list=[out])


if __name__ == "__main__":
    unittest.main()
