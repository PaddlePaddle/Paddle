# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
from op_test import OpTest

import paddle.fluid as fluid
from paddle.fluid.framework import Program, program_guard


def collect_node_patch(og, max_depth):
    """
    The naive method to construct patches
    :param og: original graph
    :param max_depth: the depth of convolution filters
    :return: convolution patches
    """

    def gen(node, max_depth):
        collected = [(node, 1, 1, 0, max_depth)]

        def recurse_helper(node, depth):
            if depth > max_depth:
                return
            l = len(og[node])
            for idx, c in enumerate(og[node], 1):
                if depth + 1 < max_depth:
                    collected.append((c, idx, l, depth + 1, max_depth))
                    recurse_helper(c, depth + 1)

        recurse_helper(node, 0)
        return collected

    res = []
    for u in range(1, len(og)):
        lis = gen(u, max_depth)
        if len(lis) > 0:
            res.append(lis)
    return res


class TestTreeConvOp(OpTest):
    def setUp(self):
        self.n = 17
        self.fea_size = 3
        self.output_size = 1
        self.max_depth = 2
        self.batch_size = 2
        self.num_filters = 1
        adj_array = [
            1,
            2,
            1,
            3,
            1,
            4,
            1,
            5,
            2,
            6,
            2,
            7,
            2,
            8,
            4,
            9,
            4,
            10,
            5,
            11,
            6,
            12,
            6,
            13,
            9,
            14,
            9,
            15,
            9,
            16,
            9,
            17,
        ]
        adj = np.array(adj_array).reshape((1, self.n - 1, 2)).astype('int32')
        adj = np.tile(adj, (self.batch_size, 1, 1))
        self.op_type = 'tree_conv'
        vectors = np.random.random(
            (self.batch_size, self.n, self.fea_size)
        ).astype('float64')
        self.inputs = {
            'EdgeSet': adj,
            'NodesVector': vectors,
            'Filter': np.random.random(
                (self.fea_size, 3, self.output_size, self.num_filters)
            ).astype('float64'),
        }
        self.attrs = {'max_depth': self.max_depth}
        vectors = []
        for i in range(self.batch_size):
            vector = self.get_output_naive(i)
            vectors.append(vector)
        self.outputs = {
            'Out': np.array(vectors).astype('float64'),
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(
            ['NodesVector', 'Filter'], 'Out', max_relative_error=0.5
        )

    def get_output_naive(self, batch_id):
        og = [[] for i in range(1, self.n + 2)]
        st = np.array(self.inputs['EdgeSet'][batch_id]).tolist()
        for e in st:
            og[e[0]].append(e[1])
        patches = collect_node_patch(og, self.max_depth)
        W = np.array(self.inputs['Filter']).astype('float64')
        W = np.transpose(W, axes=[1, 0, 2, 3])
        vec = []
        for i, patch in enumerate(patches, 1):
            result = np.zeros((1, W.shape[2], W.shape[3]))
            for v in patch:
                eta_t = float(v[4] - v[3]) / float(v[4])
                eta_l = (1.0 - eta_t) * (
                    0.5 if v[2] == 1 else float(v[1] - 1.0) / float(v[2] - 1.0)
                )
                eta_r = (1.0 - eta_t) * (1.0 - eta_l)
                x = self.inputs['NodesVector'][batch_id][v[0] - 1]
                eta = (
                    np.array([eta_l, eta_r, eta_t])
                    .reshape((3, 1))
                    .astype('float64')
                )
                Wconvi = np.tensordot(eta, W, axes=([0], [0]))
                x = np.array(x).reshape((1, 1, self.fea_size))
                res = np.tensordot(x, Wconvi, axes=2)
                result = result + res
            vec.append(result)
        vec = np.concatenate(vec, axis=0)
        vec = np.concatenate(
            [
                vec,
                np.zeros(
                    (self.n - vec.shape[0], W.shape[2], W.shape[3]),
                    dtype='float64',
                ),
            ],
            axis=0,
        )
        return vec


class TestTreeConv_OpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            nodes_vector_1 = np.random.random((10, 5)).astype("float32")
            edge_set_1 = fluid.layers.data(
                name='edge_set_1', shape=[10, 2], dtype='float32'
            )
            # the nodes_vector of tree_conv must be Variable.
            self.assertRaises(
                TypeError,
                fluid.contrib.layers.tree_conv,
                nodes_vector_1,
                edge_set_1,
                3,
            )

            nodes_vector_2 = fluid.layers.data(
                name='vectors2', shape=[10, 5], dtype='float32'
            )
            edge_set_2 = np.random.random((10, 2)).astype("float32")
            # the edge_set of tree_conv must be Variable.
            self.assertRaises(
                TypeError,
                fluid.contrib.layers.tree_conv,
                nodes_vector_2,
                edge_set_2,
                3,
            )


if __name__ == "__main__":
    unittest.main()
