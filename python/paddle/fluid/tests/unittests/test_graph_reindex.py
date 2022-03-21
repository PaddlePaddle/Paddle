# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import paddle.fluid as fluid


class TestGraphReindex(unittest.TestCase):
    def setUp(self):
        # We suppose x is unique.
        self.x = np.unique(np.random.randint(100, size=5)).astype("int64")
        self.neighbors = np.random.randint(100, size=20).astype("int64")
        self.count = np.array([2, 8, 4, 3, 3], dtype="int32")

        # Get numpy result.
        out_nodes = list(self.x)
        for neighbor in self.neighbors:
            if neighbor not in out_nodes:
                out_nodes.append(neighbor)
        self.out_nodes = np.array(out_nodes, dtype="int64")
        reindex_dict = {node: ind for ind, node in enumerate(self.out_nodes)}
        self.reindex_src = np.array(
            [reindex_dict[node] for node in self.neighbors])
        reindex_dst = []
        for node, c in zip(self.x, self.count):
            for i in range(c):
                reindex_dst.append(reindex_dict[node])
        self.reindex_dst = np.array(reindex_dst, dtype="int64")

    def test_reindex_result(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.x)
        neighbors = paddle.to_tensor(self.neighbors)
        count = paddle.to_tensor(self.count)

        reindex_src, reindex_dst, out_nodes = \
            paddle.incubate.graph_reindex(x, neighbors, count)

        self.assertTrue(np.allclose(self.reindex_src, reindex_src))
        self.assertTrue(np.allclose(self.reindex_dst, reindex_dst))
        self.assertTrue(np.allclose(self.out_nodes, out_nodes))

    def test_reindex_result_static(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(
                name="x", shape=self.x.shape, dtype=self.x.dtype)
            neighbors = paddle.static.data(
                name="neighbors",
                shape=self.neighbors.shape,
                dtype=self.neighbors.dtype)
            count = paddle.static.data(
                name="count", shape=self.count.shape, dtype=self.count.dtype)

            reindex_src, reindex_dst, out_nodes = \
                paddle.incubate.graph_reindex(x, neighbors, count)
            exe = paddle.static.Executor(paddle.CPUPlace())
            ret = exe.run(feed={
                'x': self.x,
                'neighbors': self.neighbors,
                'count': self.count
            },
                          fetch_list=[reindex_src, reindex_dst, out_nodes])
            reindex_src, reindex_dst, out_nodes = ret
            self.assertTrue(np.allclose(self.reindex_src, reindex_src))
            self.assertTrue(np.allclose(self.reindex_dst, reindex_dst))
            self.assertTrue(np.allclose(self.out_nodes, out_nodes))


if __name__ == "__main__":
    unittest.main()
