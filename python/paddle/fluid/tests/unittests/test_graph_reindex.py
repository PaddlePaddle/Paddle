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
        self.x = np.arange(5).astype("int64")
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
        self.num_nodes = np.max(np.concatenate([self.x, self.neighbors])) + 1

    def test_reindex_result(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.x)
        neighbors = paddle.to_tensor(self.neighbors)
        count = paddle.to_tensor(self.count)
        value_buffer = paddle.full([self.num_nodes], -1, dtype="int32")
        index_buffer = paddle.full([self.num_nodes], -1, dtype="int32")

        reindex_src, reindex_dst, out_nodes = \
            paddle.incubate.graph_reindex(x, neighbors, count)
        np.testing.assert_allclose(self.reindex_src, reindex_src, rtol=1e-05)
        np.testing.assert_allclose(self.reindex_dst, reindex_dst, rtol=1e-05)
        np.testing.assert_allclose(self.out_nodes, out_nodes, rtol=1e-05)

        reindex_src, reindex_dst, out_nodes = \
            paddle.incubate.graph_reindex(x, neighbors, count,
                                          value_buffer, index_buffer,
                                          flag_buffer_hashtable=True)
        np.testing.assert_allclose(self.reindex_src, reindex_src, rtol=1e-05)
        np.testing.assert_allclose(self.reindex_dst, reindex_dst, rtol=1e-05)
        np.testing.assert_allclose(self.out_nodes, out_nodes, rtol=1e-05)

    def test_heter_reindex_result(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.x)
        neighbors = paddle.to_tensor(self.neighbors)
        neighbors = paddle.concat([neighbors, neighbors])
        count = paddle.to_tensor(self.count)
        count = paddle.concat([count, count])

        reindex_src, reindex_dst, out_nodes = \
            paddle.incubate.graph_reindex(x, neighbors, count)
        np.testing.assert_allclose(self.reindex_src,
                                   reindex_src[:self.neighbors.shape[0]],
                                   rtol=1e-05)
        np.testing.assert_allclose(self.reindex_src,
                                   reindex_src[self.neighbors.shape[0]:],
                                   rtol=1e-05)
        np.testing.assert_allclose(self.reindex_dst,
                                   reindex_dst[:self.neighbors.shape[0]],
                                   rtol=1e-05)
        np.testing.assert_allclose(self.reindex_dst,
                                   reindex_dst[self.neighbors.shape[0]:],
                                   rtol=1e-05)
        np.testing.assert_allclose(self.out_nodes, out_nodes, rtol=1e-05)

    def test_heter_reindex_result_v2(self):
        paddle.disable_static()
        x = np.arange(5).astype("int64")
        neighbors1 = np.random.randint(100, size=20).astype("int64")
        count1 = np.array([2, 8, 4, 3, 3], dtype="int32")
        neighbors2 = np.random.randint(100, size=20).astype("int64")
        count2 = np.array([4, 5, 1, 6, 4], dtype="int32")
        neighbors = np.concatenate([neighbors1, neighbors2])
        counts = np.concatenate([count1, count2])

        # Get numpy result.
        out_nodes = list(x)
        for neighbor in neighbors:
            if neighbor not in out_nodes:
                out_nodes.append(neighbor)
        out_nodes = np.array(out_nodes, dtype="int64")
        reindex_dict = {node: ind for ind, node in enumerate(out_nodes)}
        reindex_src = np.array([reindex_dict[node] for node in neighbors])
        reindex_dst = []
        for count in [count1, count2]:
            for node, c in zip(x, count):
                for i in range(c):
                    reindex_dst.append(reindex_dict[node])
        reindex_dst = np.array(reindex_dst, dtype="int64")

        reindex_src_, reindex_dst_, out_nodes_ = \
            paddle.incubate.graph_reindex(paddle.to_tensor(x),
                                          paddle.to_tensor(neighbors),
                                          paddle.to_tensor(counts))
        np.testing.assert_allclose(reindex_src, reindex_src_, rtol=1e-05)
        np.testing.assert_allclose(reindex_dst, reindex_dst_, rtol=1e-05)
        np.testing.assert_allclose(out_nodes, out_nodes_, rtol=1e-05)

    def test_reindex_result_static(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(name="x",
                                   shape=self.x.shape,
                                   dtype=self.x.dtype)
            neighbors = paddle.static.data(name="neighbors",
                                           shape=self.neighbors.shape,
                                           dtype=self.neighbors.dtype)
            count = paddle.static.data(name="count",
                                       shape=self.count.shape,
                                       dtype=self.count.dtype)
            value_buffer = paddle.static.data(name="value_buffer",
                                              shape=[self.num_nodes],
                                              dtype="int32")
            index_buffer = paddle.static.data(name="index_buffer",
                                              shape=[self.num_nodes],
                                              dtype="int32")

            reindex_src_1, reindex_dst_1, out_nodes_1 = \
                paddle.incubate.graph_reindex(x, neighbors, count)
            reindex_src_2, reindex_dst_2, out_nodes_2 = \
                paddle.incubate.graph_reindex(x, neighbors, count,
                                              value_buffer, index_buffer,
                                              flag_buffer_hashtable=True)

            exe = paddle.static.Executor(paddle.CPUPlace())
            ret = exe.run(feed={
                'x':
                self.x,
                'neighbors':
                self.neighbors,
                'count':
                self.count,
                'value_buffer':
                np.full([self.num_nodes], -1, dtype="int32"),
                'index_buffer':
                np.full([self.num_nodes], -1, dtype="int32")
            },
                          fetch_list=[
                              reindex_src_1, reindex_dst_1, out_nodes_1,
                              reindex_src_2, reindex_dst_2, out_nodes_2
                          ])
            reindex_src_1, reindex_dst_1, out_nodes_1, reindex_src_2, \
                reindex_dst_2, out_nodes_2 = ret
            np.testing.assert_allclose(self.reindex_src,
                                       reindex_src_1,
                                       rtol=1e-05)
            np.testing.assert_allclose(self.reindex_dst,
                                       reindex_dst_1,
                                       rtol=1e-05)
            np.testing.assert_allclose(self.out_nodes, out_nodes_1, rtol=1e-05)
            np.testing.assert_allclose(self.reindex_src,
                                       reindex_src_2,
                                       rtol=1e-05)
            np.testing.assert_allclose(self.reindex_dst,
                                       reindex_dst_2,
                                       rtol=1e-05)
            np.testing.assert_allclose(self.out_nodes, out_nodes_2, rtol=1e-05)


class TestGeometricGraphReindex(unittest.TestCase):

    def setUp(self):
        self.x = np.arange(5).astype("int64")
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
        self.num_nodes = np.max(np.concatenate([self.x, self.neighbors])) + 1

    def test_reindex_result(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.x)
        neighbors = paddle.to_tensor(self.neighbors)
        count = paddle.to_tensor(self.count)
        value_buffer = paddle.full([self.num_nodes], -1, dtype="int32")
        index_buffer = paddle.full([self.num_nodes], -1, dtype="int32")

        reindex_src, reindex_dst, out_nodes = \
            paddle.geometric.reindex_graph(x, neighbors, count)
        np.testing.assert_allclose(self.reindex_src, reindex_src, rtol=1e-05)
        np.testing.assert_allclose(self.reindex_dst, reindex_dst, rtol=1e-05)
        np.testing.assert_allclose(self.out_nodes, out_nodes, rtol=1e-05)

        reindex_src, reindex_dst, out_nodes = \
            paddle.geometric.reindex_graph(x, neighbors, count,
                                          value_buffer, index_buffer)
        np.testing.assert_allclose(self.reindex_src, reindex_src, rtol=1e-05)
        np.testing.assert_allclose(self.reindex_dst, reindex_dst, rtol=1e-05)
        np.testing.assert_allclose(self.out_nodes, out_nodes, rtol=1e-05)

    def test_heter_reindex_result(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.x)
        neighbors = paddle.to_tensor(self.neighbors)
        neighbors = paddle.concat([neighbors, neighbors])
        count = paddle.to_tensor(self.count)
        count = paddle.concat([count, count])

        reindex_src, reindex_dst, out_nodes = \
            paddle.geometric.reindex_graph(x, neighbors, count)
        np.testing.assert_allclose(self.reindex_src,
                                   reindex_src[:self.neighbors.shape[0]],
                                   rtol=1e-05)
        np.testing.assert_allclose(self.reindex_src,
                                   reindex_src[self.neighbors.shape[0]:],
                                   rtol=1e-05)
        np.testing.assert_allclose(self.reindex_dst,
                                   reindex_dst[:self.neighbors.shape[0]],
                                   rtol=1e-05)
        np.testing.assert_allclose(self.reindex_dst,
                                   reindex_dst[self.neighbors.shape[0]:],
                                   rtol=1e-05)
        np.testing.assert_allclose(self.out_nodes, out_nodes, rtol=1e-05)

    def test_heter_reindex_result_v2(self):
        paddle.disable_static()
        x = np.arange(5).astype("int64")
        neighbors1 = np.random.randint(100, size=20).astype("int64")
        count1 = np.array([2, 8, 4, 3, 3], dtype="int32")
        neighbors2 = np.random.randint(100, size=20).astype("int64")
        count2 = np.array([4, 5, 1, 6, 4], dtype="int32")
        neighbors = np.concatenate([neighbors1, neighbors2])
        counts = np.concatenate([count1, count2])

        # Get numpy result.
        out_nodes = list(x)
        for neighbor in neighbors:
            if neighbor not in out_nodes:
                out_nodes.append(neighbor)
        out_nodes = np.array(out_nodes, dtype="int64")
        reindex_dict = {node: ind for ind, node in enumerate(out_nodes)}
        reindex_src = np.array([reindex_dict[node] for node in neighbors])
        reindex_dst = []
        for count in [count1, count2]:
            for node, c in zip(x, count):
                for i in range(c):
                    reindex_dst.append(reindex_dict[node])
        reindex_dst = np.array(reindex_dst, dtype="int64")

        reindex_src_, reindex_dst_, out_nodes_ = \
            paddle.geometric.reindex_graph(paddle.to_tensor(x),
                                          paddle.to_tensor(neighbors),
                                          paddle.to_tensor(counts))
        np.testing.assert_allclose(reindex_src, reindex_src_, rtol=1e-05)
        np.testing.assert_allclose(reindex_dst, reindex_dst_, rtol=1e-05)
        np.testing.assert_allclose(out_nodes, out_nodes_, rtol=1e-05)

    def test_heter_reindex_result_v3(self):
        paddle.disable_static()
        x = np.arange(5).astype("int64")
        neighbors1 = np.random.randint(100, size=20).astype("int64")
        count1 = np.array([2, 8, 4, 3, 3], dtype="int32")
        neighbors2 = np.random.randint(100, size=20).astype("int64")
        count2 = np.array([4, 5, 1, 6, 4], dtype="int32")
        neighbors = np.concatenate([neighbors1, neighbors2])
        count = np.concatenate([count1, count2])

        # Get numpy result.
        out_nodes = list(x)
        for neighbor in neighbors:
            if neighbor not in out_nodes:
                out_nodes.append(neighbor)
        out_nodes = np.array(out_nodes, dtype="int64")
        reindex_dict = {node: ind for ind, node in enumerate(out_nodes)}
        reindex_src = np.array([reindex_dict[node] for node in neighbors])
        reindex_dst = []
        for count in [count1, count2]:
            for node, c in zip(x, count):
                for i in range(c):
                    reindex_dst.append(reindex_dict[node])
        reindex_dst = np.array(reindex_dst, dtype="int64")

        neighbors = [paddle.to_tensor(neighbors1), paddle.to_tensor(neighbors2)]
        count = [paddle.to_tensor(count1), paddle.to_tensor(count2)]
        reindex_src_, reindex_dst_, out_nodes_ = \
            paddle.geometric.reindex_heter_graph(paddle.to_tensor(x),
                                                 neighbors, count)
        np.testing.assert_allclose(reindex_src, reindex_src_, rtol=1e-05)
        np.testing.assert_allclose(reindex_dst, reindex_dst_, rtol=1e-05)
        np.testing.assert_allclose(out_nodes, out_nodes_, rtol=1e-05)

    def test_reindex_result_static(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(name="x",
                                   shape=self.x.shape,
                                   dtype=self.x.dtype)
            neighbors = paddle.static.data(name="neighbors",
                                           shape=self.neighbors.shape,
                                           dtype=self.neighbors.dtype)
            count = paddle.static.data(name="count",
                                       shape=self.count.shape,
                                       dtype=self.count.dtype)
            value_buffer = paddle.static.data(name="value_buffer",
                                              shape=[self.num_nodes],
                                              dtype="int32")
            index_buffer = paddle.static.data(name="index_buffer",
                                              shape=[self.num_nodes],
                                              dtype="int32")

            reindex_src_1, reindex_dst_1, out_nodes_1 = \
                paddle.geometric.reindex_graph(x, neighbors, count)
            reindex_src_2, reindex_dst_2, out_nodes_2 = \
                paddle.geometric.reindex_graph(x, neighbors, count,
                                              value_buffer, index_buffer)

            exe = paddle.static.Executor(paddle.CPUPlace())
            ret = exe.run(feed={
                'x':
                self.x,
                'neighbors':
                self.neighbors,
                'count':
                self.count,
                'value_buffer':
                np.full([self.num_nodes], -1, dtype="int32"),
                'index_buffer':
                np.full([self.num_nodes], -1, dtype="int32")
            },
                          fetch_list=[
                              reindex_src_1, reindex_dst_1, out_nodes_1,
                              reindex_src_2, reindex_dst_2, out_nodes_2
                          ])
            reindex_src_1, reindex_dst_1, out_nodes_1, reindex_src_2, \
                reindex_dst_2, out_nodes_2 = ret
            np.testing.assert_allclose(self.reindex_src,
                                       reindex_src_1,
                                       rtol=1e-05)
            np.testing.assert_allclose(self.reindex_dst,
                                       reindex_dst_1,
                                       rtol=1e-05)
            np.testing.assert_allclose(self.out_nodes, out_nodes_1, rtol=1e-05)
            np.testing.assert_allclose(self.reindex_src,
                                       reindex_src_2,
                                       rtol=1e-05)
            np.testing.assert_allclose(self.reindex_dst,
                                       reindex_dst_2,
                                       rtol=1e-05)
            np.testing.assert_allclose(self.out_nodes, out_nodes_2, rtol=1e-05)

    def test_heter_reindex_result_static(self):
        paddle.enable_static()
        np_x = np.arange(5).astype("int64")
        np_neighbors1 = np.random.randint(100, size=20).astype("int64")
        np_count1 = np.array([2, 8, 4, 3, 3], dtype="int32")
        np_neighbors2 = np.random.randint(100, size=20).astype("int64")
        np_count2 = np.array([4, 5, 1, 6, 4], dtype="int32")
        np_neighbors = np.concatenate([np_neighbors1, np_neighbors2])
        np_count = np.concatenate([np_count1, np_count2])

        # Get numpy result.
        out_nodes = list(np_x)
        for neighbor in np_neighbors:
            if neighbor not in out_nodes:
                out_nodes.append(neighbor)
        out_nodes = np.array(out_nodes, dtype="int64")
        reindex_dict = {node: ind for ind, node in enumerate(out_nodes)}
        reindex_src = np.array([reindex_dict[node] for node in np_neighbors])
        reindex_dst = []
        for count in [np_count1, np_count2]:
            for node, c in zip(np_x, count):
                for i in range(c):
                    reindex_dst.append(reindex_dict[node])
        reindex_dst = np.array(reindex_dst, dtype="int64")

        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(name="x", shape=[5], dtype="int64")
            neighbors1 = paddle.static.data(name="neighbors1",
                                            shape=[20],
                                            dtype="int64")
            count1 = paddle.static.data(name="count1", shape=[5], dtype="int32")
            neighbors2 = paddle.static.data(name="neighbors2",
                                            shape=[20],
                                            dtype="int64")
            count2 = paddle.static.data(name="count2", shape=[5], dtype="int32")
            value_buffer = paddle.static.data(name="value_buffer",
                                              shape=[5],
                                              dtype="int32")
            index_buffer = paddle.static.data(name="index_buffer",
                                              shape=[5],
                                              dtype="int32")

            reindex_src_1, reindex_dst_1, out_nodes_1 = \
                paddle.geometric.reindex_heter_graph(x,
                                                     [neighbors1, neighbors2],
                                                     [count1, count2])
            reindex_src_2, reindex_dst_2, out_nodes_2 = \
                paddle.geometric.reindex_heter_graph(x,
                                                     [neighbors1, neighbors2],
                                                     [count1, count2],
                                                     value_buffer, index_buffer)

            exe = paddle.static.Executor(paddle.CPUPlace())
            ret = exe.run(feed={
                'x': np_x,
                'neighbors1': np_neighbors1,
                'count1': np_count1,
                'neighbors2': np_neighbors2,
                'count2': np_count2,
                'value_buffer': np.full([5], -1, dtype="int32"),
                'index_buffer': np.full([5], -1, dtype="int32")
            },
                          fetch_list=[
                              reindex_src_1, reindex_dst_1, out_nodes_1,
                              reindex_src_2, reindex_dst_2, out_nodes_2
                          ])

            reindex_src_1, reindex_dst_1, out_nodes_1, reindex_src_2, \
                reindex_dst_2, out_nodes_2 = ret
            np.testing.assert_allclose(reindex_src, reindex_src_1, rtol=1e-05)
            np.testing.assert_allclose(reindex_dst, reindex_dst_1, rtol=1e-05)
            np.testing.assert_allclose(out_nodes, out_nodes_1, rtol=1e-05)
            np.testing.assert_allclose(reindex_src, reindex_src_2, rtol=1e-05)
            np.testing.assert_allclose(reindex_dst, reindex_dst_2, rtol=1e-05)
            np.testing.assert_allclose(out_nodes, out_nodes_2, rtol=1e-05)


if __name__ == "__main__":
    unittest.main()
