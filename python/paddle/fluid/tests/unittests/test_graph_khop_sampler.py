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


class TestGraphKhopSampler(unittest.TestCase):
    def setUp(self):
        num_nodes = 20
        edges = np.random.randint(num_nodes, size=(100, 2))
        edges = np.unique(edges, axis=0)
        edges_id = np.arange(0, len(edges))
        sorted_edges = edges[np.argsort(edges[:, 1])]
        sorted_eid = edges_id[np.argsort(edges[:, 1])]

        # Calculate dst index cumsum counts.
        dst_count = np.zeros(num_nodes)
        dst_src_dict = {}
        for dst in range(0, num_nodes):
            true_index = sorted_edges[:, 1] == dst
            dst_count[dst] = np.sum(true_index)
            dst_src_dict[dst] = sorted_edges[:, 0][true_index]
        dst_count = dst_count.astype("int64")
        colptr = np.cumsum(dst_count)
        colptr = np.insert(colptr, 0, 0)

        self.row = sorted_edges[:, 0].astype("int64")
        self.colptr = colptr.astype("int64")
        self.sorted_eid = sorted_eid.astype("int64")
        self.nodes = np.unique(np.random.randint(
            num_nodes, size=5)).astype("int64")
        self.sample_sizes = [5, 5]
        self.dst_src_dict = dst_src_dict

    def func_sample_result(self):
        paddle.disable_static()
        row = paddle.to_tensor(self.row)
        colptr = paddle.to_tensor(self.colptr)
        nodes = paddle.to_tensor(self.nodes)

        edge_src, edge_dst, sample_index, reindex_nodes = \
            paddle.incubate.graph_khop_sampler(row, colptr,
                                                   nodes, self.sample_sizes,
                                                   return_eids=False)
        # Reindex edge_src and edge_dst to original index.
        edge_src = edge_src.reshape([-1])
        edge_dst = edge_dst.reshape([-1])
        sample_index = sample_index.reshape([-1])

        for i in range(len(edge_src)):
            edge_src[i] = sample_index[edge_src[i]]
            edge_dst[i] = sample_index[edge_dst[i]]

        for n in self.nodes:
            edge_src_n = edge_src[edge_dst == n]
            if edge_src_n.shape[0] == 0:
                continue
            # Ensure no repetitive sample neighbors.
            self.assertTrue(
                edge_src_n.shape[0] == paddle.unique(edge_src_n).shape[0])
            # Ensure the correct sample size.
            self.assertTrue(edge_src_n.shape[0] == self.sample_sizes[0] or
                            edge_src_n.shape[0] == len(self.dst_src_dict[n]))
            in_neighbors = np.isin(edge_src_n.numpy(), self.dst_src_dict[n])
            # Ensure the correct sample neighbors.
            self.assertTrue(np.sum(in_neighbors) == in_neighbors.shape[0])

    def test_sample_result(self):
        with fluid.framework._test_eager_guard():
            self.func_sample_result()
        self.func_sample_result()

    def func_uva_sample_result(self):
        paddle.disable_static()
        if paddle.fluid.core.is_compiled_with_cuda():
            row = None
            if fluid.framework.in_dygraph_mode():
                row = paddle.fluid.core.eager.to_uva_tensor(
                    self.row.astype(self.row.dtype), 0)
                sorted_eid = paddle.fluid.core.eager.to_uva_tensor(
                    self.sorted_eid.astype(self.sorted_eid.dtype), 0)
            else:
                row = paddle.fluid.core.to_uva_tensor(
                    self.row.astype(self.row.dtype))
                sorted_eid = paddle.fluid.core.to_uva_tensor(
                    self.sorted_eid.astype(self.sorted_eid.dtype))
            colptr = paddle.to_tensor(self.colptr)
            nodes = paddle.to_tensor(self.nodes)

            edge_src, edge_dst, sample_index, reindex_nodes, edge_eids = \
                paddle.incubate.graph_khop_sampler(row, colptr,
                                                       nodes, self.sample_sizes,
                                                       sorted_eids=sorted_eid,
                                                       return_eids=True)
            edge_src = edge_src.reshape([-1])
            edge_dst = edge_dst.reshape([-1])
            sample_index = sample_index.reshape([-1])

            for i in range(len(edge_src)):
                edge_src[i] = sample_index[edge_src[i]]
                edge_dst[i] = sample_index[edge_dst[i]]

            for n in self.nodes:
                edge_src_n = edge_src[edge_dst == n]
                if edge_src_n.shape[0] == 0:
                    continue
                self.assertTrue(
                    edge_src_n.shape[0] == paddle.unique(edge_src_n).shape[0])
                self.assertTrue(
                    edge_src_n.shape[0] == self.sample_sizes[0] or
                    edge_src_n.shape[0] == len(self.dst_src_dict[n]))
                in_neighbors = np.isin(edge_src_n.numpy(), self.dst_src_dict[n])
                self.assertTrue(np.sum(in_neighbors) == in_neighbors.shape[0])

    def test_uva_sample_result(self):
        with fluid.framework._test_eager_guard():
            self.func_uva_sample_result()
        self.func_uva_sample_result()

    def test_sample_result_static_with_eids(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            row = paddle.static.data(
                name="row", shape=self.row.shape, dtype=self.row.dtype)
            sorted_eids = paddle.static.data(
                name="eids",
                shape=self.sorted_eid.shape,
                dtype=self.sorted_eid.dtype)
            colptr = paddle.static.data(
                name="colptr", shape=self.colptr.shape, dtype=self.colptr.dtype)
            nodes = paddle.static.data(
                name="nodes", shape=self.nodes.shape, dtype=self.nodes.dtype)

            edge_src, edge_dst, sample_index, reindex_nodes, edge_eids = \
                paddle.incubate.graph_khop_sampler(row, colptr,
                                                       nodes, self.sample_sizes,
                                                       sorted_eids, True)
            exe = paddle.static.Executor(paddle.CPUPlace())
            ret = exe.run(feed={
                'row': self.row,
                'eids': self.sorted_eid,
                'colptr': self.colptr,
                'nodes': self.nodes
            },
                          fetch_list=[edge_src, edge_dst, sample_index])

            edge_src, edge_dst, sample_index = ret
            edge_src = edge_src.reshape([-1])
            edge_dst = edge_dst.reshape([-1])
            sample_index = sample_index.reshape([-1])

            for i in range(len(edge_src)):
                edge_src[i] = sample_index[edge_src[i]]
                edge_dst[i] = sample_index[edge_dst[i]]

            for n in self.nodes:
                edge_src_n = edge_src[edge_dst == n]
                if edge_src_n.shape[0] == 0:
                    continue
                self.assertTrue(
                    edge_src_n.shape[0] == np.unique(edge_src_n).shape[0])
                self.assertTrue(
                    edge_src_n.shape[0] == self.sample_sizes[0] or
                    edge_src_n.shape[0] == len(self.dst_src_dict[n]))
                in_neighbors = np.isin(edge_src_n, self.dst_src_dict[n])
                self.assertTrue(np.sum(in_neighbors) == in_neighbors.shape[0])

    def test_sample_result_static_without_eids(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            row = paddle.static.data(
                name="row", shape=self.row.shape, dtype=self.row.dtype)
            colptr = paddle.static.data(
                name="colptr", shape=self.colptr.shape, dtype=self.colptr.dtype)
            nodes = paddle.static.data(
                name="nodes", shape=self.nodes.shape, dtype=self.nodes.dtype)
            edge_src, edge_dst, sample_index, reindex_nodes = \
                paddle.incubate.graph_khop_sampler(row, colptr,
                                                       nodes, self.sample_sizes)
            exe = paddle.static.Executor(paddle.CPUPlace())
            ret = exe.run(feed={
                'row': self.row,
                'colptr': self.colptr,
                'nodes': self.nodes
            },
                          fetch_list=[edge_src, edge_dst, sample_index])
            edge_src, edge_dst, sample_index = ret
            edge_src = edge_src.reshape([-1])
            edge_dst = edge_dst.reshape([-1])
            sample_index = sample_index.reshape([-1])

            for i in range(len(edge_src)):
                edge_src[i] = sample_index[edge_src[i]]
                edge_dst[i] = sample_index[edge_dst[i]]

            for n in self.nodes:
                edge_src_n = edge_src[edge_dst == n]
                if edge_src_n.shape[0] == 0:
                    continue
                self.assertTrue(
                    edge_src_n.shape[0] == np.unique(edge_src_n).shape[0])
                self.assertTrue(
                    edge_src_n.shape[0] == self.sample_sizes[0] or
                    edge_src_n.shape[0] == len(self.dst_src_dict[n]))
                in_neighbors = np.isin(edge_src_n, self.dst_src_dict[n])
                self.assertTrue(np.sum(in_neighbors) == in_neighbors.shape[0])


if __name__ == "__main__":
    unittest.main()
