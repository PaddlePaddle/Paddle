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


class TestGraphSampleNeighbors(unittest.TestCase):
    def setUp(self):
        num_nodes = 20
        edges = np.random.randint(num_nodes, size=(100, 2))
        edges = np.unique(edges, axis=0)
        self.edges_id = np.arange(0, len(edges)).astype("int64")
        sorted_edges = edges[np.argsort(edges[:, 1])]

        # Calculate dst index cumsum counts, also means colptr
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
        self.nodes = np.unique(np.random.randint(
            num_nodes, size=5)).astype("int64")
        self.sample_size = 5
        self.dst_src_dict = dst_src_dict

    def test_sample_result(self):
        paddle.disable_static()
        row = paddle.to_tensor(self.row)
        colptr = paddle.to_tensor(self.colptr)
        nodes = paddle.to_tensor(self.nodes)

        out_neighbors, out_count = paddle.incubate.graph_sample_neighbors(
            row, colptr, nodes, sample_size=self.sample_size)
        out_count_cumsum = paddle.cumsum(out_count)
        for i in range(len(out_count)):
            if i == 0:
                neighbors = out_neighbors[0:out_count_cumsum[i]]
            else:
                neighbors = out_neighbors[out_count_cumsum[i - 1]:
                                          out_count_cumsum[i]]
            # Ensure the correct sample size.
            self.assertTrue(
                out_count[i] == self.sample_size or
                out_count[i] == len(self.dst_src_dict[self.nodes[i]]))
            # Ensure no repetitive sample neighbors.
            self.assertTrue(
                neighbors.shape[0] == paddle.unique(neighbors).shape[0])
            # Ensure the correct sample neighbors.
            in_neighbors = np.isin(neighbors.numpy(),
                                   self.dst_src_dict[self.nodes[i]])
            self.assertTrue(np.sum(in_neighbors) == in_neighbors.shape[0])

    def test_sample_result_fisher_yates_sampling(self):
        paddle.disable_static()
        if fluid.core.is_compiled_with_cuda():
            row = paddle.to_tensor(self.row)
            colptr = paddle.to_tensor(self.colptr)
            nodes = paddle.to_tensor(self.nodes)
            perm_buffer = paddle.to_tensor(self.edges_id)

            out_neighbors, out_count = paddle.incubate.graph_sample_neighbors(
                row,
                colptr,
                nodes,
                perm_buffer=perm_buffer,
                sample_size=self.sample_size,
                flag_perm_buffer=True)
            out_count_cumsum = paddle.cumsum(out_count)
            for i in range(len(out_count)):
                if i == 0:
                    neighbors = out_neighbors[0:out_count_cumsum[i]]
                else:
                    neighbors = out_neighbors[out_count_cumsum[i - 1]:
                                              out_count_cumsum[i]]
                # Ensure the correct sample size.
                self.assertTrue(
                    out_count[i] == self.sample_size or
                    out_count[i] == len(self.dst_src_dict[self.nodes[i]]))
                # Ensure no repetitive sample neighbors.
                self.assertTrue(
                    neighbors.shape[0] == paddle.unique(neighbors).shape[0])
                # Ensure the correct sample neighbors.
                in_neighbors = np.isin(neighbors.numpy(),
                                       self.dst_src_dict[self.nodes[i]])
                self.assertTrue(np.sum(in_neighbors) == in_neighbors.shape[0])

    def test_sample_result_static(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            row = paddle.static.data(
                name="row", shape=self.row.shape, dtype=self.row.dtype)
            colptr = paddle.static.data(
                name="colptr", shape=self.colptr.shape, dtype=self.colptr.dtype)
            nodes = paddle.static.data(
                name="nodes", shape=self.nodes.shape, dtype=self.nodes.dtype)

            out_neighbors, out_count = paddle.incubate.graph_sample_neighbors(
                row, colptr, nodes, sample_size=self.sample_size)
            exe = paddle.static.Executor(paddle.CPUPlace())
            ret = exe.run(feed={
                'row': self.row,
                'colptr': self.colptr,
                'nodes': self.nodes
            },
                          fetch_list=[out_neighbors, out_count])
            out_neighbors, out_count = ret
            out_count_cumsum = np.cumsum(out_count)
            out_neighbors = np.split(out_neighbors, out_count_cumsum)[:-1]
            for neighbors, node, count in zip(out_neighbors, self.nodes,
                                              out_count):
                self.assertTrue(count == self.sample_size or
                                count == len(self.dst_src_dict[node]))
                self.assertTrue(
                    neighbors.shape[0] == np.unique(neighbors).shape[0])
                in_neighbors = np.isin(neighbors, self.dst_src_dict[node])
                self.assertTrue(np.sum(in_neighbors) == in_neighbors.shape[0])

    def test_raise_errors(self):
        paddle.disable_static()
        row = paddle.to_tensor(self.row)
        colptr = paddle.to_tensor(self.colptr)
        nodes = paddle.to_tensor(self.nodes)

        def check_eid_error():
            paddle.incubate.graph_sample_neighbors(
                row,
                colptr,
                nodes,
                sample_size=self.sample_size,
                return_eids=True)

        def check_perm_buffer_error():
            paddle.incubate.graph_sample_neighbors(
                row,
                colptr,
                nodes,
                sample_size=self.sample_size,
                flag_perm_buffer=True)

        self.assertRaises(ValueError, check_eid_error)
        self.assertRaises(ValueError, check_perm_buffer_error)

    def test_sample_result_with_eids(self):
        paddle.disable_static()
        row = paddle.to_tensor(self.row)
        colptr = paddle.to_tensor(self.colptr)
        nodes = paddle.to_tensor(self.nodes)
        eids = paddle.to_tensor(self.edges_id)
        perm_buffer = paddle.to_tensor(self.edges_id)

        out_neighbors, out_count, out_eids = paddle.incubate.graph_sample_neighbors(
            row,
            colptr,
            nodes,
            eids=eids,
            sample_size=self.sample_size,
            return_eids=True)

        out_neighbors, out_count, out_eids = paddle.incubate.graph_sample_neighbors(
            row,
            colptr,
            nodes,
            eids=eids,
            perm_buffer=perm_buffer,
            sample_size=self.sample_size,
            return_eids=True,
            flag_perm_buffer=True)

        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            row = paddle.static.data(
                name="row", shape=self.row.shape, dtype=self.row.dtype)
            colptr = paddle.static.data(
                name="colptr", shape=self.colptr.shape, dtype=self.colptr.dtype)
            nodes = paddle.static.data(
                name="nodes", shape=self.nodes.shape, dtype=self.nodes.dtype)
            eids = paddle.static.data(
                name="eids", shape=self.edges_id.shape, dtype=self.nodes.dtype)

            out_neighbors, out_count, out_eids = paddle.incubate.graph_sample_neighbors(
                row,
                colptr,
                nodes,
                eids,
                sample_size=self.sample_size,
                return_eids=True)
            exe = paddle.static.Executor(paddle.CPUPlace())
            ret = exe.run(feed={
                'row': self.row,
                'colptr': self.colptr,
                'nodes': self.nodes,
                'eids': self.edges_id
            },
                          fetch_list=[out_neighbors, out_count, out_eids])


if __name__ == "__main__":
    unittest.main()
