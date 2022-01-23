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
from paddle.fluid.core import to_uva_tensor


class TestGraphSampleNeighbors(unittest.TestCase):
    def setUp(self):
        num_nodes = 20
        edges = np.random.randint(num_nodes, size=(100, 2))
        edges = np.unique(edges, axis=0)
        edges_id = np.arange(0, len(edges))
        sorted_edges = edges[np.argsort(edges[:, 1])]
        sorted_edges_id = edges_id[np.argsort(edges[:, 1])]

        # Calculate dst index cumsum counts.
        dst_count = np.zeros(num_nodes)
        dst_src_dict = {}
        for dst in range(0, num_nodes):
            true_index = sorted_edges[:, 1] == dst
            dst_count[dst] = np.sum(true_index)
            dst_src_dict[dst] = sorted_edges[:, 0][true_index]
        dst_count = dst_count.astype("int64")
        dst_cumsum_counts = np.cumsum(dst_count)
        dst_cumsum_counts = np.insert(dst_cumsum_counts, 0, 0)

        self.sorted_src = sorted_edges[:, 0]
        self.dst_cumsum_counts = dst_cumsum_counts
        self.sorted_edges_id = sorted_edges_id
        self.nodes = np.unique(np.random.randint(num_nodes, size=5))
        self.sample_sizes = [5, 5]
        self.dst_src_dict = dst_src_dict

    """
    def test_sample_result(self):
        sorted_src = paddle.to_tensor(self.sorted_src)
        dst_cumsum_counts = paddle.to_tensor(self.dst_cumsum_counts)
        nodes = paddle.to_tensor(self.nodes)
        
        return_eids = False
        empty_eids = paddle.to_tensor([], dtype=sorted_src.dtype)
        edge_src, edge_dst, sample_index, reindex_nodes = \
            paddle.incubate.graph_sample_neighbors(sorted_src, dst_cumsum_counts, 
                                                   nodes, self.sample_sizes)
        # Reindex edge_src and edge_dst to original index.
        edge_src = edge_src.reshape([-1])
        edge_dst = edge_dst.reshape([-1])
        sample_index = sample_index.reshape([-1])

        print(edge_src)
        print(edge_dst)
        print(sample_index)
        print(reindex_nodes)
        for i in range(len(edge_src)):
           edge_src[i] = sample_index[edge_src[i]]
           edge_dst[i] = sample_index[edge_dst[i]]
        print(edge_src)
        print(edge_dst) 
        
        for n in self.nodes:
            edge_src_n = edge_src[edge_dst == n]
            # Ensure no repetitive sample neighbors.
            self.assertTrue(edge_src_n.shape[0] == paddle.unique(edge_src_n).shape[0])
            # Ensure the correct sample size.
            self.assertTrue(edge_src_n.shape[0] == self.sample_sizes[0] or 
                            edge_src_n.shape[0] == len(self.dst_src_dict[n]))
            in_neighbors = np.isin(edge_src_n.numpy(), self.dst_src_dict[n])
            # Ensure the correct sample neighbors.
            self.assertTrue(np.sum(in_neighbors) == in_neighbors.shape[0])
    """

    def test_uva_sample_result(self):
        if paddle.fluid.core.is_compiled_with_cuda():
            sorted_src = to_uva_tensor(self.sorted_src, 0)
            sorted_edges_id = to_uva_tensor(self.sorted_edges_id, 0)
            dst_cumsum_counts = paddle.to_tensor(self.dst_cumsum_counts)
            nodes = paddle.to_tensor(self.nodes)

            return_eids = True
            edge_src, edge_dst, sample_index, reindex_nodes, edge_eids = \
                paddle.incubate.graph_sample_neighbors(sorted_src, dst_cumsum_counts,
                                                       nodes, self.sample_sizes,
                                                       sorted_eids=sorted_edges_id,
                                                       return_eids=True)

            print(edge_src.reshape([-1]))
            print(edge_dst.reshape([-1]))
            print(sample_index)
            print(reindex_nodes)
            print(edge_eids)
            # Reindex edge_src and edge_dst to original index.
            edge_src = edge_src.reshape([-1])
            edge_dst = edge_dst.reshape([-1])
            sample_index = sample_index.reshape([-1])

            for i in range(len(edge_src)):
                edge_src[i] = sample_index[edge_src[i]]
                edge_dst[i] = sample_index[edge_dst[i]]

            print(edge_src)
            print(edge_dst)

            for n in self.nodes:
                edge_src_n = edge_src[edge_dst == n]
                print(edge_src_n)
                # Ensure no repetitive sample neighbors.
                self.assertTrue(
                    edge_src_n.shape[0] == paddle.unique(edge_src_n).shape[0])
                # Ensure the correct sample size.
                self.assertTrue(
                    edge_src_n.shape[0] == self.sample_sizes[0] or
                    edge_src_n.shape[0] == len(self.dst_src_dict[n]))
                in_neighbors = np.isin(edge_src_n.numpy(), self.dst_src_dict[n])
                # Ensure the correct sample neighbors.
                print(edge_src_n.numpy(), self.dst_src_dict[n])
                self.assertTrue(np.sum(in_neighbors) == in_neighbors.shape[0])
            # Ensure correct eids.


if __name__ == "__main__":
    unittest.main()
