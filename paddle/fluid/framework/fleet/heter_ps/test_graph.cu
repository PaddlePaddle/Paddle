/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <gtest/gtest.h>

#include <vector>

#include "paddle/fluid/framework/fleet/heter_ps/feature_value.h"
#include "paddle/fluid/framework/fleet/heter_ps/graph_gpu_ps_table.h"
#include "paddle/fluid/framework/fleet/heter_ps/heter_comm.h"
#include "paddle/fluid/framework/fleet/heter_ps/heter_resource.h"
#include "paddle/fluid/framework/fleet/heter_ps/optimizer.cuh.h"
#include "paddle/phi/core/platform/cuda_device_guard.h"

using paddle::framework;
TEST(TEST_FLEET, graph_comm) {
  int gpu_count = 3;
  std::vector<int> dev_ids;
  dev_ids.push_back(0);
  dev_ids.push_back(1);
  dev_ids.push_back(2);
  std::shared_ptr<HeterPsResource> resource =
      std::make_shared<HeterPsResource>(dev_ids);
  resource->enable_p2p();
  GpuPsGraphTable g(resource);
  int node_count = 10;
  std::vector<std::vector<int64_t>> neighbors(node_count);
  int ind = 0;
  int64_t node_id = 0;
  std::vector<GpuPsCommGraph> graph_list(gpu_count);
  while (ind < node_count) {
    int neighbor_size = ind + 1;
    graph_list[ind % gpu_count].node_size++;
    graph_list[ind % gpu_count].neighbor_size += neighbor_size;
    while (neighbor_size--) {
      neighbors[ind].push_back(node_id++);
    }
    ind++;
  }
  std::vector<int> neighbor_offset(gpu_count, 0), node_index(gpu_count, 0);
  for (int i = 0; i < graph_list.size(); i++) {
    graph_list[i].node_list = new uint64_t[graph_list[i].node_size];
    graph_list[i].node_info_list = new GpuPsNodeInfo[graph_list[i].node_size];
    graph_list[i].neighbor_list = new int64_t[graph_list[i].neighbor_size];
  }
  for (int i = 0; i < node_count; i++) {
    ind = i % gpu_count;
    graph_list[ind].node_list[node_index[ind]] = i;
    graph_list[ind].node_info_list[node_index[ind]].neighbor_offset =
        neighbor_offset[ind];
    graph_list[ind].node_info_list[node_index[ind]].neighbor_size =
        neighbors[i].size();
    for (auto x : neighbors[i]) {
      graph_list[ind].neighbor_list[neighbor_offset[ind]++] = x;
    }
    node_index[ind]++;
  }
  g.build_graph_from_cpu(graph_list);
  /*
  gpu 0:
  0,3,6,9
  gpu 1:
  1,4,7
  gpu 2:
  2,5,8

  query(2,6) returns nodes [6,9,1,4,7,2]
  */
  int64_t answer[6] = {6, 9, 1, 4, 7, 2};
  int64_t *res = new int64_t[6];
  auto query_res = g.query_node_list(0, 2, 6);
  cudaMemcpy(res, query_res->val, 48, cudaMemcpyDeviceToHost);
  ASSERT_EQ(query_res->actual_sample_size, 6);
  for (int i = 0; i < 6; i++) {
    ASSERT_EQ(res[i], answer[i]);
  }
  delete[] res;
  delete query_res;
  /*
   node x's neighbor list = [(1+x)*x/2,(1+x)*x/2 + 1,.....,(1+x)*x/2 + x]
   so node 6's neighbors are [21,22...,27]
   node 7's neighbors are [28,29,..35]
    node 0's neighbors are [0]
   query([7,0,6],sample_size=3) should return [28,29,30,0,x,x,21,22,23]
   6 --index-->2
   0 --index--->0
   7 --index-->2
  */

  int64_t cpu_key[3] = {7, 0, 6};
  void *key;
  cudaMalloc(reinterpret_cast<void **>(&key), 3 * sizeof(int64_t));
  cudaMemcpy(key, cpu_key, 3 * sizeof(int64_t), cudaMemcpyHostToDevice);
  auto neighbor_sample_res =
      g.graph_neighbor_sample(0, reinterpret_cast<int64_t *>(key), 3, 3);
  res = new int64_t[7];
  cudaMemcpy(res, neighbor_sample_res->val, 56, cudaMemcpyDeviceToHost);
  int *actual_sample_size = new int[3];
  cudaMemcpy(actual_sample_size,
             neighbor_sample_res->actual_sample_size,
             12,
             cudaMemcpyDeviceToHost);  // 3, 1, 3
  int *cumsum_sample_size = new int[3];
  cudaMemcpy(cumsum_sample_size,
             neighbor_sample_res->offset,
             12,
             cudaMemcpyDeviceToHost);  // 0, 3, 4

  std::vector<std::vector<int64_t>> neighbors_;
  std::vector<int64_t> neighbors_7 = {28, 29, 30, 31, 32, 33, 34, 35};
  std::vector<int64_t> neighbors_0 = {0};
  std::vector<int64_t> neighbors_6 = {21, 22, 23, 24, 25, 26, 27};
  neighbors_.push_back(neighbors_7);
  neighbors_.push_back(neighbors_0);
  neighbors_.push_back(neighbors_6);
  for (int i = 0; i < 3; i++) {
    for (int j = cumsum_sample_size[i];
         j < cumsum_sample_size[i] + actual_sample_size[i];
         j++) {
      bool flag = false;
      for (int k = 0; k < neighbors_[i].size(); k++) {
        if (res[j] == neighbors_[i][k]) {
          flag = true;
          break;
        }
      }
      ASSERT_EQ(flag, true);
    }
  }

  delete[] res;
  delete[] actual_sample_size;
  delete[] cumsum_sample_size;
  delete neighbor_sample_res;
}
