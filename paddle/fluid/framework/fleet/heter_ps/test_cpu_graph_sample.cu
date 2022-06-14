// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>
#include <algorithm>
#include <vector>
#include "paddle/fluid/framework/fleet/heter_ps/feature_value.h"
#include "paddle/fluid/framework/fleet/heter_ps/graph_gpu_ps_table.h"
#include "paddle/fluid/framework/fleet/heter_ps/heter_comm.h"
#include "paddle/fluid/framework/fleet/heter_ps/heter_resource.h"
#include "paddle/fluid/framework/fleet/heter_ps/optimizer.cuh.h"
#include "paddle/fluid/platform/cuda_device_guard.h"

using namespace paddle::framework;
void prepare_file(char file_name[], std::vector<std::string> data) {
  std::ofstream ofile;
  ofile.open(file_name);
  for (auto x : data) {
    ofile << x << std::endl;
  }

  ofile.close();
}
char edge_file_name[] = "edges.txt";
TEST(TEST_FLEET, graph_sample) {
  std::vector<std::string> edges;
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
  // std::vector<GpuPsCommGraph> graph_list(gpu_count);
  while (ind < node_count) {
    int neighbor_size = ind + 1;
    while (neighbor_size--) {
      edges.push_back(std::to_string(ind) + "\t" + std::to_string(node_id) +
                      "\t1.0");
      node_id++;
    }
    ind++;
  }
  /*
  gpu 0:
  0,3,6,9
  gpu 1:
  1,4,7
  gpu 2:
  2,5,8
  query(2,6) returns nodes [6,9,1,4,7,2]
  */
  ::paddle::distributed::GraphParameter table_proto;
  table_proto.set_gpups_mode(true);
  table_proto.set_shard_num(127);
  table_proto.set_gpu_num(3);
  table_proto.set_gpups_graph_sample_class("BasicBfsGraphSampler");
  table_proto.set_gpups_graph_sample_args("100,5,5,1,1");
  prepare_file(edge_file_name, edges);
  g.init_cpu_table(table_proto);
  g.load(std::string(edge_file_name), std::string("e>"));
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
  cudaMalloc((void **)&key, 3 * sizeof(int64_t));
  cudaMemcpy(key, cpu_key, 3 * sizeof(int64_t), cudaMemcpyHostToDevice);
  auto neighbor_sample_res = g.graph_neighbor_sample(0, (int64_t *)key, 3, 3);
  int64_t *res = new int64_t[7];
  /*
  cudaMemcpy(res, neighbor_sample_res->val, 56, cudaMemcpyDeviceToHost);
  std::sort(res, res + 3);
  std::sort(res + 4, res + 7);
  //int64_t expected_sample_val[] = {28, 29, 30, 0, -1, -1, 21, 22, 23};
  int64_t expected_sample_val[] = {28, 29, 30, 0, 21, 22, 23};
  for (int i = 0; i < 7; i++) {
    VLOG(0)<<i<<" "<<res[i];
    if (expected_sample_val[i] != -1) {
      ASSERT_EQ(res[i], expected_sample_val[i]);
    }
  }
  delete[] res;
  delete neighbor_sample_res;
  */
  cudaMemcpy(res, neighbor_sample_res->val, 56, cudaMemcpyDeviceToHost);
  int *actual_sample_size = new int[3];
  cudaMemcpy(actual_sample_size, neighbor_sample_res->actual_sample_size, 12,
             cudaMemcpyDeviceToHost);  // 3, 1, 3
  int *cumsum_sample_size = new int[3];
  cudaMemcpy(cumsum_sample_size, neighbor_sample_res->offset, 12,
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
         j < cumsum_sample_size[i] + actual_sample_size[i]; j++) {
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
