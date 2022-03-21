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
  table_proto.set_gpups_graph_sample_args("5,5,1,1");
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
  int64_t *res = new int64_t[9];
  cudaMemcpy(res, neighbor_sample_res->val, 72, cudaMemcpyDeviceToHost);
  std::sort(res, res + 3);
  std::sort(res + 6, res + 9);
  int64_t expected_sample_val[] = {28, 29, 30, 0, -1, -1, 21, 22, 23};
  for (int i = 0; i < 9; i++) {
    if (expected_sample_val[i] != -1) {
      ASSERT_EQ(res[i], expected_sample_val[i]);
    }
  }
  delete[] res;
  delete neighbor_sample_res;
}
