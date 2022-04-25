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
namespace platform = paddle::platform;
// paddle::framework::GpuPsCommGraph GraphTable::make_gpu_ps_graph
// paddle::framework::GpuPsCommGraph GraphTable::make_gpu_ps_graph(
//     std::vector<int64_t> ids)

std::string nodes[] = {
    std::string("user\t37\ta 0.34\tb 13 14\tc hello\td abc"),
    std::string("user\t96\ta 0.31\tb 15 10\tc 96hello\td abcd"),
    std::string("user\t59\ta 0.11\tb 11 14"),
    std::string("user\t97\ta 0.11\tb 12 11"),
    std::string("item\t45\ta 0.21"),
    std::string("item\t145\ta 0.21"),
    std::string("item\t112\ta 0.21"),
    std::string("item\t48\ta 0.21"),
    std::string("item\t247\ta 0.21"),
    std::string("item\t111\ta 0.21"),
    std::string("item\t46\ta 0.21"),
    std::string("item\t146\ta 0.21"),
    std::string("item\t122\ta 0.21"),
    std::string("item\t49\ta 0.21"),
    std::string("item\t248\ta 0.21"),
    std::string("item\t113\ta 0.21")};
char node_file_name[] = "nodes.txt";
std::vector<std::string> user_feature_name = {"a", "b", "c", "d"};
std::vector<std::string> item_feature_name = {"a"};
std::vector<std::string> user_feature_dtype = {"float32", "int32", "string",
                                               "string"};
std::vector<std::string> item_feature_dtype = {"float32"};
std::vector<int> user_feature_shape = {1, 2, 1, 1};
std::vector<int> item_feature_shape = {1};
void prepare_file(char file_name[]) {
  std::ofstream ofile;
  ofile.open(file_name);

  for (auto x : nodes) {
    ofile << x << std::endl;
  }
  ofile.close();
}
TEST(TEST_FLEET, test_cpu_cache) {
  int gpu_num = 0;
  int st = 0, u = 0;
  std::vector<int> device_id_mapping;
  for (int i = 0; i < 2; i++) device_id_mapping.push_back(i);
  gpu_num = device_id_mapping.size();
  ::paddle::distributed::GraphParameter table_proto;
  table_proto.add_edge_types("u2u");
  table_proto.add_node_types("user");
  table_proto.add_node_types("item");
  ::paddle::distributed::GraphFeature *g_f = table_proto.add_graph_feature();

  for (int i = 0; i < user_feature_name.size(); i++) {
    g_f->add_name(user_feature_name[i]);
    g_f->add_dtype(user_feature_dtype[i]);
    g_f->add_shape(user_feature_shape[i]);
  }
  ::paddle::distributed::GraphFeature *g_f1 = table_proto.add_graph_feature();
  for (int i = 0; i < item_feature_name.size(); i++) {
    g_f1->add_name(item_feature_name[i]);
    g_f1->add_dtype(item_feature_dtype[i]);
    g_f1->add_shape(item_feature_shape[i]);
  }
  prepare_file(node_file_name);
  table_proto.set_shard_num(24);

  std::shared_ptr<HeterPsResource> resource =
      std::make_shared<HeterPsResource>(device_id_mapping);
  resource->enable_p2p();
  int use_nv = 1;
  GpuPsGraphTable g(resource, use_nv);
  g.init_cpu_table(table_proto);
  g.cpu_graph_table->Load(node_file_name, "nuser");
  g.cpu_graph_table->Load(node_file_name, "nitem");
  std::remove(node_file_name);
  std::vector<paddle::framework::GpuPsCommGraph> vec;
  std::vector<int64_t> node_ids;
  node_ids.push_back(37);
  node_ids.push_back(96);
  std::vector<std::vector<std::string>> node_feat(2,
                                                  std::vector<std::string>(2));
  std::vector<std::string> feature_names;
  feature_names.push_back(std::string("c"));
  feature_names.push_back(std::string("d"));
  g.cpu_graph_table->get_node_feat(0, node_ids, feature_names, node_feat);
  VLOG(0) << "get_node_feat: " << node_feat[0][0];
  VLOG(0) << "get_node_feat: " << node_feat[0][1];
  VLOG(0) << "get_node_feat: " << node_feat[1][0];
  VLOG(0) << "get_node_feat: " << node_feat[1][1];
  int n = 10;
  std::vector<int64_t> ids0, ids1;
  for (int i = 0; i < n; i++) {
    g.cpu_graph_table->add_comm_edge(0, i, (i + 1) % n);
    g.cpu_graph_table->add_comm_edge(0, i, (i - 1 + n) % n);
    if (i % 2 == 0) ids0.push_back(i);
  }
  g.cpu_graph_table->build_sampler(0);
  ids1.push_back(5);
  vec.push_back(g.cpu_graph_table->make_gpu_ps_graph(0, ids0));
  vec.push_back(g.cpu_graph_table->make_gpu_ps_graph(0, ids1));
  vec[0].display_on_cpu();
  vec[1].display_on_cpu();
  g.build_graph_from_cpu(vec);
  int64_t cpu_key[3] = {0, 1, 2};
  /*
  std::vector<std::shared_ptr<char>> buffers(3);
  std::vector<int> actual_sizes(3,0);
  g.cpu_graph_table->random_sample_neighbors(cpu_key,2,buffers,actual_sizes,false);
  for(int i = 0;i < 3;i++){
    VLOG(0)<<"sample from cpu key->"<<cpu_key[i]<<" actual sample size =
  "<<actual_sizes[i]/sizeof(int64_t);
  }
  */
  void *key;
  platform::CUDADeviceGuard guard(0);
  cudaMalloc((void **)&key, 3 * sizeof(int64_t));
  cudaMemcpy(key, cpu_key, 3 * sizeof(int64_t), cudaMemcpyHostToDevice);
  auto neighbor_sample_res = g.graph_neighbor_sample(0, (int64_t *)key, 2, 3);
  int64_t *res = new int64_t[7];
  cudaMemcpy(res, neighbor_sample_res->val, 3 * 2 * sizeof(int64_t),
             cudaMemcpyDeviceToHost);
  int *actual_sample_size = new int[3];
  cudaMemcpy(actual_sample_size, neighbor_sample_res->actual_sample_size,
             3 * sizeof(int),
             cudaMemcpyDeviceToHost);  // 3, 1, 3

  //{0,9} or {9,0} is expected for key 0
  //{0,2} or {2,0} is expected for key 1
  //{1,3} or {3,1} is expected for key 2
  for (int i = 0; i < 3; i++) {
    VLOG(0) << "actual sample size for " << i << " is "
            << actual_sample_size[i];
    for (int j = 0; j < actual_sample_size[i]; j++) {
      VLOG(0) << "sampled an neighbor for node" << i << " : " << res[i * 2 + j];
    }
  }
}
