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
#include "paddle/fluid/framework/fleet/heter_ps/graph_gpu_wrapper.h"
#include "paddle/fluid/framework/fleet/heter_ps/heter_comm.h"
#include "paddle/fluid/framework/fleet/heter_ps/heter_resource.h"
#include "paddle/fluid/framework/fleet/heter_ps/optimizer.cuh.h"
#include "paddle/phi/core/platform/cuda_device_guard.h"

using namespace paddle::framework;  // NOLINT
namespace platform = paddle::platform;

const char *edges[] = {
    "0\t1",
    "0\t9",
    "1\t2",
    "1\t0",
    "2\t1",
    "2\t3",
    "3\t2",
    "3\t4",
    "4\t3",
    "4\t5",
    "5\t4",
    "5\t6",
    "6\t5",
    "6\t7",
    "7\t6",
    "7\t8",
};
char edge_file_name[] = "edges1.txt";

const char *nodes[] = {"user\t37\ta 0.34\tb 13 14\tc hello\td abc",
                       "user\t96\ta 0.31\tb 15 10\tc 96hello\td abcd",
                       "user\t59\ta 0.11\tb 11 14",
                       "user\t97\ta 0.11\tb 12 11",
                       "item\t45\ta 0.21",
                       "item\t145\ta 0.21",
                       "item\t112\ta 0.21",
                       "item\t48\ta 0.21",
                       "item\t247\ta 0.21",
                       "item\t111\ta 0.21",
                       "item\t46\ta 0.21",
                       "item\t146\ta 0.21",
                       "item\t122\ta 0.21",
                       "item\t49\ta 0.21",
                       "item\t248\ta 0.21",
                       "item\t113\ta 0.21"};
char node_file_name[] = "nodes.txt";
std::vector<std::string> user_feature_name = {"a", "b", "c", "d"};
std::vector<std::string> item_feature_name = {"a"};
std::vector<std::string> user_feature_dtype = {
    "float32", "int32", "string", "string"};
std::vector<std::string> item_feature_dtype = {"float32"};
std::vector<int> user_feature_shape = {1, 2, 1, 1};
std::vector<int> item_feature_shape = {1};
void prepare_file(char file_name[], bool load_edge) {
  std::ofstream ofile;
  ofile.open(file_name);
  if (load_edge) {
    for (auto x : edges) {
      ofile << x << std::endl;
    }
  } else {
    for (auto x : nodes) {
      ofile << x << std::endl;
    }
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
  prepare_file(node_file_name, false);
  prepare_file(edge_file_name, true);
  table_proto.set_shard_num(24);
  table_proto.set_search_level(2);
  std::shared_ptr<HeterPsResource> resource =
      std::make_shared<HeterPsResource>(device_id_mapping);
  resource->enable_p2p();
  int use_nv = 1;
  GpuPsGraphTable g(resource, 2);
  g.init_cpu_table(table_proto);
  g.cpu_graph_table_->Load(node_file_name, "nuser");
  g.cpu_graph_table_->Load(node_file_name, "nitem");
  std::remove(node_file_name);
  std::vector<paddle::framework::GpuPsCommGraph> vec;
  std::vector<uint64_t> node_ids;
  node_ids.push_back(37);
  node_ids.push_back(96);
  std::vector<std::vector<std::string>> node_feat(2,
                                                  std::vector<std::string>(2));
  std::vector<std::string> feature_names;
  feature_names.push_back(std::string("c"));
  feature_names.push_back(std::string("d"));
  g.cpu_graph_table_->get_node_feat(0, node_ids, feature_names, node_feat);
  VLOG(0) << "get_node_feat: " << node_feat[0][0];
  VLOG(0) << "get_node_feat: " << node_feat[0][1];
  VLOG(0) << "get_node_feat: " << node_feat[1][0];
  VLOG(0) << "get_node_feat: " << node_feat[1][1];
  int n = 10;
  std::vector<uint64_t> ids0, ids1;
  for (int i = 0; i < n; i++) {
    g.cpu_graph_table_->add_comm_edge(0, i, (i + 1) % n);
    g.cpu_graph_table_->add_comm_edge(0, i, (i - 1 + n) % n);
    if (i % 2 == 0) ids0.push_back(i);
  }
  g.cpu_graph_table_->build_sampler(0);
  ids1.push_back(5);
  ids1.push_back(7);
  vec.push_back(g.cpu_graph_table_->make_gpu_ps_graph(0, ids0));
  vec.push_back(g.cpu_graph_table_->make_gpu_ps_graph(0, ids1));
  vec[0].display_on_cpu();
  vec[1].display_on_cpu();
  // g.build_graph_from_cpu(vec);
  g.build_graph_on_single_gpu(vec[0], 0, 0);
  g.build_graph_on_single_gpu(vec[1], 1, 0);
  uint64_t cpu_key[3] = {0, 1, 2};
  void *key;
  int device_len = 2;
  for (int i = 0; i < 2; i++) {
    // platform::CUDADeviceGuard guard(i);
    LOG(0) << "query on card " << i;
    // {1,9} or {9,1} is expected for key 0
    // {0,2} or {2,0} is expected for key 1
    // {1,3} or {3,1} is expected for key 2
    int step = 2;
    int cur = 0;
    while (true) {
      auto node_query_res = g.query_node_list(i, 0, cur, step);
      node_query_res.display();
      if (node_query_res.get_len() == 0) {
        VLOG(0) << "no more ids,break";
        break;
      }
      cur += node_query_res.get_len();
      NeighborSampleQuery query;
      query.initialize(
          i, 0, node_query_res.get_val(), 1, node_query_res.get_len(), 50, 1);
      query.display();
      auto c = g.graph_neighbor_sample_v3(query, false, true, false);
      c.display();
    }
  }
  g.cpu_graph_table_->clear_graph(0);
  g.cpu_graph_table_->set_search_level(2);
  g.cpu_graph_table_->Load(edge_file_name, "e>u2u");
  g.cpu_graph_table_->make_partitions(0, 64, 2);
  int index = 0;
  /*
  while (g.cpu_graph_table_->load_next_partition(0) != -1) {
    auto all_ids = g.cpu_graph_table_->get_all_id(0, 0, device_len);
    for (auto x : all_ids) {
      for (auto y : x) {
        VLOG(0) << "part " << index << " " << y;
      }
    }
    for (int i = 0; i < all_ids.size(); i++) {
      GpuPsCommGraph sub_graph =
          g.cpu_graph_table_->make_gpu_ps_graph(0, all_ids[i]);
      g.build_graph_on_single_gpu(sub_graph, i, 0);
      VLOG(2) << "sub graph on gpu " << i << " is built";
    }
    VLOG(0) << "start to iterate gpu graph node";
    g.cpu_graph_table_->make_complementary_graph(0, 64);
    for (int i = 0; i < 2; i++) {
      // platform::CUDADeviceGuard guard(i);
      LOG(0) << "query on card " << i;
      int step = 2;
      int cur = 0;
      while (true) {
        auto node_query_res = g.query_node_list(i, 0, cur, step);
        node_query_res.display();
        if (node_query_res.get_len() == 0) {
          VLOG(0) << "no more ids,break";
          break;
        }
        cur += node_query_res.get_len();
        NeighborSampleQuery query, q1;
        query.initialize(i, 0, node_query_res.get_val(), 4,
                         node_query_res.get_len());
        query.display();
        auto c = g.graph_neighbor_sample_v3(query, true, true);
        c.display();
        platform::CUDADeviceGuard guard(i);
        uint64_t *key;
        VLOG(0) << "sample key 1 globally";
        g.cpu_graph_table_->set_search_level(2);
        cudaMalloc((void **)&key, sizeof(uint64_t));
        uint64_t t_key = 1;
        cudaMemcpy(key, &t_key, sizeof(uint64_t), cudaMemcpyHostToDevice);
        q1.initialize(i, 0, (uint64_t)key, 2, 1);
        auto d = g.graph_neighbor_sample_v3(q1, true, true);
        d.display();
        cudaFree(key);
        g.cpu_graph_table_->set_search_level(1);
      }
    }
    index++;
  }
  auto iter = paddle::framework::GraphGpuWrapper::GetInstance();
  std::vector<int> device;
  device.push_back(0);
  device.push_back(1);
  iter->set_device(device);
  */
}
