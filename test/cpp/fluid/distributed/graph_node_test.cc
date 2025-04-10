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

#include "paddle/fluid/distributed/ps/table/graph/graph_node.h"

#include <unistd.h>

#include <condition_variable>  // NOLINT
#include <fstream>
#include <iomanip>
#include <string>
#include <thread>  // NOLINT
#include <unordered_set>
#include <vector>

#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"
#include "paddle/fluid/distributed/ps/service/brpc_ps_client.h"
#include "paddle/fluid/distributed/ps/service/brpc_ps_server.h"
#include "paddle/fluid/distributed/ps/service/env.h"
#include "paddle/fluid/distributed/ps/service/graph_brpc_client.h"
#include "paddle/fluid/distributed/ps/service/graph_brpc_server.h"
#include "paddle/fluid/distributed/ps/service/ps_client.h"
#include "paddle/fluid/distributed/ps/service/ps_service/graph_py_service.h"
#include "paddle/fluid/distributed/ps/service/ps_service/service.h"
#include "paddle/fluid/distributed/ps/service/sendrecv.pb.h"
#include "paddle/fluid/distributed/the_one_ps.pb.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/utils/string/printf.h"

namespace framework = paddle::framework;
namespace distributed = paddle::distributed;

// void testSampleNodes(
//     std::shared_ptr<paddle::distributed::GraphBrpcClient>& worker_ptr_) {
//   std::vector<int64_t> ids;
//   auto pull_status = worker_ptr_->random_sample_nodes(0, 0, 6, ids);
//   std::unordered_set<int64_t> s;
//   std::unordered_set<int64_t> s1 = {37, 59};
//   pull_status.wait();
//   for (auto id : ids) s.insert(id);
//   ASSERT_EQ(true, s.size() == s1.size());
//   for (auto id : s) {
//     ASSERT_EQ(true, s1.find(id) != s1.end());
//   }
// }

void testFeatureNodeSerializeInt() {
  std::string out =
      distributed::FeatureNode::parse_value_to_bytes<int32_t>({"123", "345"});
  std::vector<int32_t> out2 =
      distributed::FeatureNode::parse_bytes_to_array<int32_t>(out);
  ASSERT_EQ(out2[0], 123);
  ASSERT_EQ(out2[1], 345);
}

void testFeatureNodeSerializeInt64() {
  std::string out =
      distributed::FeatureNode::parse_value_to_bytes<int64_t>({"123", "345"});
  std::vector<int64_t> out2 =
      distributed::FeatureNode::parse_bytes_to_array<int64_t>(out);
  ASSERT_EQ(out2[0], 123);
  ASSERT_EQ(out2[1], 345);
}

void testFeatureNodeSerializeFloat32() {
  std::string out = distributed::FeatureNode::parse_value_to_bytes<float>(
      {"123.123", "345.123"});
  std::vector<float> out2 =
      distributed::FeatureNode::parse_bytes_to_array<float>(out);
  float eps;
  std::cout << "Float " << out2[0] << " " << 123.123 << std::endl;
  eps = out2[0] - 123.123;
  ASSERT_LE(eps * eps, 1e-5);
  eps = out2[1] - 345.123;
  ASSERT_LE(eps * eps, 1e-5);
}

void testFeatureNodeSerializeFloat64() {
  std::string out = distributed::FeatureNode::parse_value_to_bytes<double>(
      {"123.123", "345.123"});
  std::vector<double> out2 =
      distributed::FeatureNode::parse_bytes_to_array<double>(out);
  float eps;
  eps = out2[0] - 123.123;
  std::cout << "Float64 " << out2[0] << " " << 123.123 << std::endl;
  ASSERT_LE(eps * eps, 1e-5);
  eps = out2[1] - 345.123;
  ASSERT_LE(eps * eps, 1e-5);
}

// void testSingleSampleNeighbour(
//     std::shared_ptr<paddle::distributed::GraphBrpcClient>& worker_ptr_) {
//   std::vector<std::vector<int64_t>> vs;
//   std::vector<std::vector<float>> vs1;
//   auto pull_status = worker_ptr_->batch_sample_neighbors(
//       0, std::vector<int64_t>(1, 37), 4, vs, vs1, true);
//   pull_status.wait();

//   std::unordered_set<int64_t> s;
//   std::unordered_set<int64_t> s1 = {112, 45, 145};
//   for (auto g : vs[0]) {
//     s.insert(g);
//   }
//   ASSERT_EQ(s.size(), 3);
//   for (auto g : s) {
//     ASSERT_EQ(true, s1.find(g) != s1.end());
//   }
//   s.clear();
//   s1.clear();
//   vs.clear();
//   vs1.clear();
//   pull_status = worker_ptr_->batch_sample_neighbors(
//       0, std::vector<int64_t>(1, 96), 4, vs, vs1, true);
//   pull_status.wait();
//   s1 = {111, 48, 247};
//   for (auto g : vs[0]) {
//     s.insert(g);
//   }
//   ASSERT_EQ(s.size(), 3);
//   for (auto g : s) {
//     ASSERT_EQ(true, s1.find(g) != s1.end());
//   }
//   vs.clear();
//   pull_status =
//       worker_ptr_->batch_sample_neighbors(0, {96, 37}, 4, vs, vs1, true, 0);
//   pull_status.wait();
//   ASSERT_EQ(vs.size(), 2);
// }

// void testAddNode(
//     std::shared_ptr<paddle::distributed::GraphBrpcClient>& worker_ptr_) {
//   worker_ptr_->clear_nodes(0);
//   int total_num = 270000;
//   int64_t id;
//   std::unordered_set<int64_t> id_set;
//   for (int i = 0; i < total_num; i++) {
//     while (id_set.find(id = rand()) != id_set.end())
//       ;
//     id_set.insert(id);
//   }
//   std::vector<int64_t> id_list(id_set.begin(), id_set.end());
//   std::vector<bool> weight_list;
//   auto status = worker_ptr_->add_graph_node(0, id_list, weight_list);
//   status.wait();
//   std::vector<int64_t> ids[2];
//   for (int i = 0; i < 2; i++) {
//     auto sample_status =
//         worker_ptr_->random_sample_nodes(0, i, total_num, ids[i]);
//     sample_status.wait();
//   }
//   std::unordered_set<int64_t> id_set_check(ids[0].begin(), ids[0].end());
//   for (auto x : ids[1]) id_set_check.insert(x);
//   ASSERT_EQ(id_set.size(), id_set_check.size());
//   for (auto x : id_set) {
//     ASSERT_EQ(id_set_check.find(x) != id_set_check.end(), true);
//   }
//   std::vector<int64_t> remove_ids;
//   for (auto p : id_set_check) {
//     if (remove_ids.size() == 0)
//       remove_ids.push_back(p);
//     else if (remove_ids.size() < total_num / 2 && rand() % 2 == 1) {
//       remove_ids.push_back(p);
//     }
//   }
//   for (auto p : remove_ids) id_set_check.erase(p);
//   status = worker_ptr_->remove_graph_node(0, remove_ids);
//   status.wait();
//   for (int i = 0; i < 2; i++) ids[i].clear();
//   for (int i = 0; i < 2; i++) {
//     auto sample_status =
//         worker_ptr_->random_sample_nodes(0, i, total_num, ids[i]);
//     sample_status.wait();
//   }
//   std::unordered_set<int64_t> id_set_check1(ids[0].begin(), ids[0].end());
//   for (auto x : ids[1]) id_set_check1.insert(x);
//   ASSERT_EQ(id_set_check1.size(), id_set_check.size());
//   for (auto x : id_set_check1) {
//     ASSERT_EQ(id_set_check.find(x) != id_set_check.end(), true);
//   }
// }
// void testBatchSampleNeighbor(
//     std::shared_ptr<paddle::distributed::GraphBrpcClient>& worker_ptr_) {
//   std::vector<std::vector<int64_t>> vs;
//   std::vector<std::vector<float>> vs1;
//   std::vector<std::int64_t> v = {37, 96};
//   auto pull_status =
//       worker_ptr_->batch_sample_neighbors(0, v, 4, vs, vs1, false);
//   pull_status.wait();
//   std::unordered_set<int64_t> s;
//   std::unordered_set<int64_t> s1 = {112, 45, 145};
//   for (auto g : vs[0]) {
//     s.insert(g);
//   }
//   ASSERT_EQ(s.size(), 3);
//   for (auto g : s) {
//     ASSERT_EQ(true, s1.find(g) != s1.end());
//   }
//   s.clear();
//   s1.clear();
//   s1 = {111, 48, 247};
//   for (auto g : vs[1]) {
//     s.insert(g);
//   }
//   ASSERT_EQ(s.size(), 3);
//   for (auto g : s) {
//     ASSERT_EQ(true, s1.find(g) != s1.end());
//   }
// }

// void testCache();
void testGraphToBuffer();

const char* edges[] = {"37\t45\t0.34",  // NOLINT
                       "37\t145\t0.31",
                       "37\t112\t0.21",
                       "96\t48\t1.4",
                       "96\t247\t0.31",
                       "96\t111\t1.21",
                       "59\t45\t0.34",
                       "59\t145\t0.31",
                       "59\t122\t0.21",
                       "97\t48\t0.34",
                       "97\t247\t0.31",
                       "97\t111\t0.21"};  // NOLINT
char edge_file_name[] = "edges.txt";      // NOLINT

const char* nodes[] = {"user\t37\ta 0.34\tb 13 14\tc hello\td abc",  // NOLINT
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
                       "item\t113\ta 0.21"};  // NOLINT
char node_file_name[] = "nodes.txt";          // NOLINT

void prepare_file(char file_name[], bool load_edge) {  // NOLINT
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
void GetDownpourSparseTableProto(
    ::paddle::distributed::TableParameter* sparse_table_proto) {
  sparse_table_proto->set_table_id(0);
  sparse_table_proto->set_table_class("GraphTable");
  sparse_table_proto->set_shard_num(127);
  sparse_table_proto->set_type(::paddle::distributed::PS_SPARSE_TABLE);
  ::paddle::distributed::TableAccessorParameter* accessor_proto =
      sparse_table_proto->mutable_accessor();
  accessor_proto->set_accessor_class("CommMergeAccessor");
}

::paddle::distributed::PSParameter GetServerProto() {
  // Generate server proto desc
  ::paddle::distributed::PSParameter server_fleet_desc;
  ::paddle::distributed::ServerParameter* server_proto =
      server_fleet_desc.mutable_server_param();
  ::paddle::distributed::DownpourServerParameter* downpour_server_proto =
      server_proto->mutable_downpour_server_param();
  ::paddle::distributed::ServerServiceParameter* server_service_proto =
      downpour_server_proto->mutable_service_param();
  server_service_proto->set_service_class("GraphBrpcService");
  server_service_proto->set_server_class("GraphBrpcServer");
  server_service_proto->set_client_class("GraphBrpcClient");
  server_service_proto->set_start_server_port(0);
  server_service_proto->set_server_thread_num(12);

  ::paddle::distributed::TableParameter* sparse_table_proto =
      downpour_server_proto->add_downpour_table_param();
  GetDownpourSparseTableProto(sparse_table_proto);
  return server_fleet_desc;
}

::paddle::distributed::PSParameter GetWorkerProto() {
  ::paddle::distributed::PSParameter worker_fleet_desc;
  ::paddle::distributed::WorkerParameter* worker_proto =
      worker_fleet_desc.mutable_worker_param();

  ::paddle::distributed::DownpourWorkerParameter* downpour_worker_proto =
      worker_proto->mutable_downpour_worker_param();

  ::paddle::distributed::TableParameter* worker_sparse_table_proto =
      downpour_worker_proto->add_downpour_table_param();
  GetDownpourSparseTableProto(worker_sparse_table_proto);

  ::paddle::distributed::ServerParameter* server_proto =
      worker_fleet_desc.mutable_server_param();
  ::paddle::distributed::DownpourServerParameter* downpour_server_proto =
      server_proto->mutable_downpour_server_param();
  ::paddle::distributed::ServerServiceParameter* server_service_proto =
      downpour_server_proto->mutable_service_param();
  server_service_proto->set_service_class("GraphBrpcService");
  server_service_proto->set_server_class("GraphBrpcServer");
  server_service_proto->set_client_class("GraphBrpcClient");
  server_service_proto->set_start_server_port(0);
  server_service_proto->set_server_thread_num(12);

  ::paddle::distributed::TableParameter* server_sparse_table_proto =
      downpour_server_proto->add_downpour_table_param();
  GetDownpourSparseTableProto(server_sparse_table_proto);

  return worker_fleet_desc;
}

/*-------------------------------------------------------------------------*/

const char* ip_ = "127.0.0.1";
const char* ip2 = "127.0.0.1";
uint32_t port_ = 5209, port2 = 5210;

std::vector<std::string> host_sign_list_;

std::shared_ptr<paddle::distributed::GraphBrpcServer> pserver_ptr_,
    pserver_ptr2;

std::shared_ptr<paddle::distributed::GraphBrpcClient> worker_ptr_;

void RunServer() {
  LOG(INFO) << "init first server";
  ::paddle::distributed::PSParameter server_proto = GetServerProto();

  auto _ps_env = paddle::distributed::PaddlePSEnvironment();
  _ps_env.SetPsServers(&host_sign_list_, 2);  // test
  pserver_ptr_ = std::shared_ptr<paddle::distributed::GraphBrpcServer>(
      (paddle::distributed::GraphBrpcServer*)
          paddle::distributed::PSServerFactory::Create(server_proto));
  std::vector<framework::ProgramDesc> empty_vec;
  framework::ProgramDesc empty_prog;
  empty_vec.push_back(empty_prog);
  pserver_ptr_->Configure(server_proto, _ps_env, 0, empty_vec);
  LOG(INFO) << "first server, run start(ip,port)";
  pserver_ptr_->Start(ip_, port_);
  pserver_ptr_->build_peer2peer_connection(0);
  LOG(INFO) << "init first server Done";
}

void RunServer2() {
  LOG(INFO) << "init second server";
  ::paddle::distributed::PSParameter server_proto2 = GetServerProto();

  auto _ps_env2 = paddle::distributed::PaddlePSEnvironment();
  _ps_env2.SetPsServers(&host_sign_list_, 2);  // test
  pserver_ptr2 = std::shared_ptr<paddle::distributed::GraphBrpcServer>(
      (paddle::distributed::GraphBrpcServer*)
          paddle::distributed::PSServerFactory::Create(server_proto2));
  std::vector<framework::ProgramDesc> empty_vec2;
  framework::ProgramDesc empty_prog2;
  empty_vec2.push_back(empty_prog2);
  pserver_ptr2->Configure(server_proto2, _ps_env2, 1, empty_vec2);
  pserver_ptr2->Start(ip2, port2);
  pserver_ptr2->build_peer2peer_connection(1);
}

void RunClient(
    const std::map<uint64_t, std::vector<paddle::distributed::Region>>&
        dense_regions,
    int index,
    paddle::distributed::PsBaseService* service) {
  ::paddle::distributed::PSParameter worker_proto = GetWorkerProto();
  paddle::distributed::PaddlePSEnvironment _ps_env;
  auto servers_ = host_sign_list_.size();
  _ps_env = paddle::distributed::PaddlePSEnvironment();
  _ps_env.SetPsServers(&host_sign_list_, servers_);
  worker_ptr_ = std::shared_ptr<paddle::distributed::GraphBrpcClient>(
      (paddle::distributed::GraphBrpcClient*)
          paddle::distributed::PSClientFactory::Create(worker_proto));
  worker_ptr_->Configure(worker_proto, dense_regions, _ps_env, 0);
  worker_ptr_->set_shard_num(127);
  worker_ptr_->set_local_channel(index);
  worker_ptr_->set_local_graph_service(
      (paddle::distributed::GraphBrpcService*)service);
}

void RunBrpcPushSparse() {
  // testCache();
  setenv("http_proxy", "", 1);
  setenv("https_proxy", "", 1);
  prepare_file(edge_file_name, true);
  prepare_file(node_file_name, false);
  // auto ph_host = paddle::distributed::PSHost(ip_, port_, 0);
  // host_sign_list_.push_back(ph_host.SerializeToString());

  // // test-start
  // auto ph_host2 = paddle::distributed::PSHost(ip2, port2, 1);
  // host_sign_list_.push_back(ph_host2.SerializeToString());
  // // test-end
  // // Start Server
  // std::thread* server_thread = new std::thread(RunServer);
  // std::thread* server_thread2 = new std::thread(RunServer2);
  // sleep(1);

  // std::map<uint64_t, std::vector<paddle::distributed::Region>> dense_regions;
  // dense_regions.insert(
  //     std::pair<int64_t, std::vector<paddle::distributed::Region>>(0, {}));
  // auto regions = dense_regions[0];

  // RunClient(dense_regions, 0, pserver_ptr_->get_service());

  // /*-----------------------Test Server
  // Init----------------------------------*/
  // auto pull_status =
  //     worker_ptr_->Load(0, std::string(edge_file_name), std::string("e>"));
  // srand(time(0));
  // pull_status.wait();
  // std::vector<std::vector<int64_t>> _vs;
  // std::vector<std::vector<float>> vs;
  // testSampleNodes(worker_ptr_);
  // sleep(5);
  // testSingleSampleNeighbor(worker_ptr_);
  // testBatchSampleNeighbor(worker_ptr_);
  // pull_status = worker_ptr_->batch_sample_neighbors(
  //     0, std::vector<int64_t>(1, 10240001024), 4, _vs, vs, true);
  // pull_status.wait();
  // ASSERT_EQ(0, _vs[0].size());
  // paddle::distributed::GraphTable* g =
  //     (paddle::distributed::GraphTable*)pserver_ptr_->GetTable(0);
  // size_t ttl = 6;
  // g->make_neighbor_sample_cache(4, ttl);
  // int round = 5;
  // while (round--) {
  //   vs.clear();
  //   pull_status = worker_ptr_->batch_sample_neighbors(
  //       0, std::vector<int64_t>(1, 37), 1, _vs, vs, false);
  //   pull_status.wait();

  //   for (int i = 0; i < ttl; i++) {
  //     std::vector<std::vector<int64_t>> vs1;
  //     std::vector<std::vector<float>> vs2;
  //     pull_status = worker_ptr_->batch_sample_neighbors(
  //         0, std::vector<int64_t>(1, 37), 1, vs1, vs2, false);
  //     pull_status.wait();
  //     ASSERT_EQ(_vs[0].size(), vs1[0].size());

  //     for (size_t j = 0; j < _vs[0].size(); j++) {
  //       ASSERT_EQ(_vs[0][j], vs1[0][j]);
  //     }
  //   }
  // }

  std::vector<distributed::FeatureNode> nodes;
  // pull_status = worker_ptr_->pull_graph_list(0, 0, 0, 1, 1, nodes);
  // pull_status.wait();
  // ASSERT_EQ(nodes.size(), 1);
  // ASSERT_EQ(nodes[0].get_id(), 37);
  // nodes.clear();
  // pull_status = worker_ptr_->pull_graph_list(0, 0, 1, 4, 1, nodes);
  // pull_status.wait();
  // ASSERT_EQ(nodes.size(), 1);
  // ASSERT_EQ(nodes[0].get_id(), 59);
  // for (auto g : nodes) {
  //   std::cout << g.get_id() << std::endl;
  // }
  distributed::GraphPyServer server1, server2;
  distributed::GraphPyClient client1, client2;
  std::string ips_str = "127.0.0.1:5217;127.0.0.1:5218";
  std::vector<std::string> edge_types = {std::string("user2item")};
  std::vector<std::string> node_types = {std::string("user"),
                                         std::string("item")};
  VLOG(0) << "make 2 servers";
  server1.set_up(ips_str, 127, node_types, edge_types, 0);
  server2.set_up(ips_str, 127, node_types, edge_types, 1);
  VLOG(0) << "make 2 servers done";
  server1.add_table_feat_conf("user", "a", "float32", 1);
  server1.add_table_feat_conf("user", "b", "int32", 2);
  server1.add_table_feat_conf("user", "c", "string", 1);
  server1.add_table_feat_conf("user", "d", "string", 1);
  server1.add_table_feat_conf("item", "a", "float32", 1);

  server2.add_table_feat_conf("user", "a", "float32", 1);
  server2.add_table_feat_conf("user", "b", "int32", 2);
  server2.add_table_feat_conf("user", "c", "string", 1);
  server2.add_table_feat_conf("user", "d", "string", 1);
  server2.add_table_feat_conf("item", "a", "float32", 1);
  VLOG(0) << "add conf 1 done";
  client1.set_up(ips_str, 127, node_types, edge_types, 0);

  client1.add_table_feat_conf("user", "a", "float32", 1);
  client1.add_table_feat_conf("user", "b", "int32", 2);
  client1.add_table_feat_conf("user", "c", "string", 1);
  client1.add_table_feat_conf("user", "d", "string", 1);
  client1.add_table_feat_conf("item", "a", "float32", 1);

  client2.set_up(ips_str, 127, node_types, edge_types, 1);

  client2.add_table_feat_conf("user", "a", "float32", 1);
  client2.add_table_feat_conf("user", "b", "int32", 2);
  client2.add_table_feat_conf("user", "c", "string", 1);
  client2.add_table_feat_conf("user", "d", "string", 1);
  client2.add_table_feat_conf("item", "a", "float32", 1);

  VLOG(0) << "add conf 2 done";
  server1.start_server(false);
  std::cout << "first server done" << std::endl;
  server2.start_server(false);
  std::cout << "second server done" << std::endl;
  client1.start_client();
  std::cout << "first client done" << std::endl;
  client2.start_client();
  std::cout << "first client done" << std::endl;
  std::cout << "started" << std::endl;
  VLOG(0) << "come to set local server";
  client1.bind_local_server(0, server1);
  VLOG(0) << "first bound";
  client2.bind_local_server(1, server2);
  VLOG(0) << "second bound";
  client1.load_node_file(std::string("user"), std::string(node_file_name));
  client1.load_node_file(std::string("item"), std::string(node_file_name));
  client1.load_edge_file(
      std::string("user2item"), std::string(edge_file_name), false);
  nodes.clear();
  VLOG(0) << "start to pull graph list";
  nodes = client1.pull_graph_list(std::string("user"), 0, 1, 4, 1);
  VLOG(0) << "pull list done";
  ASSERT_EQ(nodes[0].get_id(), 59UL);
  nodes.clear();

  // Test Pull by step

  std::unordered_set<int64_t> count_item_nodes;
  // pull by step 2
  for (int test_step = 1; test_step < 4; test_step++) {
    count_item_nodes.clear();
    std::cout << "check pull graph list by step " << test_step << std::endl;
    for (int server_id = 0; server_id < 2; server_id++) {
      for (int start_step = 0; start_step < test_step; start_step++) {
        nodes = client1.pull_graph_list(
            std::string("item"), server_id, start_step, 12, test_step);
        for (auto g : nodes) {
          count_item_nodes.insert(g.get_id());
        }
        nodes.clear();
      }
    }
    ASSERT_EQ(count_item_nodes.size(), 12UL);
  }

  std::pair<std::vector<std::vector<int64_t>>, std::vector<float>> res;
  VLOG(0) << "start to sample neighbors ";
  res = client1.batch_sample_neighbors(
      std::string("user2item"), std::vector<int64_t>(1, 96), 4, true, false);
  ASSERT_EQ(res.first[0].size(), 3UL);
  std::vector<int64_t> node_ids;
  node_ids.push_back(96);
  node_ids.push_back(37);
  res = client1.batch_sample_neighbors(
      std::string("user2item"), node_ids, 4, true, false);

  ASSERT_EQ(res.first[1].size(), 1UL);
  std::vector<int64_t> nodes_ids = client2.random_sample_nodes("user", 0, 6);
  ASSERT_EQ(nodes_ids.size(), 2UL);
  ASSERT_EQ(true,
            (nodes_ids[0] == 59 && nodes_ids[1] == 37) ||
                (nodes_ids[0] == 37 && nodes_ids[1] == 59));

  VLOG(0) << "start to test get node feat";
  // Test get node feat
  node_ids.clear();
  node_ids.push_back(37);
  node_ids.push_back(96);
  std::vector<std::string> feature_names;
  feature_names.push_back(std::string("c"));
  feature_names.push_back(std::string("d"));
  auto node_feat =
      client1.get_node_feat(std::string("user"), node_ids, feature_names);
  ASSERT_EQ(node_feat.size(), 2UL);
  ASSERT_EQ(node_feat[0].size(), 2UL);
  VLOG(0) << "get_node_feat: " << node_feat[0][0];
  VLOG(0) << "get_node_feat: " << node_feat[0][1];
  VLOG(0) << "get_node_feat: " << node_feat[1][0];
  VLOG(0) << "get_node_feat: " << node_feat[1][1];

  node_feat[1][0] = "helloworld";

  client1.set_node_feat(
      std::string("user"), node_ids, feature_names, node_feat);

  // sleep(5);
  node_feat =
      client1.get_node_feat(std::string("user"), node_ids, feature_names);
  VLOG(0) << "get_node_feat: " << node_feat[1][0];
  ASSERT_EQ(node_feat[1][0], "helloworld");

  // Test string
  node_ids.clear();
  node_ids.push_back(37);
  node_ids.push_back(96);
  // std::vector<std::string> feature_names;
  feature_names.clear();
  feature_names.push_back(std::string("a"));
  feature_names.push_back(std::string("b"));
  node_feat =
      client1.get_node_feat(std::string("user"), node_ids, feature_names);
  ASSERT_EQ(node_feat.size(), 2UL);
  ASSERT_EQ(node_feat[0].size(), 2UL);
  VLOG(0) << "get_node_feat: " << node_feat[0][0].size();
  VLOG(0) << "get_node_feat: " << node_feat[0][1].size();
  VLOG(0) << "get_node_feat: " << node_feat[1][0].size();
  VLOG(0) << "get_node_feat: " << node_feat[1][1].size();

  std::remove(edge_file_name);
  std::remove(node_file_name);
  // testAddNode(worker_ptr_);
  // LOG(INFO) << "Run stop_server";
  // worker_ptr_->StopServer();
  // LOG(INFO) << "Run finalize_worker";
  // worker_ptr_->FinalizeWorker();
  testFeatureNodeSerializeInt();
  testFeatureNodeSerializeInt64();
  testFeatureNodeSerializeFloat32();
  testFeatureNodeSerializeFloat64();
  testGraphToBuffer();
  client1.StopServer();
}

/*void testCache() {
  ::paddle::distributed::ScaledLRU<::paddle::distributed::SampleKey,
                                   ::paddle::distributed::SampleResult>
      st(1, 2, 4);
  char* str = new char[7];
  strcpy(str, "54321");
  ::paddle::distributed::SampleResult* result =
      new ::paddle::distributed::SampleResult(5, str);
  ::paddle::distributed::SampleKey skey = {6, 1, false};
  std::vector<std::pair<::paddle::distributed::SampleKey,
                        paddle::distributed::SampleResult>>
      r;
  st.query(0, &skey, 1, r);
  ASSERT_EQ((int)r.size(), 0);

  st.insert(0, &skey, result, 1);
  for (size_t i = 0; i < st.get_ttl(); i++) {
    st.query(0, &skey, 1, r);
    ASSERT_EQ((int)r.size(), 1);
    char* p = (char*)r[0].second.buffer.get();
    for (size_t j = 0; j < r[0].second.actual_size; j++)
      ASSERT_EQ(p[j], str[j]);
    r.clear();
  }
  st.query(0, &skey, 1, r);
  ASSERT_EQ((int)r.size(), 0);
  str = new char[10];
  strcpy(str, "54321678");
  result = new ::paddle::distributed::SampleResult(strlen(str), str);
  st.insert(0, &skey, result, 1);
  for (size_t i = 0; i < st.get_ttl() / 2; i++) {
    st.query(0, &skey, 1, r);
    ASSERT_EQ((int)r.size(), 1);
    char* p = (char*)r[0].second.buffer.get();
    for (size_t j = 0; j < r[0].second.actual_size; j++)
      ASSERT_EQ(p[j], str[j]);
    r.clear();
  }
  str = new char[18];
  strcpy(str, "343332d4321");
  result = new ::paddle::distributed::SampleResult(strlen(str), str);
  st.insert(0, &skey, result, 1);
  for (size_t i = 0; i < st.get_ttl(); i++) {
    st.query(0, &skey, 1, r);
    ASSERT_EQ((int)r.size(), 1);
    char* p = (char*)r[0].second.buffer.get();
    for (int j = 0; j < (int)r[0].second.actual_size; j++)
      ASSERT_EQ(p[j], str[j]);
    r.clear();
  }
  st.query(0, &skey, 1, r);
  ASSERT_EQ((int)r.size(), 0);
}*/
void testGraphToBuffer() {
  ::paddle::distributed::GraphNode s, s1;
  s.set_feature_size(1);
  s.set_feature(0, std::string("hhhh"));
  s.set_id(65);
  int size = s.get_size(true);
  char str[size];  // NOLINT
  s.to_buffer(str, true);
  s1.recover_from_buffer(str);
  ASSERT_EQ(s.get_id(), s1.get_id());
  VLOG(0) << s.get_feature(0);
  VLOG(0) << s1.get_feature(0);
}

TEST(RunBrpcPushSparse, Run) { RunBrpcPushSparse(); }
