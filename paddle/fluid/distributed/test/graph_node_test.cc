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
#include "paddle/fluid/distributed/ps.pb.h"
#include "paddle/fluid/distributed/service/brpc_ps_client.h"
#include "paddle/fluid/distributed/service/brpc_ps_server.h"
#include "paddle/fluid/distributed/service/env.h"
#include "paddle/fluid/distributed/service/graph_brpc_client.h"
#include "paddle/fluid/distributed/service/graph_brpc_server.h"
#include "paddle/fluid/distributed/service/graph_py_service.h"
#include "paddle/fluid/distributed/service/ps_client.h"
#include "paddle/fluid/distributed/service/sendrecv.pb.h"
#include "paddle/fluid/distributed/service/service.h"
#include "paddle/fluid/distributed/table/graph/graph_node.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/string/printf.h"

namespace framework = paddle::framework;
namespace platform = paddle::platform;
namespace operators = paddle::operators;
namespace math = paddle::operators::math;
namespace memory = paddle::memory;
namespace distributed = paddle::distributed;

void testSampleNodes(
    std::shared_ptr<paddle::distributed::GraphBrpcClient>& worker_ptr_) {
  std::vector<uint64_t> ids;
  auto pull_status = worker_ptr_->random_sample_nodes(0, 0, 6, ids);
  std::unordered_set<uint64_t> s;
  std::unordered_set<uint64_t> s1 = {37, 59};
  pull_status.wait();
  for (auto id : ids) s.insert(id);
  ASSERT_EQ(true, s.size() == s1.size());
  for (auto id : s) {
    ASSERT_EQ(true, s1.find(id) != s1.end());
  }
}

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

void testSingleSampleNeighboor(
    std::shared_ptr<paddle::distributed::GraphBrpcClient>& worker_ptr_) {
  std::vector<std::vector<std::pair<uint64_t, float>>> vs;
  auto pull_status = worker_ptr_->batch_sample_neighboors(
      0, std::vector<uint64_t>(1, 37), 4, vs);
  pull_status.wait();

  std::unordered_set<uint64_t> s;
  std::unordered_set<uint64_t> s1 = {112, 45, 145};
  for (auto g : vs[0]) {
    s.insert(g.first);
  }
  ASSERT_EQ(s.size(), 3);
  for (auto g : s) {
    ASSERT_EQ(true, s1.find(g) != s1.end());
  }
  VLOG(0) << "test single done";
  s.clear();
  s1.clear();
  vs.clear();
  pull_status = worker_ptr_->batch_sample_neighboors(
      0, std::vector<uint64_t>(1, 96), 4, vs);
  pull_status.wait();
  s1 = {111, 48, 247};
  for (auto g : vs[0]) {
    s.insert(g.first);
  }
  ASSERT_EQ(s.size(), 3);
  for (auto g : s) {
    ASSERT_EQ(true, s1.find(g) != s1.end());
  }
}

void testBatchSampleNeighboor(
    std::shared_ptr<paddle::distributed::GraphBrpcClient>& worker_ptr_) {
  std::vector<std::vector<std::pair<uint64_t, float>>> vs;
  std::vector<std::uint64_t> v = {37, 96};
  auto pull_status = worker_ptr_->batch_sample_neighboors(0, v, 4, vs);
  pull_status.wait();
  std::unordered_set<uint64_t> s;
  std::unordered_set<uint64_t> s1 = {112, 45, 145};
  for (auto g : vs[0]) {
    s.insert(g.first);
  }
  ASSERT_EQ(s.size(), 3);
  for (auto g : s) {
    ASSERT_EQ(true, s1.find(g) != s1.end());
  }
  s.clear();
  s1.clear();
  s1 = {111, 48, 247};
  for (auto g : vs[1]) {
    s.insert(g.first);
  }
  ASSERT_EQ(s.size(), 3);
  for (auto g : s) {
    ASSERT_EQ(true, s1.find(g) != s1.end());
  }
}

void testGraphToBuffer();
// std::string nodes[] = {std::string("37\taa\t45;0.34\t145;0.31\t112;0.21"),
//                        std::string("96\tfeature\t48;1.4\t247;0.31\t111;1.21"),
//                        std::string("59\ttreat\t45;0.34\t145;0.31\t112;0.21"),
//                        std::string("97\tfood\t48;1.4\t247;0.31\t111;1.21")};

std::string edges[] = {
    std::string("37\t45\t0.34"),  std::string("37\t145\t0.31"),
    std::string("37\t112\t0.21"), std::string("96\t48\t1.4"),
    std::string("96\t247\t0.31"), std::string("96\t111\t1.21"),
    std::string("59\t45\t0.34"),  std::string("59\t145\t0.31"),
    std::string("59\t122\t0.21"), std::string("97\t48\t0.34"),
    std::string("97\t247\t0.31"), std::string("97\t111\t0.21")};
char edge_file_name[] = "edges.txt";

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

std::string ip_ = "127.0.0.1", ip2 = "127.0.0.1";
uint32_t port_ = 5209, port2 = 5210;

std::vector<std::string> host_sign_list_;

std::shared_ptr<paddle::distributed::GraphBrpcServer> pserver_ptr_,
    pserver_ptr2;

std::shared_ptr<paddle::distributed::GraphBrpcClient> worker_ptr_;

void RunServer() {
  LOG(INFO) << "init first server";
  ::paddle::distributed::PSParameter server_proto = GetServerProto();

  auto _ps_env = paddle::distributed::PaddlePSEnvironment();
  _ps_env.set_ps_servers(&host_sign_list_, 2);  // test
  pserver_ptr_ = std::shared_ptr<paddle::distributed::GraphBrpcServer>(
      (paddle::distributed::GraphBrpcServer*)
          paddle::distributed::PSServerFactory::create(server_proto));
  std::vector<framework::ProgramDesc> empty_vec;
  framework::ProgramDesc empty_prog;
  empty_vec.push_back(empty_prog);
  pserver_ptr_->configure(server_proto, _ps_env, 0, empty_vec);
  LOG(INFO) << "first server, run start(ip,port)";
  pserver_ptr_->start(ip_, port_);
  LOG(INFO) << "init first server Done";
}

void RunServer2() {
  LOG(INFO) << "init second server";
  ::paddle::distributed::PSParameter server_proto2 = GetServerProto();

  auto _ps_env2 = paddle::distributed::PaddlePSEnvironment();
  _ps_env2.set_ps_servers(&host_sign_list_, 2);  // test
  pserver_ptr2 = std::shared_ptr<paddle::distributed::GraphBrpcServer>(
      (paddle::distributed::GraphBrpcServer*)
          paddle::distributed::PSServerFactory::create(server_proto2));
  std::vector<framework::ProgramDesc> empty_vec2;
  framework::ProgramDesc empty_prog2;
  empty_vec2.push_back(empty_prog2);
  pserver_ptr2->configure(server_proto2, _ps_env2, 1, empty_vec2);
  pserver_ptr2->start(ip2, port2);
}

void RunClient(
    std::map<uint64_t, std::vector<paddle::distributed::Region>>& dense_regions,
    int index, paddle::distributed::PsBaseService* service) {
  ::paddle::distributed::PSParameter worker_proto = GetWorkerProto();
  paddle::distributed::PaddlePSEnvironment _ps_env;
  auto servers_ = host_sign_list_.size();
  _ps_env = paddle::distributed::PaddlePSEnvironment();
  _ps_env.set_ps_servers(&host_sign_list_, servers_);
  worker_ptr_ = std::shared_ptr<paddle::distributed::GraphBrpcClient>(
      (paddle::distributed::GraphBrpcClient*)
          paddle::distributed::PSClientFactory::create(worker_proto));
  worker_ptr_->configure(worker_proto, dense_regions, _ps_env, 0);
  worker_ptr_->set_shard_num(127);
  worker_ptr_->set_local_channel(index);
  worker_ptr_->set_local_graph_service(
      (paddle::distributed::GraphBrpcService*)service);
}

void RunBrpcPushSparse() {
  setenv("http_proxy", "", 1);
  setenv("https_proxy", "", 1);
  prepare_file(edge_file_name, 1);
  prepare_file(node_file_name, 0);
  auto ph_host = paddle::distributed::PSHost(ip_, port_, 0);
  host_sign_list_.push_back(ph_host.serialize_to_string());

  // test-start
  auto ph_host2 = paddle::distributed::PSHost(ip2, port2, 1);
  host_sign_list_.push_back(ph_host2.serialize_to_string());
  // test-end
  // Srart Server
  std::thread* server_thread = new std::thread(RunServer);
  std::thread* server_thread2 = new std::thread(RunServer2);
  sleep(1);

  std::map<uint64_t, std::vector<paddle::distributed::Region>> dense_regions;
  dense_regions.insert(
      std::pair<uint64_t, std::vector<paddle::distributed::Region>>(0, {}));
  auto regions = dense_regions[0];

  RunClient(dense_regions, 0, pserver_ptr_->get_service());

  /*-----------------------Test Server Init----------------------------------*/
  auto pull_status =
      worker_ptr_->load(0, std::string(edge_file_name), std::string("e>"));
  srand(time(0));
  pull_status.wait();
  std::vector<std::vector<std::pair<uint64_t, float>>> vs;
  testSampleNodes(worker_ptr_);
  sleep(5);
  testSingleSampleNeighboor(worker_ptr_);
  testBatchSampleNeighboor(worker_ptr_);
  pull_status = worker_ptr_->batch_sample_neighboors(
      0, std::vector<uint64_t>(1, 10240001024), 4, vs);
  pull_status.wait();
  ASSERT_EQ(0, vs[0].size());

  std::vector<distributed::FeatureNode> nodes;
  pull_status = worker_ptr_->pull_graph_list(0, 0, 0, 1, 1, nodes);
  pull_status.wait();
  ASSERT_EQ(nodes.size(), 1);
  ASSERT_EQ(nodes[0].get_id(), 37);
  nodes.clear();
  pull_status = worker_ptr_->pull_graph_list(0, 0, 1, 4, 1, nodes);
  pull_status.wait();
  ASSERT_EQ(nodes.size(), 1);
  ASSERT_EQ(nodes[0].get_id(), 59);
  for (auto g : nodes) {
    std::cout << g.get_id() << std::endl;
  }
  distributed::GraphPyServer server1, server2;
  distributed::GraphPyClient client1, client2;
  std::string ips_str = "127.0.0.1:5211;127.0.0.1:5212";
  std::vector<std::string> edge_types = {std::string("user2item")};
  std::vector<std::string> node_types = {std::string("user"),
                                         std::string("item")};
  VLOG(0) << "make 2 servers";
  server1.set_up(ips_str, 127, node_types, edge_types, 0);
  server2.set_up(ips_str, 127, node_types, edge_types, 1);

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
  client1.load_edge_file(std::string("user2item"), std::string(edge_file_name),
                         0);
  nodes.clear();

  nodes = client1.pull_graph_list(std::string("user"), 0, 1, 4, 1);

  ASSERT_EQ(nodes[0].get_id(), 59);
  nodes.clear();

  // Test Pull by step

  std::unordered_set<uint64_t> count_item_nodes;
  // pull by step 2
  for (int test_step = 1; test_step < 4; test_step++) {
    count_item_nodes.clear();
    std::cout << "check pull graph list by step " << test_step << std::endl;
    for (int server_id = 0; server_id < 2; server_id++) {
      for (int start_step = 0; start_step < test_step; start_step++) {
        nodes = client1.pull_graph_list(std::string("item"), server_id,
                                        start_step, 12, test_step);
        for (auto g : nodes) {
          count_item_nodes.insert(g.get_id());
        }
        nodes.clear();
      }
    }
    ASSERT_EQ(count_item_nodes.size(), 12);
  }

  vs = client1.batch_sample_neighboors(std::string("user2item"),
                                       std::vector<uint64_t>(1, 96), 4);
  ASSERT_EQ(vs[0].size(), 3);
  std::vector<uint64_t> node_ids;
  node_ids.push_back(96);
  node_ids.push_back(37);
  vs = client1.batch_sample_neighboors(std::string("user2item"), node_ids, 4);

  ASSERT_EQ(vs.size(), 2);
  std::vector<uint64_t> nodes_ids = client2.random_sample_nodes("user", 0, 6);
  ASSERT_EQ(nodes_ids.size(), 2);
  ASSERT_EQ(true, (nodes_ids[0] == 59 && nodes_ids[1] == 37) ||
                      (nodes_ids[0] == 37 && nodes_ids[1] == 59));

  // Test get node feat
  node_ids.clear();
  node_ids.push_back(37);
  node_ids.push_back(96);
  std::vector<std::string> feature_names;
  feature_names.push_back(std::string("c"));
  feature_names.push_back(std::string("d"));
  auto node_feat =
      client1.get_node_feat(std::string("user"), node_ids, feature_names);
  ASSERT_EQ(node_feat.size(), 2);
  ASSERT_EQ(node_feat[0].size(), 2);
  VLOG(0) << "get_node_feat: " << node_feat[0][0];
  VLOG(0) << "get_node_feat: " << node_feat[0][1];
  VLOG(0) << "get_node_feat: " << node_feat[1][0];
  VLOG(0) << "get_node_feat: " << node_feat[1][1];

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
  ASSERT_EQ(node_feat.size(), 2);
  ASSERT_EQ(node_feat[0].size(), 2);
  VLOG(0) << "get_node_feat: " << node_feat[0][0].size();
  VLOG(0) << "get_node_feat: " << node_feat[0][1].size();
  VLOG(0) << "get_node_feat: " << node_feat[1][0].size();
  VLOG(0) << "get_node_feat: " << node_feat[1][1].size();

  std::remove(edge_file_name);
  std::remove(node_file_name);
  LOG(INFO) << "Run stop_server";
  worker_ptr_->stop_server();
  LOG(INFO) << "Run finalize_worker";
  worker_ptr_->finalize_worker();
  testFeatureNodeSerializeInt();
  testFeatureNodeSerializeInt64();
  testFeatureNodeSerializeFloat32();
  testFeatureNodeSerializeFloat64();
  testGraphToBuffer();
  client1.stop_server();
}

void testGraphToBuffer() {
  ::paddle::distributed::GraphNode s, s1;
  s.set_feature_size(1);
  s.set_feature(0, std::string("hhhh"));
  s.set_id(65);
  int size = s.get_size(true);
  char str[size];
  s.to_buffer(str, true);
  s1.recover_from_buffer(str);
  ASSERT_EQ(s.get_id(), s1.get_id());
  VLOG(0) << s.get_feature(0);
  VLOG(0) << s1.get_feature(0);
}

TEST(RunBrpcPushSparse, Run) { RunBrpcPushSparse(); }
