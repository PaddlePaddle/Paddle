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
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/framework/variable.h"

#include "paddle/fluid/distributed/ps.pb.h"
#include "paddle/fluid/distributed/service/brpc_ps_client.h"
#include "paddle/fluid/distributed/service/brpc_ps_server.h"
#include "paddle/fluid/distributed/service/env.h"
#include "paddle/fluid/distributed/service/graph_py_service.h"
#include "paddle/fluid/distributed/service/ps_client.h"
#include "paddle/fluid/distributed/service/sendrecv.pb.h"
#include "paddle/fluid/distributed/service/service.h"
#include "paddle/fluid/distributed/table/graph_node.h"
#include "paddle/fluid/framework/program_desc.h"
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
    std::shared_ptr<paddle::distributed::PSClient>& worker_ptr_) {
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
void testSingleSampleNeighboor(
    std::shared_ptr<paddle::distributed::PSClient>& worker_ptr_) {
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
    std::shared_ptr<paddle::distributed::PSClient>& worker_ptr_) {
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

std::string nodes[] = {
    std::string("37\t45\t0.34"),  std::string("37\t145\t0.31"),
    std::string("37\t112\t0.21"), std::string("96\t48\t1.4"),
    std::string("96\t247\t0.31"), std::string("96\t111\t1.21"),
    std::string("59\t45\t0.34"),  std::string("59\t145\t0.31"),
    std::string("59\t122\t0.21"), std::string("97\t48\t0.34"),
    std::string("97\t247\t0.31"), std::string("97\t111\t0.21"),
};
char file_name[] = "nodes.txt";
void prepare_file(char file_name[]) {
  std::ofstream ofile;
  ofile.open(file_name);
  for (auto x : nodes) {
    ofile << x << std::endl;
  }
  // for(int i = 0;i < 10;i++){
  //   for(int j = 0;j < 10;j++){
  //    ofile<<i * 127 + j<<"\t"<<i <<"\t"<< 0.5<<std::endl;
  //   }
  //}
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
  server_fleet_desc.set_shard_num(127);
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
  worker_fleet_desc.set_shard_num(127);
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
uint32_t port_ = 4209, port2 = 4210;

std::vector<std::string> host_sign_list_;

std::shared_ptr<paddle::distributed::PSServer> pserver_ptr_, pserver_ptr2;

std::shared_ptr<paddle::distributed::PSClient> worker_ptr_;

void RunServer() {
  LOG(INFO) << "init first server";
  ::paddle::distributed::PSParameter server_proto = GetServerProto();

  auto _ps_env = paddle::distributed::PaddlePSEnvironment();
  _ps_env.set_ps_servers(&host_sign_list_, 2);  // test
  pserver_ptr_ = std::shared_ptr<paddle::distributed::PSServer>(
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
  pserver_ptr2 = std::shared_ptr<paddle::distributed::PSServer>(
      paddle::distributed::PSServerFactory::create(server_proto2));
  std::vector<framework::ProgramDesc> empty_vec2;
  framework::ProgramDesc empty_prog2;
  empty_vec2.push_back(empty_prog2);
  pserver_ptr2->configure(server_proto2, _ps_env2, 1, empty_vec2);
  pserver_ptr2->start(ip2, port2);
}

void RunClient(std::map<uint64_t, std::vector<paddle::distributed::Region>>&
                   dense_regions) {
  ::paddle::distributed::PSParameter worker_proto = GetWorkerProto();
  paddle::distributed::PaddlePSEnvironment _ps_env;
  auto servers_ = host_sign_list_.size();
  _ps_env = paddle::distributed::PaddlePSEnvironment();
  _ps_env.set_ps_servers(&host_sign_list_, servers_);
  worker_ptr_ = std::shared_ptr<paddle::distributed::PSClient>(
      paddle::distributed::PSClientFactory::create(worker_proto));
  worker_ptr_->configure(worker_proto, dense_regions, _ps_env, 0);
}

void RunBrpcPushSparse() {
  setenv("http_proxy", "", 1);
  setenv("https_proxy", "", 1);
  prepare_file(file_name);
  auto ph_host = paddle::distributed::PSHost(ip_, port_, 0);
  host_sign_list_.push_back(ph_host.serialize_to_string());

  // test-start
  auto ph_host2 = paddle::distributed::PSHost(ip2, port2, 1);
  host_sign_list_.push_back(ph_host2.serialize_to_string());
  // test-end
  // Srart Server
  std::thread server_thread(RunServer);
  std::thread server_thread2(RunServer2);
  sleep(1);

  std::map<uint64_t, std::vector<paddle::distributed::Region>> dense_regions;
  dense_regions.insert(
      std::pair<uint64_t, std::vector<paddle::distributed::Region>>(0, {}));
  auto regions = dense_regions[0];

  RunClient(dense_regions);

  /*-----------------------Test Server Init----------------------------------*/
  auto pull_status =
      worker_ptr_->load(0, std::string(file_name), std::string("edge"));
  srand(time(0));
  pull_status.wait();
  std::vector<std::vector<std::pair<uint64_t, float>>> vs;
  // for(int i = 0;i < 100000000;i++){
  //   std::vector<distributed::GraphNode> nodes;
  // pull_status = worker_ptr_->pull_graph_list(0, 0, 0, 1, nodes);
  //  pull_status.wait();
  // pull_status = worker_ptr_->batch_sample(0, std::vector<uint64_t>(1, 37), 4,
  // vs);
  // pull_status.wait();
  // }
  // std::vector<std::pair<uint64_t, float>> v;
  // pull_status = worker_ptr_->sample(0, 37, 4, v);
  testSampleNodes(worker_ptr_);
  testSingleSampleNeighboor(worker_ptr_);
  testBatchSampleNeighboor(worker_ptr_);
  pull_status = worker_ptr_->batch_sample_neighboors(
      0, std::vector<uint64_t>(1, 10240001024), 4, vs);
  pull_status.wait();
  ASSERT_EQ(0, vs[0].size());

  std::vector<distributed::GraphNode> nodes;
  pull_status = worker_ptr_->pull_graph_list(0, 0, 0, 1, nodes);
  pull_status.wait();
  ASSERT_EQ(nodes.size(), 1);
  ASSERT_EQ(nodes[0].get_id(), 37);
  nodes.clear();
  pull_status = worker_ptr_->pull_graph_list(0, 0, 1, 4, nodes);
  pull_status.wait();
  ASSERT_EQ(nodes.size(), 1);
  ASSERT_EQ(nodes[0].get_id(), 59);
  for (auto g : nodes) {
    std::cout << g.get_id() << std::endl;
  }
  distributed::GraphPyServer server1, server2;
  distributed::GraphPyClient client1, client2;
  std::string ips_str = "127.0.0.1:4211;127.0.0.1:4212";
  std::vector<std::string> edge_types = {std::string("user2item")};
  server1.set_up(ips_str, 127, edge_types, 0);
  server2.set_up(ips_str, 127, edge_types, 1);
  client1.set_up(ips_str, 127, edge_types, 0);
  client2.set_up(ips_str, 127, edge_types, 1);
  server1.start_server();
  std::cout << "first server done" << std::endl;
  server2.start_server();
  std::cout << "second server done" << std::endl;
  client1.start_client();
  std::cout << "first client done" << std::endl;
  client2.start_client();
  std::cout << "first client done" << std::endl;
  std::cout << "started" << std::endl;
  client1.load_edge_file(std::string("user2item"), std::string(file_name), 0);
  // client2.load_edge_file(std::string("user2item"), std::string(file_name),
  // 0);
  nodes.clear();
  nodes = client2.pull_graph_list(std::string("user2item"), 0, 1, 4);
  ASSERT_EQ(nodes[0].get_id(), 59);
  nodes.clear();
  vs = client1.batch_sample_k(std::string("user2item"),
                              std::vector<uint64_t>(1, 96), 4);
  ASSERT_EQ(vs[0].size(), 3);
  std::cout << "batch sample result" << std::endl;
  for (auto p : vs[0]) {
    std::cout << p.first << " " << p.second << std::endl;
  }
  std::vector<uint64_t> node_ids;
  node_ids.push_back(96);
  node_ids.push_back(37);
  vs = client1.batch_sample_k(std::string("user2item"), node_ids, 4);
  ASSERT_EQ(vs.size(), 2);
  // to test in python,try this:
  //   from paddle.fluid.core import GraphPyService
  // ips_str = "127.0.0.1:4211;127.0.0.1:4212"
  // gps1 = GraphPyService();
  // gps2 = GraphPyService();
  // gps1.set_up(ips_str, 127, 0, 0, 0);
  // gps2.set_up(ips_str, 127, 1, 1, 0);
  // gps1.load_file("input.txt");

  // list = gps2.pull_graph_list(0,1,4)
  // for x in list:
  //     print(x.get_id())

  // list = gps2.sample_k(96, "user", 4);
  // for x in list:
  //     print(x.get_id())

  std::remove(file_name);
  LOG(INFO) << "Run stop_server";
  worker_ptr_->stop_server();
  LOG(INFO) << "Run finalize_worker";
  worker_ptr_->finalize_worker();
  server_thread.join();
  server_thread2.join();
  testGraphToBuffer();
}

void testGraphToBuffer() {
  ::paddle::distributed::GraphNode s, s1;
  s.set_feature("hhhh");
  s.set_id(65);
  int size = s.get_size(true);
  char str[size];
  s.to_buffer(str, true);
  s1.recover_from_buffer(str);
  ASSERT_EQ(s.get_id(), s1.get_id());
  VLOG(0) << s.get_feature();
  VLOG(0) << s1.get_feature();
}

TEST(RunBrpcPushSparse, Run) { RunBrpcPushSparse(); }
