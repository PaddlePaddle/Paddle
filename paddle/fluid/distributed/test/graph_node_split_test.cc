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
#include "paddle/fluid/distributed/ps/service/brpc_ps_client.h"
#include "paddle/fluid/distributed/ps/service/brpc_ps_server.h"
#include "paddle/fluid/distributed/ps/service/env.h"
#include "paddle/fluid/distributed/ps/service/graph_brpc_client.h"
#include "paddle/fluid/distributed/ps/service/graph_brpc_server.h"
#include "paddle/fluid/distributed/ps/service/ps_client.h"
#include "paddle/fluid/distributed/ps/service/ps_service/graph_py_service.h"
#include "paddle/fluid/distributed/ps/service/ps_service/service.h"
#include "paddle/fluid/distributed/ps/service/sendrecv.pb.h"
#include "paddle/fluid/distributed/ps/table/graph/graph_node.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/string/printf.h"
#include "paddle/pten/kernels/funcs/math_function.h"

namespace framework = paddle::framework;
namespace platform = paddle::platform;
namespace operators = paddle::operators;
namespace memory = paddle::memory;
namespace distributed = paddle::distributed;

std::vector<std::string> edges = {
    std::string("37\t45\t0.34"),  std::string("37\t145\t0.31"),
    std::string("37\t112\t0.21"), std::string("96\t48\t1.4"),
    std::string("96\t247\t0.31"), std::string("96\t111\t1.21"),
    std::string("59\t45\t0.34"),  std::string("59\t145\t0.31"),
    std::string("59\t122\t0.21"), std::string("97\t48\t0.34"),
    std::string("97\t247\t0.31"), std::string("97\t111\t0.21")};
char edge_file_name[] = "edges.txt";

std::vector<std::string> nodes = {
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

std::vector<std::string> graph_split = {std::string("0\t97")};
char graph_split_file_name[] = "graph_split.txt";

void prepare_file(char file_name[], std::vector<std::string> data) {
  std::ofstream ofile;
  ofile.open(file_name);
  for (auto x : data) {
    ofile << x << std::endl;
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
  pserver_ptr_->build_peer2peer_connection(0);
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
  pserver_ptr2->build_peer2peer_connection(1);
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

void RunGraphSplit() {
  setenv("http_proxy", "", 1);
  setenv("https_proxy", "", 1);
  prepare_file(edge_file_name, edges);
  prepare_file(node_file_name, nodes);
  prepare_file(graph_split_file_name, graph_split);
  auto ph_host = paddle::distributed::PSHost(ip_, port_, 0);
  host_sign_list_.push_back(ph_host.serialize_to_string());

  // test-start
  auto ph_host2 = paddle::distributed::PSHost(ip2, port2, 1);
  host_sign_list_.push_back(ph_host2.serialize_to_string());
  // test-end
  // Srart Server
  std::thread* server_thread = new std::thread(RunServer);

  std::thread* server_thread2 = new std::thread(RunServer2);

  sleep(2);
  std::map<uint64_t, std::vector<paddle::distributed::Region>> dense_regions;
  dense_regions.insert(
      std::pair<uint64_t, std::vector<paddle::distributed::Region>>(0, {}));
  auto regions = dense_regions[0];

  RunClient(dense_regions, 0, pserver_ptr_->get_service());

  /*-----------------------Test Server Init----------------------------------*/

  auto pull_status = worker_ptr_->load_graph_split_config(
      0, std::string(graph_split_file_name));
  pull_status.wait();
  pull_status =
      worker_ptr_->load(0, std::string(edge_file_name), std::string("e>"));
  srand(time(0));
  pull_status.wait();
  std::vector<std::vector<uint64_t>> _vs;
  std::vector<std::vector<float>> vs;
  pull_status = worker_ptr_->batch_sample_neighbors(
      0, std::vector<uint64_t>(1, 10240001024), 4, _vs, vs, true);
  pull_status.wait();
  ASSERT_EQ(0, _vs[0].size());
  _vs.clear();
  vs.clear();
  pull_status = worker_ptr_->batch_sample_neighbors(
      0, std::vector<uint64_t>(1, 97), 4, _vs, vs, true);
  pull_status.wait();
  ASSERT_EQ(3, _vs[0].size());
  std::remove(edge_file_name);
  std::remove(node_file_name);
  std::remove(graph_split_file_name);
  LOG(INFO) << "Run stop_server";
  worker_ptr_->stop_server();
  LOG(INFO) << "Run finalize_worker";
  worker_ptr_->finalize_worker();
}

TEST(RunGraphSplit, Run) { RunGraphSplit(); }
