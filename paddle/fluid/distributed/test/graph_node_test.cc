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

void testGraphToBuffer();
std::string nodes[] = {
    std::string("37\tuser\t45;user;0.34\t145;user;0.31\t112;item;0.21"),
    std::string("96\tuser\t48;user;1.4\t247;user;0.31\t111;item;1.21")};
char file_name[] = "nodes.txt";
void prepare_file(char file_name[]) {
  std::ofstream ofile;
  ofile.open(file_name);
  for (auto x : nodes) {
    ofile << x << std::endl;
  }
  ofile.close();
}
distributed::GraphNodeType get_graph_node_type(std::string& str) {
  distributed::GraphNodeType type;
  if (str == "user")
    type = distributed::GraphNodeType::user;
  else if (str == "item")
    type = distributed::GraphNodeType::item;
  else if (str == "query")
    type = distributed::GraphNodeType::query;
  else
    type = distributed::GraphNodeType::unknown;
  return type;
}
void GetDownpourSparseTableProto(
    ::paddle::distributed::TableParameter* sparse_table_proto) {
  sparse_table_proto->set_table_id(0);
  sparse_table_proto->set_table_class("GraphTable");
  sparse_table_proto->set_shard_num(256);
  sparse_table_proto->set_type(::paddle::distributed::PS_SPARSE_TABLE);
  ::paddle::distributed::TableAccessorParameter* accessor_proto =
      sparse_table_proto->mutable_accessor();
  ::paddle::distributed::CommonAccessorParameter* common_proto =
      sparse_table_proto->mutable_common();
}

distributed::GraphNodeType get_graph_node_type(std::string str) {
  distributed::GraphNodeType type;
  if (str == "user")
    type = distributed::GraphNodeType::user;
  else if (str == "item")
    type = distributed::GraphNodeType::item;
  else if (str == "query")
    type = distributed::GraphNodeType::query;
  else
    type = distributed::GraphNodeType::unknown;
  return type;
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
  LOG(INFO) << "Run pull_sparse_param";
  auto pull_status =
      worker_ptr_->load(0, std::string(file_name), std::string(""));
  pull_status.wait();
  std::vector<distributed::GraphNode> v;
  pull_status = worker_ptr_->sample(
      0, 37, get_graph_node_type(std::string("user")), 4, v);
  pull_status.wait();
  for (auto g : v) {
    std::cout << g.get_id() << " " << g.get_graph_node_type() << std::endl;
  }
  /*-----------------------Test Push Param----------------------------------*/

  //   LOG(INFO) << "Run push_sparse_param";
  //   paddle::distributed::DownpourBrpcClosure* closure_push_param =
  //       new paddle::distributed::DownpourBrpcClosure(2, [&](void* done) {
  //         int ret = 0;
  //         auto* closure = (paddle::distributed::DownpourBrpcClosure*)done;
  //         for (size_t i = 0; i < 2; ++i) {
  //   LOG(INFO) << "check response" <<i;
  //           if (closure->check_response(
  //                   i, paddle::distributed::PS_PUSH_SPARSE_PARAM) != 0) {
  //             ret = -1;
  //             break;
  //           }
  //           VLOG(0)<<i<<" push_sparse_param response
  //           "<<closure->check_response(
  //                   i, paddle::distributed::PS_PUSH_SPARSE_PARAM);
  //         }
  //         closure->set_promise_value(ret);
  //       });
  //   VLOG(0)<<"begin to push_sparse_param";
  //   auto push_status = worker_ptr_->push_sparse_param(
  //       0, fea_keys.data(), (const float**)fea_value_ptr.data(),
  //       fea_keys.size(),
  //       closure_push_param);
  //   push_status.wait();

  //   auto pull_param_status = worker_ptr_->pull_sparse(
  //       fea_temp_value_ptr.data(), 0, fea_keys.data(), fea_keys.size());
  //   pull_param_status.wait();

  //   for (size_t idx = 0; idx < tensor->numel(); ++idx) {
  //     EXPECT_FLOAT_EQ(fea_temp_values[idx], fea_values[idx]);
  //   }
  // LOG(INFO) << "first stage done";
  //   /*-----------------------Test Push
  //   Grad----------------------------------*/

  //   paddle::distributed::DownpourBrpcClosure* closure_push_grad =
  //       new paddle::distributed::DownpourBrpcClosure(2, [&](void* done) {
  //         int ret = 0;
  //         auto* closure = (paddle::distributed::DownpourBrpcClosure*)done;
  //         for (size_t i = 0; i < 2; ++i) {
  //           if (closure->check_response(
  //                   i, paddle::distributed::PS_PUSH_SPARSE_TABLE) != 0) {
  //             ret = -1;
  //             break;
  //           }
  //         }
  //         closure->set_promise_value(ret);
  //       });

  //   LOG(INFO) << "Run pull_sparse_grad";
  //   std::vector<float*> push_g_vec;
  //   for (auto i = 0; i < static_cast<int>(fea_keys.size()); ++i) {
  //     push_g_vec.push_back(tensor->data<float>() + i * 10);
  //   }
  //   auto push_grad_status = worker_ptr_->push_sparse_raw_gradient(
  //       0, fea_keys.data(), (const float**)push_g_vec.data(),
  //       fea_keys.size(),
  //       closure_push_grad);
  //   push_grad_status.wait();

  //   auto pull_update_status = worker_ptr_->pull_sparse(
  //       fea_temp_value_ptr.data(), 0, fea_keys.data(), fea_keys.size());
  //   pull_update_status.wait();

  //   for (size_t idx = 0; idx < tensor->numel(); ++idx) {
  //     EXPECT_FLOAT_EQ(fea_temp_values[idx], fea_values[idx] - 1.0);
  //   }

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
  s.set_graph_node_type(::paddle::distributed::GraphNodeType(0));
  int size = s.get_size();
  char str[size];
  s.to_buffer(str);
  s1.recover_from_buffer(str);
  ASSERT_EQ(s.get_id(), s1.get_id());
  ASSERT_EQ((int)s.get_graph_node_type(), (int)s1.get_graph_node_type());
  VLOG(0) << s.get_feature();
  VLOG(0) << s1.get_feature();
}
TEST(RunBrpcPushSparse, Run) { RunBrpcPushSparse(); }
