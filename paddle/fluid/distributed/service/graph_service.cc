// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <sstream>
#include <string>
#include <vector>
// #include <unistd.h>
// #include <condition_variable>  // NOLINT
// #include <thread>  // NOLINT
// #include<fstream>
// #include<iomanip>
// #include<iostream>
// #include "google/protobuf/text_format.h"

// #include "gtest/gtest.h"
// #include "paddle/fluid/framework/lod_tensor.h"
// #include "paddle/fluid/framework/scope.h"
// #include "paddle/fluid/framework/tensor_util.h"
// #include "paddle/fluid/framework/variable.h"

#include "paddle/fluid/distributed/ps.pb.h"
// #include "paddle/fluid/distributed/service/brpc_ps_client.h"
// #include "paddle/fluid/distributed/service/brpc_ps_server.h"
// #include "paddle/fluid/distributed/service/env.h"
// #include "paddle/fluid/distributed/service/ps_client.h"
// #include "paddle/fluid/distributed/service/sendrecv.pb.h"
// #include "paddle/fluid/distributed/service/service.h"
#include "paddle/fluid/distributed/table/graph_node.h"
// #include "paddle/fluid/framework/program_desc.h"
// #include "paddle/fluid/operators/math/math_function.h"
// #include "paddle/fluid/platform/place.h"
// #include "paddle/fluid/string/printf.h"
namespace paddle {
namespace distributed {
class graph_service {
  std::vector<int> keys;
  std::vector<std::string> server_list, port_list, host_sign_list;
  int server_size, shard_num;

 public:
  // void GetDownpourSparseTableProto(
  //     ::paddle::distributed::TableParameter* sparse_table_proto) {
  //   sparse_table_proto->set_table_id(0);
  //   sparse_table_proto->set_table_class("GraphTable");
  //   sparse_table_proto->set_shard_num(256);
  //   sparse_table_proto->set_type(::paddle::distributed::PS_SPARSE_TABLE);
  //   ::paddle::distributed::TableAccessorParameter* accessor_proto =
  //       sparse_table_proto->mutable_accessor();
  //   ::paddle::distributed::CommonAccessorParameter* common_proto =
  //       sparse_table_proto->mutable_common();

  //   accessor_proto->set_accessor_class("CommMergeAccessor");
  //   accessor_proto->set_fea_dim(0);
  //   accessor_proto->set_embedx_dim(10);

  //   common_proto->set_name("sgd");
  //   common_proto->set_table_name("MergedDense");
  //   common_proto->set_trainer_num(1);
  //   common_proto->set_sync(false);
  //   common_proto->add_params("Param");
  //   common_proto->add_dims(10);
  //   common_proto->add_initializers("uniform_random&0&-1.0&1.0");
  //   common_proto->add_params("LearningRate");
  //   common_proto->add_dims(1);
  //   common_proto->add_initializers("fill_constant&1.0");
  // }

  //  ::paddle::distributed::PSParameter GetWorkerProto(int shard_num) {
  //   ::paddle::distributed::PSParameter worker_fleet_desc;
  //   worker_fleet_desc.set_shard_num(shard_num);
  //   ::paddle::distributed::WorkerParameter* worker_proto =
  //       worker_fleet_desc.mutable_worker_param();

  //   ::paddle::distributed::DownpourWorkerParameter* downpour_worker_proto =
  //       worker_proto->mutable_downpour_worker_param();

  //   ::paddle::distributed::TableParameter* worker_sparse_table_proto =
  //       downpour_worker_proto->add_downpour_table_param();
  //   GetDownpourSparseTableProto(worker_sparse_table_proto);

  //   ::paddle::distributed::ServerParameter* server_proto =
  //       worker_fleet_desc.mutable_server_param();
  //   ::paddle::distributed::DownpourServerParameter* downpour_server_proto =
  //       server_proto->mutable_downpour_server_param();
  //   ::paddle::distributed::ServerServiceParameter* server_service_proto =
  //       downpour_server_proto->mutable_service_param();
  //   server_service_proto->set_service_class("GraphBrpcService");
  //   server_service_proto->set_server_class("GraphBrpcServer");
  //   server_service_proto->set_client_class("GraphBrpcClient");
  //   server_service_proto->set_start_server_port(0);
  //   server_service_proto->set_server_thread_num(12);

  //   ::paddle::distributed::TableParameter* server_sparse_table_proto =
  //       downpour_server_proto->add_downpour_table_param();
  //   GetDownpourSparseTableProto(server_sparse_table_proto);

  //   return worker_fleet_desc;
  // }
  void set_server_size(int server_size) { this->server_size = server_size; }
  int get_server_size(int server_size) { return server_size; }
  std::vector<std::string> split(std::string &str, const char pattern) {
    std::vector<std::string> res;
    std::stringstream input(str);
    std::string temp;
    while (std::getline(input, temp, pattern)) {
      res.push_back(temp);
    }
    return res;
  }
  void start_client() {
    //   framework::Scope client_scope;
    //   platform::CPUPlace place;
    //   InitTensorsOnClient(&client_scope, &place, 100);
    //   std::map<uint64_t, std::vector<paddle::distributed::Region>>
    //   dense_regions;
    //   dense_regions.insert(
    //       std::pair<uint64_t, std::vector<paddle::distributed::Region>>(0,
    //       {}));
    //   auto regions = dense_regions[0];
    //   framework::Variable* var = client_scope.FindVar("x");
    //   framework::LoDTensor* tensor = var->GetMutable<framework::LoDTensor>();

    //   RunClient(dense_regions);
    //   ::paddle::distributed::PSParameter worker_proto = GetWorkerProto();
    //   paddle::distributed::PaddlePSEnvironment _ps_env;
    //   auto servers_ = host_sign_list_.size();
    //   _ps_env = paddle::distributed::PaddlePSEnvironment();
    //   _ps_env.set_ps_servers(&host_sign_list_, servers_);
    //   worker_ptr_ = std::shared_ptr<paddle::distributed::PSClient>(
    //       paddle::distributed::PSClientFactory::create(worker_proto));
    //   worker_ptr_->configure(worker_proto, dense_regions, _ps_env, 0);
  }
  void init(std::string ips_str, int shard_num) {
    std::istringstream stream(ips_str);
    std::string ip, port;
    server_size = 0;
    std::vector<std::string> ips_list = split(ips_str, ';');
    int index = 0;
    for (auto ips : ips_list) {
      auto ip_and_port = split(ips, ':');
      server_list.push_back(ip_and_port[0]);
      port_list.push_back(ip_and_port[1]);
      // auto ph_host = paddle::distributed::PSHost(ip_and_port[0],
      // ip_and_port[1], index);
      // host_sign_list_.push_back(ph_host.serialize_to_string());
      index++;
    }
    start_client();
  }

 public:
  void set_keys(std::vector<int> keys) {  // just for test
    this->keys = keys;
  }
  std::vector<int> get_keys(int start, int size) {  // just for test
    return std::vector<int>(keys.begin() + start, keys.begin() + start + size);
  }
};
}
}