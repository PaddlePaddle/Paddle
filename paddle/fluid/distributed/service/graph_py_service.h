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

#pragma once
#include <unistd.h>
#include <condition_variable>  // NOLINT
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>  // NOLINT
#include <unordered_map>
#include <vector>
#include "google/protobuf/text_format.h"

#include "gtest/gtest.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/framework/variable.h"

#include "paddle/fluid/distributed/ps.pb.h"
#include "paddle/fluid/distributed/service/env.h"
#include "paddle/fluid/distributed/service/graph_brpc_client.h"
#include "paddle/fluid/distributed/service/graph_brpc_server.h"
#include "paddle/fluid/distributed/service/sendrecv.pb.h"
#include "paddle/fluid/distributed/service/service.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/string/printf.h"
namespace paddle {
namespace distributed {
class GraphPyService {
  std::vector<int> keys;
  std::vector<std::string> server_list, port_list, host_sign_list;
  int server_size, shard_num, rank, client_id;
  std::unordered_map<std::string, uint32_t> table_id_map;
  std::thread *server_thread, *client_thread;

  std::shared_ptr<paddle::distributed::PSServer> pserver_ptr;

  std::shared_ptr<paddle::distributed::PSClient> worker_ptr;

 public:
  std::shared_ptr<paddle::distributed::PSServer> get_ps_server() {
    return pserver_ptr;
  }
  std::shared_ptr<paddle::distributed::PSClient> get_ps_client() {
    return worker_ptr;
  }
  int get_client_id() { return client_id; }
  void set_client_Id(int client_Id) { this->client_id = client_id; }
  int get_rank() { return rank; }
  void set_rank(int rank) { this->rank = rank; }
  int get_shard_num() { return shard_num; }
  void set_shard_num(int shard_num) { this->shard_num = shard_num; }
  void GetDownpourSparseTableProto(
      ::paddle::distributed::TableParameter* sparse_table_proto,
      uint32_t table_id) {
    sparse_table_proto->set_table_id(table_id);
    sparse_table_proto->set_table_class("GraphTable");
    sparse_table_proto->set_shard_num(shard_num);
    sparse_table_proto->set_type(::paddle::distributed::PS_SPARSE_TABLE);
    ::paddle::distributed::TableAccessorParameter* accessor_proto =
        sparse_table_proto->mutable_accessor();
    ::paddle::distributed::CommonAccessorParameter* common_proto =
        sparse_table_proto->mutable_common();

    accessor_proto->set_accessor_class("CommMergeAccessor");
  }

  ::paddle::distributed::PSParameter GetServerProto() {
    // Generate server proto desc
    ::paddle::distributed::PSParameter server_fleet_desc;
    server_fleet_desc.set_shard_num(shard_num);
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

    for (auto& tuple : this->table_id_map) {
      ::paddle::distributed::TableParameter* sparse_table_proto =
          downpour_server_proto->add_downpour_table_param();
      GetDownpourSparseTableProto(sparse_table_proto, tuple.second);
    }

    return server_fleet_desc;
  }

  ::paddle::distributed::PSParameter GetWorkerProto() {
    ::paddle::distributed::PSParameter worker_fleet_desc;
    worker_fleet_desc.set_shard_num(shard_num);
    ::paddle::distributed::WorkerParameter* worker_proto =
        worker_fleet_desc.mutable_worker_param();

    ::paddle::distributed::DownpourWorkerParameter* downpour_worker_proto =
        worker_proto->mutable_downpour_worker_param();

    for (auto& tuple : this->table_id_map) {
      ::paddle::distributed::TableParameter* worker_sparse_table_proto =
          downpour_worker_proto->add_downpour_table_param();
      GetDownpourSparseTableProto(worker_sparse_table_proto, tuple.second);
    }

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

    for (auto& tuple : this->table_id_map) {
      ::paddle::distributed::TableParameter* sparse_table_proto =
          downpour_server_proto->add_downpour_table_param();
      GetDownpourSparseTableProto(sparse_table_proto, tuple.second);
    }

    return worker_fleet_desc;
  }
  void set_server_size(int server_size) { this->server_size = server_size; }
  int get_server_size(int server_size) { return server_size; }
  std::vector<std::string> split(std::string& str, const char pattern);

  void load_edge_file(std::string name, std::string filepath, bool reverse) {
    std::string params = "edge";
    if (reverse) {
      params += "|reverse";
    }
    if (this->table_id_map.count(name)) {
      uint32_t table_id = this->table_id_map[name];
      auto status =
          get_ps_client()->load(table_id, std::string(filepath), params);
      status.wait();
    }
  }

  void load_node_file(std::string name, std::string filepath) {
    std::string params = "node";
    if (this->table_id_map.count(name)) {
      uint32_t table_id = this->table_id_map[name];
      auto status =
          get_ps_client()->load(table_id, std::string(filepath), params);
      status.wait();
    }
  }

  std::vector<std::pair<uint64_t, float>> sample_k(std::string name,
                                                   uint64_t node_id,
                                                   int sample_size) {
    std::vector<std::pair<uint64_t, float>> v;
    if (this->table_id_map.count(name)) {
      uint32_t table_id = this->table_id_map[name];
      auto status = worker_ptr->sample(table_id, node_id, sample_size, v);
      status.wait();
    }
    return v;
  }
  std::vector<GraphNode> pull_graph_list(std::string name, int server_index,
                                         int start, int size) {
    std::vector<GraphNode> res;
    if (this->table_id_map.count(name)) {
      uint32_t table_id = this->table_id_map[name];
      auto status =
          worker_ptr->pull_graph_list(table_id, server_index, start, size, res);
      status.wait();
    }
    return res;
  }
  void start_server(std::string ip, uint32_t port) {
    server_thread = new std::thread([this, &ip, &port]() {
      std::function<void()> func = [this, &ip, &port]() {
        VLOG(0) << "enter inner function ";
        ::paddle::distributed::PSParameter server_proto =
            this->GetServerProto();

        auto _ps_env = paddle::distributed::PaddlePSEnvironment();
        _ps_env.set_ps_servers(&this->host_sign_list,
                               this->host_sign_list.size());  // test
        pserver_ptr = std::shared_ptr<paddle::distributed::PSServer>(
            paddle::distributed::PSServerFactory::create(server_proto));
        VLOG(0) << "pserver-ptr created ";
        std::vector<framework::ProgramDesc> empty_vec;
        framework::ProgramDesc empty_prog;
        empty_vec.push_back(empty_prog);
        pserver_ptr->configure(server_proto, _ps_env, rank, empty_vec);
        VLOG(0) << " starting server " << ip << " " << port;
        pserver_ptr->start(ip, port);
      };
      std::thread t1(func);
      t1.join();
    });
  }
  void start_client() {
    VLOG(0) << "in start_client " << rank;
    std::map<uint64_t, std::vector<paddle::distributed::Region>> dense_regions;
    dense_regions.insert(
        std::pair<uint64_t, std::vector<paddle::distributed::Region>>(0, {}));
    auto regions = dense_regions[0];
    ::paddle::distributed::PSParameter worker_proto = GetWorkerProto();
    paddle::distributed::PaddlePSEnvironment _ps_env;
    auto servers_ = host_sign_list.size();
    _ps_env = paddle::distributed::PaddlePSEnvironment();
    _ps_env.set_ps_servers(&host_sign_list, servers_);
    worker_ptr = std::shared_ptr<paddle::distributed::PSClient>(
        paddle::distributed::PSClientFactory::create(worker_proto));
    worker_ptr->configure(worker_proto, dense_regions, _ps_env, client_id);
  }
  void set_up(std::string ips_str, int shard_num, int rank, int client_id,
              std::vector<std::string> edge_types);
  void set_keys(std::vector<int> keys) {  // just for test
    this->keys = keys;
  }
  std::vector<int> get_keys(int start, int size) {  // just for test
    return std::vector<int>(keys.begin() + start, keys.begin() + start + size);
  }
};
}
}
