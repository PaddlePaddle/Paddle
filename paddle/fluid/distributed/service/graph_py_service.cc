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

#include "paddle/fluid/distributed/service/graph_py_service.h"
#include <thread>  // NOLINT
#include "butil/endpoint.h"
#include "iomanip"
#include "paddle/fluid/distributed/table/table.h"
#include "paddle/fluid/framework/archive.h"
#include "paddle/fluid/platform/profiler.h"
namespace paddle {
namespace distributed {
std::vector<std::string> GraphPyService::split(std::string& str,
                                               const char pattern) {
  std::vector<std::string> res;
  std::stringstream input(str);
  std::string temp;
  while (std::getline(input, temp, pattern)) {
    res.push_back(temp);
  }
  return res;
}

void GraphPyService::set_up(std::string ips_str, int shard_num,
                            std::vector<std::string> edge_types) {
  set_shard_num(shard_num);
  // set_client_Id(client_id);
  // set_rank(rank);

  this->table_id_map[std::string("")] = 0;
  // Table 0 are for nodes
  for (size_t table_id = 0; table_id < edge_types.size(); table_id++) {
    this->table_id_map[edge_types[table_id]] = int(table_id + 1);
  }
  std::istringstream stream(ips_str);
  std::string ip;
  server_size = 0;
  std::vector<std::string> ips_list = split(ips_str, ';');
  int index = 0;
  for (auto ips : ips_list) {
    auto ip_and_port = split(ips, ':');
    server_list.push_back(ip_and_port[0]);
    port_list.push_back(ip_and_port[1]);
    uint32_t port = stoul(ip_and_port[1]);
    auto ph_host = paddle::distributed::PSHost(ip_and_port[0], port, index);
    host_sign_list.push_back(ph_host.serialize_to_string());
    index++;
  }
  // VLOG(0) << "IN set up rank = " << rank;
  // start_client();
  // start_server(server_list[rank], std::stoul(port_list[rank]));
}
void GraphPyClient::start_client() {
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
void GraphPyServer::start_server() {
  std::string ip = server_list[rank];
  uint32_t port = std::stoul(port_list[rank]);
  server_thread = new std::thread([this, &ip, &port]() {
    std::function<void()> func = [this, &ip, &port]() {
      ::paddle::distributed::PSParameter server_proto = this->GetServerProto();

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
      pserver_ptr->start(ip, port);
    };
    std::thread t1(func);
    t1.join();
  });
  sleep(3);
}
::paddle::distributed::PSParameter GraphPyServer::GetServerProto() {
  // Generate server proto desc
  ::paddle::distributed::PSParameter server_fleet_desc;
  server_fleet_desc.set_shard_num(get_shard_num());
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
    VLOG(0) << " make a new table " << tuple.second;
    ::paddle::distributed::TableParameter* sparse_table_proto =
        downpour_server_proto->add_downpour_table_param();
    GetDownpourSparseTableProto(sparse_table_proto, tuple.second);
  }

  return server_fleet_desc;
}

::paddle::distributed::PSParameter GraphPyClient::GetWorkerProto() {
  ::paddle::distributed::PSParameter worker_fleet_desc;
  worker_fleet_desc.set_shard_num(get_shard_num());
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
void GraphPyClient::load_edge_file(std::string name, std::string filepath,
                                   bool reverse) {
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

void GraphPyClient::load_node_file(std::string name, std::string filepath) {
  std::string params = "node";
  if (this->table_id_map.count(name)) {
    uint32_t table_id = this->table_id_map[name];
    auto status =
        get_ps_client()->load(table_id, std::string(filepath), params);
    status.wait();
  }
}
std::vector<std::pair<uint64_t, float>> GraphPyClient::sample_k(
    std::string name, uint64_t node_id, int sample_size) {
  std::vector<std::pair<uint64_t, float>> v;
  if (this->table_id_map.count(name)) {
    uint32_t table_id = this->table_id_map[name];
    auto status = worker_ptr->sample(table_id, node_id, sample_size, v);
    status.wait();
  }
  return v;
}
std::vector<std::vector<std::pair<uint64_t, float> > > GraphPyClient::batch_sample_k(
    std::string name, std::vector<uint64_t> node_ids, int sample_size) {
  std::vector<std::vector<std::pair<uint64_t, float> > > v;
  if (this->table_id_map.count(name)) {
    uint32_t table_id = this->table_id_map[name];
    auto status = worker_ptr->batch_sample(table_id, node_ids, sample_size, v);
    status.wait();
  }
  return v;
}
std::vector<GraphNode> GraphPyClient::pull_graph_list(std::string name,
                                                      int server_index,
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
}
}
