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

void GraphPyService::add_table_feat_conf(std::string table_name,
                                         std::string feat_name,
                                         std::string feat_dtype,
                                         int32_t feat_shape) {
  if (this->table_id_map.count(table_name)) {
    this->table_feat_conf_table_name.push_back(table_name);
    this->table_feat_conf_feat_name.push_back(feat_name);
    this->table_feat_conf_feat_dtype.push_back(feat_dtype);
    this->table_feat_conf_feat_shape.push_back(feat_shape);
  }
}

void add_graph_node(std::vector<uint64_t> node_ids,
                    std::vector<bool> weight_list) {}
void remove_graph_node(std::vector<uint64_t> node_ids) {}
void GraphPyService::set_up(std::string ips_str, int shard_num,
                            std::vector<std::string> node_types,
                            std::vector<std::string> edge_types) {
  set_shard_num(shard_num);
  set_num_node_types(node_types.size());

  for (size_t table_id = 0; table_id < node_types.size(); table_id++) {
    this->table_id_map[node_types[table_id]] = this->table_id_map.size();
  }
  for (size_t table_id = 0; table_id < edge_types.size(); table_id++) {
    this->table_id_map[edge_types[table_id]] = this->table_id_map.size();
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
  worker_ptr = std::shared_ptr<paddle::distributed::GraphBrpcClient>(
      (paddle::distributed::GraphBrpcClient*)
          paddle::distributed::PSClientFactory::create(worker_proto));
  worker_ptr->configure(worker_proto, dense_regions, _ps_env, client_id);
  worker_ptr->set_shard_num(get_shard_num());
}
void GraphPyServer::start_server(bool block) {
  std::string ip = server_list[rank];
  uint32_t port = std::stoul(port_list[rank]);
  ::paddle::distributed::PSParameter server_proto = this->GetServerProto();

  auto _ps_env = paddle::distributed::PaddlePSEnvironment();
  _ps_env.set_ps_servers(&this->host_sign_list,
                         this->host_sign_list.size());  // test
  pserver_ptr = std::shared_ptr<paddle::distributed::GraphBrpcServer>(
      (paddle::distributed::GraphBrpcServer*)
          paddle::distributed::PSServerFactory::create(server_proto));
  VLOG(0) << "pserver-ptr created ";
  std::vector<framework::ProgramDesc> empty_vec;
  framework::ProgramDesc empty_prog;
  empty_vec.push_back(empty_prog);
  pserver_ptr->configure(server_proto, _ps_env, rank, empty_vec);
  pserver_ptr->start(ip, port);
  std::condition_variable* cv_ = pserver_ptr->export_cv();
  if (block) {
    std::mutex mutex_;
    std::unique_lock<std::mutex> lock(mutex_);
    cv_->wait(lock);
  }
}
::paddle::distributed::PSParameter GraphPyServer::GetServerProto() {
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

  for (auto& tuple : this->table_id_map) {
    VLOG(0) << " make a new table " << tuple.second;
    ::paddle::distributed::TableParameter* sparse_table_proto =
        downpour_server_proto->add_downpour_table_param();
    std::vector<std::string> feat_name;
    std::vector<std::string> feat_dtype;
    std::vector<int32_t> feat_shape;
    for (size_t i = 0; i < this->table_feat_conf_table_name.size(); i++) {
      if (tuple.first == table_feat_conf_table_name[i]) {
        feat_name.push_back(table_feat_conf_feat_name[i]);
        feat_dtype.push_back(table_feat_conf_feat_dtype[i]);
        feat_shape.push_back(table_feat_conf_feat_shape[i]);
      }
    }
    std::string table_type;
    if (tuple.second < this->num_node_types) {
      table_type = "node";
    } else {
      table_type = "edge";
    }

    GetDownpourSparseTableProto(sparse_table_proto, tuple.second, tuple.first,
                                table_type, feat_name, feat_dtype, feat_shape);
  }

  return server_fleet_desc;
}

::paddle::distributed::PSParameter GraphPyClient::GetWorkerProto() {
  ::paddle::distributed::PSParameter worker_fleet_desc;
  ::paddle::distributed::WorkerParameter* worker_proto =
      worker_fleet_desc.mutable_worker_param();

  ::paddle::distributed::DownpourWorkerParameter* downpour_worker_proto =
      worker_proto->mutable_downpour_worker_param();

  for (auto& tuple : this->table_id_map) {
    VLOG(0) << " make a new table " << tuple.second;
    ::paddle::distributed::TableParameter* worker_sparse_table_proto =
        downpour_worker_proto->add_downpour_table_param();
    std::vector<std::string> feat_name;
    std::vector<std::string> feat_dtype;
    std::vector<int32_t> feat_shape;
    for (size_t i = 0; i < this->table_feat_conf_table_name.size(); i++) {
      if (tuple.first == table_feat_conf_table_name[i]) {
        feat_name.push_back(table_feat_conf_feat_name[i]);
        feat_dtype.push_back(table_feat_conf_feat_dtype[i]);
        feat_shape.push_back(table_feat_conf_feat_shape[i]);
      }
    }
    std::string table_type;
    if (tuple.second < this->num_node_types) {
      table_type = "node";
    } else {
      table_type = "edge";
    }

    GetDownpourSparseTableProto(worker_sparse_table_proto, tuple.second,
                                tuple.first, table_type, feat_name, feat_dtype,
                                feat_shape);
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
    VLOG(0) << " make a new table " << tuple.second;
    ::paddle::distributed::TableParameter* sparse_table_proto =
        downpour_server_proto->add_downpour_table_param();
    std::vector<std::string> feat_name;
    std::vector<std::string> feat_dtype;
    std::vector<int32_t> feat_shape;
    for (size_t i = 0; i < this->table_feat_conf_table_name.size(); i++) {
      if (tuple.first == table_feat_conf_table_name[i]) {
        feat_name.push_back(table_feat_conf_feat_name[i]);
        feat_dtype.push_back(table_feat_conf_feat_dtype[i]);
        feat_shape.push_back(table_feat_conf_feat_shape[i]);
      }
    }
    std::string table_type;
    if (tuple.second < this->num_node_types) {
      table_type = "node";
    } else {
      table_type = "edge";
    }

    GetDownpourSparseTableProto(sparse_table_proto, tuple.second, tuple.first,
                                table_type, feat_name, feat_dtype, feat_shape);
  }

  return worker_fleet_desc;
}
void GraphPyClient::load_edge_file(std::string name, std::string filepath,
                                   bool reverse) {
  // 'e' means load edge
  std::string params = "e";
  if (reverse) {
    // 'e<' means load edges from $2 to $1
    params += "<";
  } else {
    // 'e>' means load edges from $1 to $2
    params += ">";
  }
  if (this->table_id_map.count(name)) {
    VLOG(0) << "loadding data with type " << name << " from " << filepath;
    uint32_t table_id = this->table_id_map[name];
    auto status =
        get_ps_client()->load(table_id, std::string(filepath), params);
    status.wait();
  }
}

void GraphPyClient::clear_nodes(std::string name) {
  if (this->table_id_map.count(name)) {
    uint32_t table_id = this->table_id_map[name];
    auto status = get_ps_client()->clear_nodes(table_id);
    status.wait();
  }
}

void GraphPyClient::add_graph_node(std::string name,
                                   std::vector<uint64_t>& node_ids,
                                   std::vector<bool>& weight_list) {
  if (this->table_id_map.count(name)) {
    uint32_t table_id = this->table_id_map[name];
    auto status =
        get_ps_client()->add_graph_node(table_id, node_ids, weight_list);
    status.wait();
  }
}

void GraphPyClient::remove_graph_node(std::string name,
                                      std::vector<uint64_t>& node_ids) {
  if (this->table_id_map.count(name)) {
    uint32_t table_id = this->table_id_map[name];
    auto status = get_ps_client()->remove_graph_node(table_id, node_ids);
    status.wait();
  }
}

void GraphPyClient::load_node_file(std::string name, std::string filepath) {
  // 'n' means load nodes and 'node_type' follows
  std::string params = "n" + name;
  if (this->table_id_map.count(name)) {
    uint32_t table_id = this->table_id_map[name];
    auto status =
        get_ps_client()->load(table_id, std::string(filepath), params);
    status.wait();
  }
}
std::vector<std::vector<std::pair<uint64_t, float>>>
GraphPyClient::batch_sample_neighboors(std::string name,
                                       std::vector<uint64_t> node_ids,
                                       int sample_size) {
  std::vector<std::vector<std::pair<uint64_t, float>>> v;
  if (this->table_id_map.count(name)) {
    uint32_t table_id = this->table_id_map[name];
    auto status =
        worker_ptr->batch_sample_neighboors(table_id, node_ids, sample_size, v);
    status.wait();
  }
  return v;
}

std::vector<uint64_t> GraphPyClient::random_sample_nodes(std::string name,
                                                         int server_index,
                                                         int sample_size) {
  std::vector<uint64_t> v;
  if (this->table_id_map.count(name)) {
    uint32_t table_id = this->table_id_map[name];
    auto status =
        worker_ptr->random_sample_nodes(table_id, server_index, sample_size, v);
    status.wait();
  }
  return v;
}

// (name, dtype, ndarray)
std::vector<std::vector<std::string>> GraphPyClient::get_node_feat(
    std::string node_type, std::vector<uint64_t> node_ids,
    std::vector<std::string> feature_names) {
  std::vector<std::vector<std::string>> v(
      feature_names.size(), std::vector<std::string>(node_ids.size()));
  if (this->table_id_map.count(node_type)) {
    uint32_t table_id = this->table_id_map[node_type];
    auto status =
        worker_ptr->get_node_feat(table_id, node_ids, feature_names, v);
    status.wait();
  }
  return v;
}

void GraphPyClient::set_node_feat(
    std::string node_type, std::vector<uint64_t> node_ids,
    std::vector<std::string> feature_names,
    const std::vector<std::vector<std::string>> features) {
  if (this->table_id_map.count(node_type)) {
    uint32_t table_id = this->table_id_map[node_type];
    auto status =
        worker_ptr->set_node_feat(table_id, node_ids, feature_names, features);
    status.wait();
  }
  return;
}

std::vector<FeatureNode> GraphPyClient::pull_graph_list(std::string name,
                                                        int server_index,
                                                        int start, int size,
                                                        int step) {
  std::vector<FeatureNode> res;
  if (this->table_id_map.count(name)) {
    uint32_t table_id = this->table_id_map[name];
    auto status = worker_ptr->pull_graph_list(table_id, server_index, start,
                                              size, step, res);
    status.wait();
  }
  return res;
}

void GraphPyClient::stop_server() {
  VLOG(0) << "going to stop server";
  std::unique_lock<std::mutex> lock(mutex_);
  if (stoped_) return;
  auto status = this->worker_ptr->stop_server();
  if (status.get() == 0) stoped_ = true;
}
void GraphPyClient::finalize_worker() { this->worker_ptr->finalize_worker(); }
}
}
