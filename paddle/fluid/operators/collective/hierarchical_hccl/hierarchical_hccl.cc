/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/collective/hierarchical_hccl/hierarchical_hccl.h"

#include <thread>

#include "paddle/fluid/operators/collective/hierarchical_hccl/impl/factory.h"
#include "paddle/fluid/operators/collective/hierarchical_hccl/impl/rendezvous/brpc_store.h"
#include "paddle/fluid/operators/collective/hierarchical_hccl/impl/rendezvous/rendezvous_service.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {

void new_string_split(const std::string &str, char sep,
                      std::vector<std::string> *pieces,
                      bool ignore_null = true) {
  pieces->clear();
  if (str.empty()) {
    if (!ignore_null) {
      pieces->push_back(str);
    }
    return;
  }
  size_t pos = 0;
  size_t next = str.find(sep, pos);
  while (next != std::string::npos) {
    pieces->push_back(str.substr(pos, next - pos));
    pos = next + 1;
    next = str.find(sep, pos);
  }
  if (!str.substr(pos).empty()) {
    pieces->push_back(str.substr(pos));
  }
}

#ifndef LEAVE_SERVER_ALONE
// we need to construct a map for brpc server in rank 0
static std::unordered_map<std::string, std::shared_ptr<brpc::Server>>
    hierarchical_hccl_rpc_server_map;
static std::unordered_map<
    std::string,
    std::shared_ptr<paddle::operators::rendezvous::RendezvousServiceImpl>>
    impl_map;
#endif

static std::mutex server_thread_mutex;

// an unordered map to store unique id by group id
static std::unordered_map<std::string,
                          std::shared_ptr<HierarchicalHcclUniqueId>>
    unique_id_map;
// an unordered map to store hierarchical hccl instance by group id
static std::unordered_map<std::string,
                          std::shared_ptr<paddle::operators::HierarchicalHccl>>
    hierarchical_hccl_instance_map;
// an unordered map to store brpc store instance by group id
static std::unordered_map<
    std::string, std::shared_ptr<paddle::operators::rendezvous::BRPCStore>>
    brpc_store_map;

HierarchicalHcclResult make_range_layer_config(
    HierarchicalHcclLayerConfig *layer, HierarchicalHcclBackend backend,
    HierarchicalHcclRank start, HierarchicalHcclRank end, int level,
    HierarchicalHcclScope scope) {
  layer->member_type = RANGE;
  layer->members.range = new HierarchicalHcclMemberRange();
  layer->members.range->start = start;
  layer->members.range->end = end;
  layer->backend = backend;
  layer->level = level;
  layer->scope = scope;

  return SUCCESS;
}
HierarchicalHcclResult make_rangelist_layer_config(
    HierarchicalHcclLayerConfig *layer, HierarchicalHcclBackend backend,
    const std::vector<HierarchicalHcclRank> &rank_list, int level,
    HierarchicalHcclScope scope) {
  PADDLE_ENFORCE_GE(rank_list.size(), 0,
                    paddle::platform::errors::InvalidArgument(
                        "rank list size should more than 0"));

  layer->member_type = RANK_LIST;
  layer->members.members = new HierarchicalHcclRankList();
  layer->members.members->rank_count = rank_list.size();
  layer->members.members->ranks =
      new HierarchicalHcclRank[layer->members.members->rank_count];
  for (size_t i = 0; i < rank_list.size(); ++i) {
    layer->members.members->ranks[i] = rank_list[i];
  }
  layer->backend = backend;
  layer->level = level;
  layer->scope = scope;
  return SUCCESS;
}

HierarchicalHcclResult make_layer_config(
    HierarchicalHcclInitConfig *init_config,
    HierarchicalHcclCommGroupIdType group_id, const int rank_count,
    const int split_index) {
  std::string comm_group_id = std::string(group_id);

  PADDLE_ENFORCE_GE(split_index, 0,
                    paddle::platform::errors::InvalidArgument(
                        "split_index should be great or equal than (%d)!", 0));

  PADDLE_ENFORCE_LE(
      split_index, rank_count - 1,
      paddle::platform::errors::InvalidArgument(
          "split_index should not be great than (%d)!", rank_count - 1));

  init_config->scope = comm_group_id;

  // we try to use unified backend whatever in one-layer config
  // or two-layered config
  HierarchicalHcclBackend layer_backend;
  layer_backend = "hccl-adapter";

  if (split_index == 0) {
    init_config->backend = layer_backend;
  } else {
    // we try to construt a layer config
    init_config->backend = "hierarchical";
    init_config->layer_count = split_index + 1;
    init_config->layers =
        new HierarchicalHcclLayerConfig[init_config->layer_count];

    PADDLE_ENFORCE_EQ(rank_count % split_index, 0,
                      paddle::platform::errors::InvalidArgument(
                          "rank count (%d) %% split_index(%d) should be 0!",
                          rank_count, split_index));

    int layer_1_count = rank_count / split_index;
    std::string scope;
    // the first layer
    for (int ilayer_1 = 0; ilayer_1 < split_index; ilayer_1++) {
      scope = comm_group_id + layer_backend + std::to_string(0) + "_" +
              std::to_string(ilayer_1);
      make_range_layer_config(init_config->layers + ilayer_1, layer_backend,
                              layer_1_count * ilayer_1,
                              layer_1_count * (ilayer_1 + 1), 0, scope);
    }

    // the second layer
    scope = comm_group_id + layer_backend + std::to_string(1) + "_" +
            std::to_string(0);
    std::vector<HierarchicalHcclRank> rank_list;
    for (int ilayer_1 = 0; ilayer_1 < split_index; ilayer_1++) {
      rank_list.push_back(layer_1_count * ilayer_1);
    }
    make_rangelist_layer_config(init_config->layers + split_index,
                                layer_backend, rank_list, 1, scope);
  }
  return SUCCESS;
}

//  a help fucntion to get pure comm group id
//  for example, input comm_group_id = "group_1"
//  we want to return "group", because 1 means rank id
std::string get_pure_comm_group_id(std::string comm_group_id) {
  std::string::size_type position = comm_group_id.find_last_of("_");

  PADDLE_ENFORCE_NE(position, comm_group_id.npos,
                    paddle::platform::errors::InvalidArgument(
                        "need ${name}_${rankid}$ mode to name comm group id"));

  return comm_group_id.substr(0, position);
}

std::string gen_global_group_id(std::string comm_group_id) {
  return "global_single_layer_" + comm_group_id;
}

std::shared_ptr<paddle::operators::HierarchicalHccl> get_instance(
    HierarchicalHcclCommGroupIdType group_id, bool is_global = false) {
  std::lock_guard<std::mutex> lock(server_thread_mutex);
  std::string comm_group_id = std::string(group_id);

// force set is_global = false in ascend
#if defined(PADDLE_WITH_HIERARCHICAL_HCCL)
  is_global = false;
#endif

  // we use this to select group for different op
  if (is_global) {
    comm_group_id = gen_global_group_id(comm_group_id);
    if (hierarchical_hccl_instance_map.find(comm_group_id) ==
        hierarchical_hccl_instance_map.end()) {
      VLOG(3) << "Try to find global group [" << comm_group_id
              << "], but failed!"
              << "We just use orgin group [" << std::string(group_id) << "]!"
              << "See split index == 0 ?";
      comm_group_id = std::string(group_id);
    }
  }

  PADDLE_ENFORCE_NE(hierarchical_hccl_instance_map.find(comm_group_id),
                    hierarchical_hccl_instance_map.end(),
                    paddle::platform::errors::InvalidArgument(
                        "can not find HierarchicalHccl instance for group [%s]",
                        comm_group_id.c_str()));

  std::shared_ptr<paddle::operators::HierarchicalHccl> instance;
  instance = hierarchical_hccl_instance_map[comm_group_id];
  return instance;
}
#ifndef LEAVE_SERVER_ALONE
HierarchicalHcclResult construct_brpc_server(std::string comm_group_id,
                                             std::string endpoint) {
  std::vector<std::string> parts;
  paddle::operators::new_string_split(endpoint, ':', &parts);
  PADDLE_ENFORCE_EQ(
      parts.size(), 2,
      paddle::platform::errors::InvalidArgument(
          "endpoint should be format as ip:port, but [%s]", endpoint.c_str()));

  VLOG(1) << "Try to started a new HierarchicalHccl RPC server...";

  PADDLE_ENFORCE_EQ(
      hierarchical_hccl_rpc_server_map.find(comm_group_id),
      hierarchical_hccl_rpc_server_map.end(),
      paddle::platform::errors::InvalidArgument(
          "Find a repeated rpc server for group [%s]", comm_group_id.c_str()));

  PADDLE_ENFORCE_EQ(
      impl_map.find(comm_group_id), impl_map.end(),
      paddle::platform::errors::InvalidArgument(
          "Find a repeated rpc impl for group [%s]", comm_group_id.c_str()));

  std::shared_ptr<brpc::Server> hierarchical_hccl_rpc_server =
      std::make_shared<brpc::Server>();
  std::shared_ptr<paddle::operators::rendezvous::RendezvousServiceImpl> impl =
      std::make_shared<paddle::operators::rendezvous::RendezvousServiceImpl>();

  PADDLE_ENFORCE_EQ(
      hierarchical_hccl_rpc_server->AddService(impl.get(),
                                               brpc::SERVER_DOESNT_OWN_SERVICE),
      0, paddle::platform::errors::InvalidArgument("Add rpc server failed!"));

  VLOG(1) << "Add rpc server successfully!";

  brpc::ServerOptions options;
  PADDLE_ENFORCE_EQ(
      hierarchical_hccl_rpc_server->Start(std::stoi(parts[1]), &options), 0,
      paddle::platform::errors::InvalidArgument("Start rpc server failed!"));

  VLOG(1) << "Started HierarchicalHccl RPC server at [" << endpoint << "] for ["
          << comm_group_id << "] successfully!";

  hierarchical_hccl_rpc_server_map[comm_group_id] =
      hierarchical_hccl_rpc_server;
  impl_map[comm_group_id] = impl;

  return SUCCESS;
}

HierarchicalHcclResult destroy_brpc_server(std::string comm_group_id) {
  if (hierarchical_hccl_rpc_server_map.find(comm_group_id) ==
      hierarchical_hccl_rpc_server_map.end()) {
    VLOG(1) << "No server need to be destroyed for [" << comm_group_id << "] !";
    return SUCCESS;
  }

  std::shared_ptr<brpc::Server> hierarchical_hccl_rpc_server =
      hierarchical_hccl_rpc_server_map[comm_group_id];
  std::shared_ptr<paddle::operators::rendezvous::RendezvousServiceImpl> impl =
      impl_map[comm_group_id];

  hierarchical_hccl_rpc_server->Stop(0);
  hierarchical_hccl_rpc_server->Join();
  impl->clear_group(comm_group_id);
  PADDLE_ENFORCE_EQ(
      hierarchical_hccl_rpc_server->RemoveService(impl.get()), 0,
      paddle::platform::errors::InvalidArgument(
          "remove rpc server for %s failed ", comm_group_id.c_str()));

  hierarchical_hccl_rpc_server_map.erase(comm_group_id);
  impl_map.erase(comm_group_id);

  VLOG(1) << "Destroyed HierarchicalHccl RPC server for [" << comm_group_id
          << "] sucessfully!";

  return SUCCESS;
}
#endif

HierarchicalHcclResult construct_hierarchical_hccl_group(
    const std::string &comm_group_id, int rank_count, int my_rank,
    int split_index, const std::string &endpoint) {
  //  construct brpc_store for each group, if existed, delete it
  PADDLE_ENFORCE_EQ(brpc_store_map.find(comm_group_id), brpc_store_map.end(),
                    paddle::platform::errors::InvalidArgument(
                        "repeated group_id %s ", comm_group_id.c_str()));

  std::shared_ptr<paddle::operators::rendezvous::BRPCStore> brpc_store =
      std::make_shared<paddle::operators::rendezvous::BRPCStore>(endpoint);
  brpc_store_map[comm_group_id] = brpc_store;
  VLOG(1) << "HierarchicalHccl RPC client has been started...";

  // create hierarchical hccl instance
  // fist, we should get a init config
  // second, construct a hierarchical hccl instance
  HierarchicalHcclInitConfig init_config;
  make_layer_config(&init_config, comm_group_id.c_str(), rank_count,
                    split_index);
  VLOG(1) << "HierarchicalHccl RPC config has been constructed ...";

  PADDLE_ENFORCE_EQ(hierarchical_hccl_instance_map.find(comm_group_id),
                    hierarchical_hccl_instance_map.end(),
                    paddle::platform::errors::InvalidArgument(
                        "repeated group_id %s ", comm_group_id.c_str()));

  std::shared_ptr<paddle::operators::HierarchicalHccl> instance;
  instance.reset(paddle::operators::HierarchicalHcclFactory::create(
      init_config, brpc_store));
  hierarchical_hccl_instance_map[comm_group_id] = instance;
  VLOG(1) << "HierarchicalHccl instance has been constructed ...";

  // create unique id
  PADDLE_ENFORCE_EQ(unique_id_map.find(comm_group_id), unique_id_map.end(),
                    paddle::platform::errors::InvalidArgument(
                        "repeated group_id %s ", comm_group_id.c_str()));

  std::shared_ptr<HierarchicalHcclUniqueId> unique_id =
      std::make_shared<HierarchicalHcclUniqueId>();
  unique_id_map[comm_group_id] = unique_id;

// we try to support single process multi device model, but only for GPU
#ifdef PADDLE_WITH_HIERARCHICAL_HCCL
  std::string pure_comm_group_id = comm_group_id;
#else
  std::string pure_comm_group_id = get_pure_comm_group_id(comm_group_id);
#endif
  PADDLE_ENFORCE_NPU_SUCCESS(instance->gen_unique_id(
      unique_id.get(), pure_comm_group_id, my_rank, brpc_store));

  VLOG(1) << "Finish gen unique id in group [" << comm_group_id << "]!";
  return SUCCESS;
}

HierarchicalHcclResult destroy_hierarchical_hccl_group(
    std::string comm_group_id) {
  VLOG(1) << "Try to Destroy comm [" << comm_group_id << "] !";
  PADDLE_ENFORCE_NPU_SUCCESS(
      get_instance(comm_group_id.c_str())->destroy_comm_global(nullptr));

  std::lock_guard<std::mutex> lock(server_thread_mutex);

  if (hierarchical_hccl_instance_map.find(comm_group_id) !=
      hierarchical_hccl_instance_map.end()) {
    hierarchical_hccl_instance_map.erase(comm_group_id);
  }
  if (unique_id_map.find(comm_group_id) != unique_id_map.end()) {
    unique_id_map.erase(comm_group_id);
  }
  if (brpc_store_map.find(comm_group_id) != brpc_store_map.end()) {
    brpc_store_map.erase(comm_group_id);
  }

#ifndef LEAVE_SERVER_ALONE
  PADDLE_ENFORCE_NPU_SUCCESS(destroy_brpc_server(comm_group_id));
#endif

  VLOG(1) << "Destroy comm [" << comm_group_id << "] successfully!";
  return SUCCESS;
}

HierarchicalHcclResult init_hierarchical_hccl_comm(const int rank_count,
                                                   const int my_rank,
                                                   int my_device_id,
                                                   std::string comm_group_id) {
  VLOG(1) << "Begin Init global comm in group [" << std::string(comm_group_id)
          << "]!";

  PADDLE_ENFORCE_NE(
      unique_id_map.find(comm_group_id), unique_id_map.end(),
      paddle::platform::errors::InvalidArgument(
          "can not find unique ID for group %s ", comm_group_id.c_str()));

  PADDLE_ENFORCE_NPU_SUCCESS(
      get_instance(comm_group_id.c_str())
          ->init_comm_global(unique_id_map[comm_group_id].get(), rank_count,
                             my_rank, my_device_id));

  VLOG(1) << "Successful: Init global comm in group ["
          << std::string(comm_group_id) << "]!";
  return SUCCESS;
}
HierarchicalHcclResult hierarchical_hccl_gen_unique_id(
    const int my_rank, const char *bootstrap_endpoint, const int rank_count,
    const int split_index, HierarchicalHcclCommGroupIdType group_id) {
  VLOG(1) << "Begin gen unique id in group [" << std::string(group_id) << "]!";
  std::lock_guard<std::mutex> lock(server_thread_mutex);
  std::string endpoint = std::string(bootstrap_endpoint);
  std::string comm_group_id = std::string(group_id);

#ifndef LEAVE_SERVER_ALONE
  // start service if needed
  if (my_rank == 0) {
    PADDLE_ENFORCE_NPU_SUCCESS(construct_brpc_server(comm_group_id, endpoint));
  } else {
    VLOG(1) << "No need to start HierarchicalHccl RPC server, rank id is not "
               "0, but "
            << my_rank << " ....";
  }
#endif

  // construct a normal group
  PADDLE_ENFORCE_NPU_SUCCESS(construct_hierarchical_hccl_group(
      comm_group_id, rank_count, my_rank, split_index, endpoint));

// construct a single layer group if needed
// for ascend, we dont construct it
#ifndef PADDLE_WITH_HIERARCHICAL_HCCL
  if (split_index != 0) {
    std::string global_group_id = gen_global_group_id(comm_group_id);
    PADDLE_ENFORCE_NPU_SUCCESS(construct_hierarchical_hccl_group(
        global_group_id, rank_count, my_rank, 0, endpoint));
  }
#endif

  return SUCCESS;
}

HierarchicalHcclResult hierarchical_hccl_init_comm_global(
    const int rank_count, const int my_rank, int my_device_id,
    HierarchicalHcclCommGroupIdType group_id) {
  std::string comm_group_id = std::string(group_id);
  PADDLE_ENFORCE_NPU_SUCCESS(init_hierarchical_hccl_comm(
      rank_count, my_rank, my_device_id, comm_group_id));
  // we need to init global group
  std::string global_comm_group_id = gen_global_group_id(comm_group_id);
  if (hierarchical_hccl_instance_map.find(global_comm_group_id) !=
      hierarchical_hccl_instance_map.end()) {
    PADDLE_ENFORCE_NPU_SUCCESS(init_hierarchical_hccl_comm(
        rank_count, my_rank, my_device_id, global_comm_group_id));
  }
  return SUCCESS;
}

HierarchicalHcclResult hierarchical_hccl_destroy_comm_global(
    HierarchicalHcclCommGroupIdType group_id) {
  std::string comm_group_id = std::string(group_id);
  PADDLE_ENFORCE_NPU_SUCCESS(destroy_hierarchical_hccl_group(comm_group_id));

  // we need to destroy global group
  std::string global_comm_group_id = gen_global_group_id(comm_group_id);
  if (hierarchical_hccl_instance_map.find(global_comm_group_id) !=
      hierarchical_hccl_instance_map.end()) {
    PADDLE_ENFORCE_NPU_SUCCESS(
        destroy_hierarchical_hccl_group(global_comm_group_id));
  }

  return SUCCESS;
}

HierarchicalHcclResult hierarchical_hccl_all_reduce(
    const void *sendbuff, void *recvbuff, size_t count,
    HierarchicalHcclDataType data_type, HierarchicalHcclReductionOp op,
    HierarchicalHcclCommGroupIdType group_id,
    HierarchicalHcclRuntimeStream stream) {
  return get_instance(group_id)->all_reduce(sendbuff, recvbuff, count,
                                            data_type, op, group_id, stream);
}

HierarchicalHcclResult hierarchical_hccl_broadcast(
    const void *sendbuff, void *recvbuff, size_t count,
    HierarchicalHcclDataType data_type, int root,
    HierarchicalHcclCommGroupIdType group_id,
    HierarchicalHcclRuntimeStream stream) {
  return get_instance(group_id, true)
      ->broadcast(sendbuff, recvbuff, count, data_type, root, group_id, stream);
}

HierarchicalHcclResult hierarchical_hccl_reduce_scatter(
    const void *sendbuff, void *recvbuff, size_t recv_count,
    HierarchicalHcclDataType data_type, HierarchicalHcclReductionOp op,
    HierarchicalHcclCommGroupIdType group_id,
    HierarchicalHcclRuntimeStream stream) {
  return get_instance(group_id, true)
      ->reduce_scatter(sendbuff, recvbuff, recv_count, data_type, op, group_id,
                       stream);
}

HierarchicalHcclResult hierarchical_hccl_all_gather(
    const void *sendbuff, void *recvbuff, size_t send_count,
    HierarchicalHcclDataType data_type,
    HierarchicalHcclCommGroupIdType group_id,
    HierarchicalHcclRuntimeStream stream) {
  return get_instance(group_id, true)
      ->all_gather(sendbuff, recvbuff, send_count, data_type, group_id, stream);
}

}  // namespace operators
}  // namespace paddle
