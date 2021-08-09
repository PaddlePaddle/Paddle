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

#include "paddle/fluid/operators/collective/hierarchical_hccl/impl/hierarchical_backend.h"

#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/operators/collective/hierarchical_hccl/impl/factory.h"
#include "paddle/fluid/operators/collective/hierarchical_hccl/impl/rendezvous/brpc_store.h"

#include <iostream>

namespace paddle {
namespace operators {

HierarchicalHcclResult HierarchicalBackend::gen_unique_id(
    HierarchicalHcclUniqueId *unique_id) noexcept {
  // set to 0
  memset(unique_id->internal, 0, HIERARCHICALHCCL_UNIQUE_ID_BYTES);
  return HierarchicalHcclResult::SUCCESS;
}

HierarchicalHcclResult HierarchicalBackend::init_comm_global(
    const HierarchicalHcclUniqueId *unique_id, const int rank_count,
    const int my_rank, int my_device_id) {
  _rank_count = rank_count;
  _my_rank = my_rank;
  VLOG(1) << "Begin init global comm: "
          << "rank_count = " << _rank_count << "; my_rank = " << _my_rank
          << "; layer_count = " << _init_config.layer_count;
  for (auto i = 0; i < _init_config.layer_count; ++i) {
    HierarchicalHcclLayerConfig layer = _init_config.layers[i];

    int local_rank = 0;
    int local_count = 0;
    int group_start = 0;
    if (get_local_rank_in_layer(my_rank, &layer, &local_rank, &local_count,
                                &group_start)) {
      VLOG(1) << "layer: [" << i << "] backend: [" << layer.backend << "] "
              << "my_rank[" << my_rank << "] in layer [" << i
              << "], group_start[" << group_start << "]";
      // HierarchicalHcclLayerConfig config;
      // config.backend = layer.backend;
      // FIXME - support nested config?
      std::shared_ptr<paddle::operators::HierarchicalHccl> impl;
      impl.reset(paddle::operators::HierarchicalHcclFactory::create(
          layer, _brpc_store));
      impl->set_level(layer.level);
      HierarchicalHcclUniqueId id;
      PADDLE_ENFORCE_NPU_SUCCESS(impl->gen_unique_id(&id, std::to_string(i).c_str(), local_rank,
                                _brpc_store));
      PADDLE_ENFORCE_NPU_SUCCESS(impl->init_comm_global(&id, local_count, local_rank, local_rank));
      VLOG(1) << "successfully initialized comm";
      _layered_comm_map[layer.level] = impl;
      _layer_index_map[layer.level] = i;
      VLOG(1) << "created hierarchical hccl[" << layer.backend << "] for layer["
              << layer.level << "] at index[" << i << "]";
    }
  }
  VLOG(1) << "Init global comm successfully! ";
  return HierarchicalHcclResult::SUCCESS;
}

HierarchicalHcclResult HierarchicalBackend::destroy_comm_global(
    const HierarchicalHcclUniqueId *unique_id) {
  for (auto layer = 0; layer < layer_count(); ++layer) {
    if (_layered_comm_map.find(layer) != _layered_comm_map.end()) {
      std::shared_ptr<paddle::operators::HierarchicalHccl> impl =
          _layered_comm_map[layer];
      PADDLE_ENFORCE_NPU_SUCCESS(impl->destroy_comm_global(nullptr));
    }
  }
  return HierarchicalHcclResult::SUCCESS;
}

// layered all reduce
// example setup: 4 node, 2 layers:
//   layer 0:  [0(0), 1(1)], [2(0), 3(1)]
//   layer 1:  [0(0), 2(1)]
// where 2(0) means rank 2 in global rank and rank 0 in local rank
// We perform all reduce in the following steps:
//   step 1: (concurrently) perform all reduce in [0, 1] and [2, 3], both with
//   local rank [(0), (1)] step 2: perform all reduce in [0(0), 2(1)], now the
//   result is stored in `recvbuff` step 3: perform broadcast in [0(0), 2(1)]
//   with root[(0)] in-place step 4: (concurrently) perform broadcast in [0, 1]
//   and [2, 3], both with root[(0)] in-place
// Implementation note: we have to sync stream after each step.
HierarchicalHcclResult HierarchicalBackend::all_reduce(
    const void *sendbuff, void *recvbuff, size_t count,
    HierarchicalHcclDataType data_type, HierarchicalHcclReductionOp op,
    HierarchicalHcclCommGroupIdType group_id,
    HierarchicalHcclRuntimeStream stream) {
  VLOG(3) << "First we need some allreduce operations!";
  for (auto layer = 0; layer < layer_count(); ++layer) {
    if (_layered_comm_map.find(layer) != _layered_comm_map.end()) {
      VLOG(3) << "conduct allreduce operations in layer [" << layer << "]";
      std::shared_ptr<paddle::operators::HierarchicalHccl> impl =
          _layered_comm_map[layer];
      if (layer == 0) {
        PADDLE_ENFORCE_NPU_SUCCESS(impl->all_reduce(sendbuff, recvbuff, count, data_type, op,
                               group_id,
                               stream));
        // TODO(liuwei88) : we may need to remove this in the future.
        // PADDLE_ENFORCE_NPU_SUCCESS(impl->sync_stream(group_id, stream) ==
        // HierarchicalHcclResult::SUCCESS);
      } else {
        // TODO(liuwei88) : if we use different backend and have
        // differen mem type, we
        //        have to tackle this situation in the furture
        PADDLE_ENFORCE_NPU_SUCCESS(impl->all_reduce(reinterpret_cast<void *>(recvbuff),
                               reinterpret_cast<void *>(recvbuff), count,
                               data_type, op, group_id,
                               stream));
      }
    }
  }

  VLOG(3) << "Secondly we need some broadcast operations!";
  for (auto layer = layer_count() - 2; layer >= 0; --layer) {
    if (_layered_comm_map.find(layer) != _layered_comm_map.end()) {
      VLOG(3) << "conduct broadcast operations in layer [" << layer << "]";
      std::shared_ptr<paddle::operators::HierarchicalHccl> impl =
          _layered_comm_map[layer];
      // note we broadcast in-place
      PADDLE_ENFORCE_NPU_SUCCESS(impl->broadcast(recvbuff, recvbuff, count, data_type, 0, group_id,
                            stream));
      // TODO(liuwei88) : we may need to remove this in the future.
      // PADDLE_ENFORCE_NPU_SUCCESS(impl->sync_stream(group_id, stream) ==
      // HierarchicalHcclResult::SUCCESS);
    }
  }

  VLOG(3) << "Finally we have finish all operations for this allreduce!";
  return HierarchicalHcclResult::SUCCESS;
}

// layered broadcast
// example setup: 4 node, 2 layers:
//   layer 0:  [0(0), 1(1)], [2(0), 3(1)]
//   layer 1:  [0(0), 2(1)]
// where 2(0) means rank 2 in global rank and rank 0 in local rank
// We perform broadcast in the following steps:
//   step 1: perform broadcast in the group which root is in from layer 0
//   step 2: perform broadcast in layer 1
//   step 3: perform broadcast in the other groups
// Implementation note: we have to sync stream after each step.
HierarchicalHcclResult HierarchicalBackend::broadcast(
    const void *sendbuff, void *recvbuff, size_t count,
    HierarchicalHcclDataType data_type, int root,
    HierarchicalHcclCommGroupIdType group_id,
    HierarchicalHcclRuntimeStream stream) {
  std::unordered_map<uint32_t, uint32_t> root_group_index_map;
  std::unordered_map<uint32_t, uint32_t> root_local_rank_map;
  VLOG(3) << "start calculate ranks for root " << root;
  for (auto i = 0; i < _init_config.layer_count; ++i) {
    HierarchicalHcclLayerConfig layer = _init_config.layers[i];
    int local_rank = 0;
    int local_count = 0;
    int group_start = 0;
    if (get_local_rank_in_layer(root, &layer, &local_rank, &local_count,
                                &group_start)) {
      root_group_index_map[layer.level] = i;
      root_local_rank_map[layer.level] = local_rank;
      VLOG(3) << "layer[" << i << "], level[" << layer.level << "], local rank["
              << local_rank << "]";

      // FIXME: we require first node in range to be the leader of the
      // higher layers
      root = group_start;
    }
  }
  VLOG(3) << "end calculate ranks for root " << root;

  for (auto layer = 0; layer < layer_count() - 1; ++layer) {
    VLOG(1) << "forward pass, layer " << layer;
    if (_layered_comm_map.find(layer) != _layered_comm_map.end() &&
        root_group_index_map[layer] == _layer_index_map[layer]) {
      std::shared_ptr<paddle::operators::HierarchicalHccl> impl =
          _layered_comm_map[layer];
      VLOG(3) << "perform broadcast on layer[" << layer << "], from root["
              << root << "], with local rank[" << root_local_rank_map[layer]
              << "]";
      PADDLE_ENFORCE_NPU_SUCCESS(impl->broadcast(sendbuff, recvbuff, count, data_type,
                            root_local_rank_map[layer], group_id,
                            stream));
      PADDLE_ENFORCE_NPU_SUCCESS(impl->sync_stream(group_id, stream));
    } else {
      VLOG(3) << "no need to perform forward pass for layer " << layer;
    }
    VLOG(3) << "finish forward pass, layer " << layer;
  }

  // top-most layer
  auto layer = layer_count() - 1;
  if (_layered_comm_map.find(layer) != _layered_comm_map.end()) {
    std::shared_ptr<paddle::operators::HierarchicalHccl> impl =
        _layered_comm_map[layer];
    VLOG(3) << "perform top most broadcast on layer[" << layer
            << "], from root[" << root << "], with local rank["
            << root_local_rank_map[layer] << "]";
    // TODO(liuwei88) : if we use different backend and have differen mem
    // type, we
    //        have to tackle this situation in the furture
    PADDLE_ENFORCE_NPU_SUCCESS(impl->broadcast(reinterpret_cast<void *>(recvbuff),
                          reinterpret_cast<void *>(recvbuff), count, data_type,
                          root_local_rank_map[layer], group_id,
                          stream) );
    PADDLE_ENFORCE_NPU_SUCCESS(impl->sync_stream(group_id, stream));
  } else {
    VLOG(3) << "no need to perform broadcast for top-most layer " << layer;
  }

  for (auto layer = layer_count() - 2; layer >= 0; --layer) {
    VLOG(3) << "backward pass, layer " << layer;
    if (_layered_comm_map.find(layer) != _layered_comm_map.end() &&
        root_group_index_map[layer] != _layer_index_map[layer]) {
      std::shared_ptr<paddle::operators::HierarchicalHccl> impl =
          _layered_comm_map[layer];
      // FIXME: now we assue higher level group is composed of all 0 nodes
      // from lower level group
      VLOG(3) << "perform broadcast on layer[" << layer << "], from root["
              << root << "], with local rank[0]";
      PADDLE_ENFORCE_NPU_SUCCESS(impl->broadcast(recvbuff, recvbuff, count, data_type, 0, group_id,
                            stream));
      PADDLE_ENFORCE_NPU_SUCCESS(impl->sync_stream(group_id, stream));
    } else {
      VLOG(3) << "no need to perform backward pass for layer " << layer;
    }
  }

  return HierarchicalHcclResult::SUCCESS;
}

HierarchicalHcclResult HierarchicalBackend::reduce_scatter(
    const void *sendbuff, void *recvbuff, size_t recv_count,
    HierarchicalHcclDataType data_type, HierarchicalHcclReductionOp op,
    HierarchicalHcclCommGroupIdType group_id,
    HierarchicalHcclRuntimeStream stream) {
  throw std::runtime_error("unsupported operation: reduce_scatter");
}

// layered all gather
// example setup: 4 node, 2 layers:
//   layer 0:  [0(0), 1(1)], [2(0), 3(1)]
//   layer 1:  [0(0), 2(1)]
// where 2(0) means rank 2 in global rank and rank 0 in local rank
// We perform all gather in the following steps:
//   step 1: clear the recvbuff and copy sendbuff to recvbuff in related positon
//   step 2: conduct allreduce sum opratiton in place to achieve all gather
// Implementation note: we have to sync stream after each step.
HierarchicalHcclResult HierarchicalBackend::all_gather(
    const void *sendbuff, void *recvbuff, size_t send_count,
    HierarchicalHcclDataType data_type,
    HierarchicalHcclCommGroupIdType group_id,
    HierarchicalHcclRuntimeStream stream) {
  // First, we should clear the recvbuff and copy sendbuff to recvbuff in
  // related positon
  VLOG(3) << "First, we should clear the recvbuff and cmp sendbuff to "
             "recvbuff in related positon!";
  size_t data_type_size = 0;
  size_t send_bytes_size = 0;

  PADDLE_ENFORCE_NPU_SUCCESS(to_HcclDataTypeSize(data_type, &data_type_size));
  PADDLE_ENFORCE_NPU_SUCCESS(count_to_bytes(data_type, send_count, &send_bytes_size));

  size_t maxCount = send_bytes_size * _rank_count;
  PADDLE_ENFORCE_NPU_SUCCESS(aclrtMemset(recvbuff, maxCount, 0, maxCount));

  void *tmp_src = reinterpret_cast<void *>(reinterpret_cast<char *>(recvbuff) +
                                           _my_rank * send_bytes_size);
  PADDLE_ENFORCE_NPU_SUCCESS(aclrtMemcpy(tmp_src, send_bytes_size, sendbuff,
                                         send_bytes_size,
                                         ACL_MEMCPY_DEVICE_TO_DEVICE));

  // Second, we should conduct allreduce sum opratiton in place to achieve all
  // gather
  VLOG(3) << "Second, we should conduct allreduce sum opratiton in place to "
             "achieve all gather";
  PADDLE_ENFORCE_NPU_SUCCESS(all_reduce(recvbuff, recvbuff, _rank_count * send_count, data_type, HCCL_REDUCE_SUM,
                   group_id, stream) );
  return HierarchicalHcclResult::SUCCESS;
}

HierarchicalHcclResult HierarchicalBackend::memcpy(
    void *dst, void *src, size_t count, HierarchicalHcclDataType data_type,
    int type) {
  throw std::runtime_error("unsupported operation: memcpy");
}

// get_local_rank_in_layer calculates local rank with (global) my_rank and
// layer_config. This function returns if the member is in the group. If the
// member is in the group, local_rank and local_count will be set, otherwise
// these 2 values will be undetermined.
bool get_local_rank_in_layer(const int my_rank,
                             const HierarchicalHcclLayerConfig *layer_config,
                             int *local_rank, int *local_count,
                             int *group_start) {
  if (layer_config->member_type == RANGE) {
    *local_count =
        layer_config->members.range->end - layer_config->members.range->start;
    VLOG(1) << "layer range, start[" << layer_config->members.range->start
            << "], end[" << layer_config->members.range->end
            << "], local_count[" << *(local_count) << "]";
    PADDLE_ENFORCE_GT(*local_count, 0);

    *local_rank = my_rank - layer_config->members.range->start;
    *group_start = layer_config->members.range->start;
    return *local_rank >= 0 && *local_rank < *local_count;
  } else if (layer_config->member_type == RANK_LIST) {
    *local_count = layer_config->members.members->rank_count;
    VLOG(1) << "layer ranks, count["
            << layer_config->members.members->rank_count << "], local_count["
            << *local_count << "]";
    for (auto idx = 0; idx < layer_config->members.members->rank_count; ++idx) {
      if (idx == 0) {
        *group_start = layer_config->members.members->ranks[0];
      }
      if (layer_config->members.members->ranks[idx] == my_rank) {
        *local_rank = idx;
        return true;
      }
    }
  } else {
    LOG(ERROR) << "unsupported member_type: " << layer_config->member_type;
    throw std::runtime_error("unsupported member_type");
  }
  return false;
}

}  // namespace operators
}  // namespace paddle
