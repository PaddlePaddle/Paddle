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

#pragma once

#include "paddle/fluid/operators/collective/hierarchical_hccl/hierarchical_hccl_types.h"
#include "paddle/fluid/operators/collective/hierarchical_hccl/impl/hierarchical_hccl_impl.h"

#include <memory>
#include <unordered_map>

namespace paddle {
namespace operators {

class HierarchicalBackend : public paddle::operators::HierarchicalHccl {
 public:
  std::string name() noexcept { return "hierarchical"; }

  HierarchicalBackend(
      HierarchicalHcclInitConfig init_config,
      std::shared_ptr<paddle::operators::rendezvous::BRPCStore> brpc_store)
      : _brpc_store(brpc_store) {
    _init_config = init_config;
  }

  HierarchicalHcclResult init_comm_global(
      const HierarchicalHcclUniqueId *unique_id, const int rank_count,
      const int my_rank, int my_device_id) override;

  HierarchicalHcclResult destroy_comm_global(
      const HierarchicalHcclUniqueId *unique_id) override;

  HierarchicalHcclResult all_reduce(
      const void *sendbuff, void *recvbuff, size_t count,
      HierarchicalHcclDataType data_type, HierarchicalHcclReductionOp op,
      HierarchicalHcclCommGroupIdType group_id,
      HierarchicalHcclRuntimeStream stream = nullptr) override;

  HierarchicalHcclResult broadcast(
      const void *sendbuff, void *recvbuff, size_t count,
      HierarchicalHcclDataType data_type, int root,
      HierarchicalHcclCommGroupIdType group_id,
      HierarchicalHcclRuntimeStream stream = nullptr) override;

  HierarchicalHcclResult reduce_scatter(
      const void *sendbuff, void *recvbuff, size_t recv_count,
      HierarchicalHcclDataType data_type, HierarchicalHcclReductionOp op,
      HierarchicalHcclCommGroupIdType group_id,
      HierarchicalHcclRuntimeStream stream = nullptr) override;

  HierarchicalHcclResult all_gather(
      const void *sendbuff, void *recvbuff, size_t send_count,
      HierarchicalHcclDataType data_type,
      HierarchicalHcclCommGroupIdType group_id,
      HierarchicalHcclRuntimeStream stream = nullptr) override;

  HierarchicalHcclResult memcpy(void *dst, void *src, size_t count,
                                HierarchicalHcclDataType data_type,
                                int type) override;

 protected:
  // Don't do anything in gen_unique_id, defer gen_unique_id task to
  // init_comm_global. This is required because some libraries doesn't support
  // consecutive gen unique ids. For example, it's an error to call: gen_id ->
  // gen_id -> init -> init in hccl, however, it's perfectly okay to call
  // gen_id
  // -> init -> gen_id -> init. So, we have to defer
  HierarchicalHcclResult gen_unique_id(
      HierarchicalHcclUniqueId *unique_id) noexcept override;

  int layer_count() {
    int max_layer = 0;
    for (auto i = 0; i < _init_config.layer_count; ++i) {
      max_layer = std::max(max_layer, _init_config.layers[i].level);
    }
    return max_layer + 1;
  }

 private:
  int _rank_count;
  int _my_rank;
  HierarchicalHcclInitConfig _init_config;

  // layer -> HierarchicalHccl map
  // layer start from 0 (lowest) to `layer_count - 1`(highest)
  // each layer is managed by a single `paddle::operators::HierarchicalHccl`
  // instance
  // For example, for 2048 card training in Ascend cluster,
  // layer 0 will be managed by HcclAdapter (card 0 to card 1023, card 1024 to
  // card 2047) layer 1 will be managed by GlooAdapter (card 0 and card 1024)
  std::unordered_map<uint32_t,
                     std::shared_ptr<paddle::operators::HierarchicalHccl>>
      _layered_comm_map;
  // layered groups is used to record which group it's in
  // for example card 0 will be in group 0 in layer 0 and group 0 in layer 1
  // where card 2047 will be in group 0 in layer 0 and not in layer 1
  std::unordered_map<uint32_t, uint32_t> _layer_index_map;

  // brpc_store used to bootstrap unique ids
  std::shared_ptr<paddle::operators::rendezvous::BRPCStore> _brpc_store;

  DISALLOW_COPY_AND_ASSIGN(HierarchicalBackend);
};

bool get_local_rank_in_layer(const int my_rank,
                             const HierarchicalHcclLayerConfig *layer_config,
                             int *local_rank, int *local_count,
                             int *group_start);

}  // namespace operators
}  // namespace paddle
