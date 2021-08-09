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

#define PACKET_SPLIT_SIZE (1 << 30)

namespace paddle {
namespace operators {

class HcclBackend : public paddle::operators::HierarchicalHccl {
 public:
  HcclBackend() {}

 public:
  std::string name() noexcept { return "hccl-adapter"; }

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
  HierarchicalHcclResult gen_unique_id(
      HierarchicalHcclUniqueId *unique_id) noexcept override;
  // allocate temporary memory
  HierarchicalHcclResult allocate(void **dst, size_t count,
                                  HierarchicalHcclDataType data_type);

 private:
  HcclComm comm;
};

}  // namespace operators
}  // namespace paddle
