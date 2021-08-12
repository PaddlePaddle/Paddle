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

#include "paddle/fluid/operators/collective/hierarchical_hccl/hierarchical_hccl.h"
#include "paddle/fluid/operators/collective/hierarchical_hccl/hierarchical_hccl_types.h"
#include "paddle/fluid/operators/collective/hierarchical_hccl/impl/rendezvous/brpc_store.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {

class HierarchicalHccl {
 public:
  HierarchicalHccl() : _level(0) {}
  virtual ~HierarchicalHccl() = default;

 public:
  virtual std::string name() noexcept = 0;

  void set_level(int level) { _level = level; }

  int level() { return _level; }

  HierarchicalHcclResult gen_unique_id(
      HierarchicalHcclUniqueId *unique_id, std::string prefix,
      const int my_rank,
      std::shared_ptr<paddle::operators::rendezvous::BRPCStore>
          brpc_store) noexcept {
    HierarchicalHcclResult result = HierarchicalHcclResult::SUCCESS;
    std::string unique_id_key =
        "unique_id-" + name() + "-" + prefix + "-" + std::to_string(level());
    if (my_rank == 0) {
      result = gen_unique_id(unique_id);
      brpc_store->set(
          unique_id_key,
          std::string(unique_id->internal, HIERARCHICALHCCL_UNIQUE_ID_BYTES));
    } else {
      std::string data = brpc_store->get(unique_id_key);
      std::memcpy(unique_id->internal, data.c_str(),
                  HIERARCHICALHCCL_UNIQUE_ID_BYTES);
    }
    return result;
  };

  virtual HierarchicalHcclResult init_comm_global(
      const HierarchicalHcclUniqueId *unique_id, const int rank_count,
      const int my_rank, int my_device_id) = 0;

  virtual HierarchicalHcclResult destroy_comm_global(
      const HierarchicalHcclUniqueId *unique_id) = 0;

  virtual HierarchicalHcclResult all_reduce(
      const void *sendbuff, void *recvbuff, size_t count,
      HierarchicalHcclDataType data_type, HierarchicalHcclReductionOp op,
      HierarchicalHcclCommGroupIdType group_id,
      HierarchicalHcclRuntimeStream stream = nullptr) = 0;

  virtual HierarchicalHcclResult broadcast(
      const void *sendbuff, void *recvbuff, size_t count,
      HierarchicalHcclDataType data_type, int root,
      HierarchicalHcclCommGroupIdType group_id,
      HierarchicalHcclRuntimeStream stream = nullptr) = 0;

  virtual HierarchicalHcclResult reduce_scatter(
      const void *sendbuff, void *recvbuff, size_t recv_count,
      HierarchicalHcclDataType data_type, HierarchicalHcclReductionOp op,
      HierarchicalHcclCommGroupIdType group_id,
      HierarchicalHcclRuntimeStream stream = nullptr) = 0;

  virtual HierarchicalHcclResult all_gather(
      const void *sendbuff, void *recvbuff, size_t send_count,
      HierarchicalHcclDataType data_type,
      HierarchicalHcclCommGroupIdType group_id,
      HierarchicalHcclRuntimeStream stream = nullptr) = 0;

  virtual HierarchicalHcclResult memcpy(void *dst, void *src, size_t count,
                                        HierarchicalHcclDataType data_type,
                                        int type) = 0;

  virtual HierarchicalHcclResult count_to_bytes(
      HierarchicalHcclDataType data_type, size_t count, size_t *bytes_size) {
    size_t data_type_size;
    PADDLE_ENFORCE_NPU_SUCCESS(to_HcclDataTypeSize(data_type, &data_type_size));
    *bytes_size = count * data_type_size;
    return SUCCESS;
  }

  HierarchicalHcclResult sync_stream(HierarchicalHcclCommGroupIdType group_id,
                                     HierarchicalHcclRuntimeStream stream) {
    PADDLE_ENFORCE_NPU_SUCCESS(aclrtSynchronizeStream(stream));
    return SUCCESS;
  }

 protected:
  HierarchicalHcclResult to_HcclDataTypeSize(HierarchicalHcclDataType data_type,
                                             size_t *data_type_size) {
    switch (data_type) {
      case HCCL_DATA_TYPE_INT8: /**< int8 */
        *data_type_size = 1;
        break;
      case HCCL_DATA_TYPE_INT32: /**< int32 */
        *data_type_size = 4;
        break;
      case HCCL_DATA_TYPE_FP16: /**< fp16 */
        *data_type_size = 2;
        break;
      case HCCL_DATA_TYPE_FP32: /**< fp32 */
        *data_type_size = 4;
        break;
      case HCCL_DATA_TYPE_INT64: /**< int64 */
        *data_type_size = 8;
        break;
      case HCCL_DATA_TYPE_UINT64: /**< uint64 */
        *data_type_size = 8;
        break;
      default:
        return INTERNAL_ERROR;
    }
    return SUCCESS;
  }

 protected:
  // actually generate unique_id
  virtual HierarchicalHcclResult gen_unique_id(
      HierarchicalHcclUniqueId *unique_id) noexcept = 0;

  int _level;
};

}  // namespace operators
}  // namespace paddle
