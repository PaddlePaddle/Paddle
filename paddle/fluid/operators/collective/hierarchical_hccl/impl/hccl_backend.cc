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

#include "paddle/fluid/operators/collective/hierarchical_hccl/impl/hccl_backend.h"

#include "paddle/fluid/platform/dynload/hccl.h"

#include <cstring>
#include <iostream>

namespace paddle {
namespace operators {

HierarchicalHcclResult HcclBackend::gen_unique_id(
    HierarchicalHcclUniqueId *unique_id) noexcept {
  HcclRootInfo hcclRootInfo;
  PADDLE_ENFORCE_NPU_SUCCESS(
      paddle::platform::dynload::HcclGetRootInfo(&hcclRootInfo));
  std::memcpy(unique_id->internal, hcclRootInfo.internal, HCCL_ROOT_INFO_BYTES);
  return SUCCESS;
}

HierarchicalHcclResult HcclBackend::init_comm_global(
    const HierarchicalHcclUniqueId *unique_id, const int rank_count,
    const int my_rank, int my_device_id) {
  HcclRootInfo hcclRootInfo;
  std::memcpy(hcclRootInfo.internal, unique_id->internal, HCCL_ROOT_INFO_BYTES);
  PADDLE_ENFORCE_NPU_SUCCESS(paddle::platform::dynload::HcclCommInitRootInfo(
      rank_count, &hcclRootInfo, my_rank, &comm));

  // trigger connection
  float *buff = nullptr;
  int count = 32;
  PADDLE_ENFORCE_NPU_SUCCESS(aclrtMalloc(reinterpret_cast<void **>(&buff),
                                         count * sizeof(float),
                                         ACL_MEM_MALLOC_NORMAL_ONLY));
  aclrtStream stream;
  PADDLE_ENFORCE_NPU_SUCCESS(aclrtCreateStream(&stream));
  PADDLE_ENFORCE_NPU_SUCCESS(paddle::platform::dynload::HcclAllReduce(
      buff, buff, count, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, comm, stream));
  PADDLE_ENFORCE_NPU_SUCCESS(paddle::platform::dynload::HcclBroadcast(
      buff, count, HCCL_DATA_TYPE_FP32, 0, comm, stream));
  PADDLE_ENFORCE_NPU_SUCCESS(aclrtSynchronizeStream(stream));
  PADDLE_ENFORCE_NPU_SUCCESS(aclrtDestroyStream(stream));
  PADDLE_ENFORCE_NPU_SUCCESS(aclrtFree(buff));
  VLOG(1) << "Trigger connection done!";
  return SUCCESS;
}

HierarchicalHcclResult HcclBackend::destroy_comm_global(
    const HierarchicalHcclUniqueId *unique_id) {
  PADDLE_ENFORCE_NPU_SUCCESS(paddle::platform::dynload::HcclCommDestroy(comm));
  return SUCCESS;
}

HierarchicalHcclResult HcclBackend::all_reduce(
    const void *sendbuff, void *recvbuff, size_t count,
    HierarchicalHcclDataType data_type, HierarchicalHcclReductionOp op,
    HierarchicalHcclCommGroupIdType group_id,
    HierarchicalHcclRuntimeStream stream) {
  size_t element_count = count;
  size_t element_byte = 0;
  size_t element_offset = 0;
  size_t split_element_count = 0;
  size_t split_trans_count = 0;
  char *split_trans_sendbuff = nullptr;
  char *split_trans_recvbuff = nullptr;

  PADDLE_ENFORCE_NPU_SUCCESS(to_HcclDataTypeSize(data_type, &element_byte));

  // Due to performance drop on Ascend when the size of
  // packet transferred is more than 1GB,
  // we split the transmission to multiple times.
  split_element_count = PACKET_SPLIT_SIZE / element_byte;
  while (element_offset < element_count) {
    split_trans_count = (element_offset + split_element_count) > element_count
                            ? (element_count - element_offset)
                            : split_element_count;
    size_t offset_size = element_offset * element_byte;
    split_trans_sendbuff =
        (reinterpret_cast<char *>(const_cast<void *>(sendbuff)) + offset_size);
    split_trans_recvbuff = (reinterpret_cast<char *>(recvbuff) + offset_size);
    PADDLE_ENFORCE_NPU_SUCCESS(paddle::platform::dynload::HcclAllReduce(
        reinterpret_cast<void *>(split_trans_sendbuff),
        reinterpret_cast<void *>(split_trans_recvbuff), split_trans_count,
        data_type, op, comm, stream));
    VLOG(3) << "Allreduce send [" << split_trans_count << "] elements!";
    element_offset += split_trans_count;
  }
  VLOG(3) << "Allreduce send all elements!";
  return SUCCESS;
}

HierarchicalHcclResult HcclBackend::broadcast(
    const void *sendbuff, void *recvbuff, size_t count,
    HierarchicalHcclDataType data_type, int root,
    HierarchicalHcclCommGroupIdType group_id,
    HierarchicalHcclRuntimeStream stream) {
  // Hccl only supports in-place allreduce
  // we need to copy data from sendbuff to recvbuff
  // if we have different buffers
  // and always use recvbuff for broadcast
  if (sendbuff != recvbuff) {
    PADDLE_ENFORCE_NPU_SUCCESS(memcpy(recvbuff, const_cast<void *>(sendbuff),
                                      count, data_type,
                                      ACL_MEMCPY_DEVICE_TO_DEVICE));
  }

  PADDLE_ENFORCE_NPU_SUCCESS(paddle::platform::dynload::HcclBroadcast(
      recvbuff, count, data_type, root, comm, stream));

  return SUCCESS;
}

HierarchicalHcclResult HcclBackend::reduce_scatter(
    const void *sendbuff, void *recvbuff, size_t recv_count,
    HierarchicalHcclDataType data_type, HierarchicalHcclReductionOp op,
    HierarchicalHcclCommGroupIdType group_id,
    HierarchicalHcclRuntimeStream stream) {
  PADDLE_ENFORCE_NPU_SUCCESS(paddle::platform::dynload::HcclReduceScatter(
      const_cast<void *>(sendbuff), reinterpret_cast<void *>(recvbuff),
      recv_count, data_type, op, comm, stream));

  return SUCCESS;
}
// Allgather
HierarchicalHcclResult HcclBackend::all_gather(
    const void *sendbuff, void *recvbuff, size_t send_count,
    HierarchicalHcclDataType data_type,
    HierarchicalHcclCommGroupIdType group_id,
    HierarchicalHcclRuntimeStream stream) {
  PADDLE_ENFORCE_NPU_SUCCESS(paddle::platform::dynload::HcclAllGather(
      const_cast<void *>(sendbuff), reinterpret_cast<void *>(recvbuff),
      send_count, data_type, comm, stream));

  return SUCCESS;
}

HierarchicalHcclResult HcclBackend::memcpy(void *dst, void *src, size_t count,
                                           HierarchicalHcclDataType data_type,
                                           int type) {
  size_t data_type_size = 0;
  PADDLE_ENFORCE_NPU_SUCCESS(to_HcclDataTypeSize(data_type, &data_type_size));

  PADDLE_ENFORCE_NPU_SUCCESS(aclrtMemcpy(dst, count * data_type_size, src,
                                         count * data_type_size,
                                         (aclrtMemcpyKind)type));
  return SUCCESS;
}

HierarchicalHcclResult HcclBackend::allocate(
    void **dst, size_t count, HierarchicalHcclDataType data_type) {
  size_t data_type_size = 0;
  PADDLE_ENFORCE_NPU_SUCCESS(to_HcclDataTypeSize(data_type, &data_type_size));
  PADDLE_ENFORCE_NPU_SUCCESS(
      aclrtMalloc(dst, count * data_type_size, ACL_MEM_MALLOC_HUGE_FIRST));
  return SUCCESS;
}

}  // namespace operators
}  // namespace paddle
