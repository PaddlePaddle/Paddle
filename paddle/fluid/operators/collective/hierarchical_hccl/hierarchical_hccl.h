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

#include <string>
#include <vector>

#include "paddle/fluid/operators/collective/hierarchical_hccl/hierarchical_hccl_types.h"

namespace paddle {
namespace operators {

// Generate HierarchicalHccl unique id
HierarchicalHcclResult hierarchical_hccl_gen_unique_id(
    const int my_rank, const char *bootstrap_endpoint, const int rank_count,
    const int split_index, HierarchicalHcclCommGroupIdType comm_group_id);

// Initialize HierarchicalHccl for rank
HierarchicalHcclResult hierarchical_hccl_init_comm_global(
    const int rank_count, const int my_rank,
    int my_device_id,
    HierarchicalHcclCommGroupIdType comm_group_id);

// Destroy the HierarchicalHccl world communicator
HierarchicalHcclResult hierarchical_hccl_destroy_comm_global(
    HierarchicalHcclCommGroupIdType comm_group_id);

// Reduce
HierarchicalHcclResult hierarchical_hccl_reduce(
    const void *sendbuff, void *recvbuff, size_t count,
    HierarchicalHcclDataType data_type, HierarchicalHcclReductionOp op,
    int root, HierarchicalHcclCommGroupIdType group_id,
    HierarchicalHcclRuntimeStream stream);

// Broadcast
HierarchicalHcclResult hierarchical_hccl_broadcast(
    const void *sendbuff, void *recvbuff, size_t count,
    HierarchicalHcclDataType data_type, int root,
    HierarchicalHcclCommGroupIdType group_id,
    HierarchicalHcclRuntimeStream stream);

// All-Reduce
HierarchicalHcclResult hierarchical_hccl_all_reduce(
    const void *sendbuff, void *recvbuff, size_t count,
    HierarchicalHcclDataType data_type, HierarchicalHcclReductionOp op,
    HierarchicalHcclCommGroupIdType group_id,
    HierarchicalHcclRuntimeStream stream);

// Reduce-Scatter
HierarchicalHcclResult hierarchical_hccl_reduce_scatter(
    const void *sendbuff, void *recvbuff, size_t recv_count,
    HierarchicalHcclDataType data_type, HierarchicalHcclReductionOp op,
    HierarchicalHcclCommGroupIdType group_id,
    HierarchicalHcclRuntimeStream stream);

// All-Gather
HierarchicalHcclResult hierarchical_hccl_all_gather(
    const void *sendbuff, void *recvbuff, size_t send_count,
    HierarchicalHcclDataType data_type,
    HierarchicalHcclCommGroupIdType group_id,
    HierarchicalHcclRuntimeStream stream);

// Send
HierarchicalHcclResult hierarchical_hccl_send(
    const void *sendbuff, size_t count, HierarchicalHcclDataType data_type,
    int peer, HierarchicalHcclCommGroupIdType group_id,
    HierarchicalHcclRuntimeStream stream);

// Receive
HierarchicalHcclResult hierarchical_hccl_recv(
    void *recvbuff, size_t count, HierarchicalHcclDataType data_type, int peer,
    HierarchicalHcclCommGroupIdType group_id,
    HierarchicalHcclRuntimeStream stream);

}  // namespace operators
}  // namespace paddle
