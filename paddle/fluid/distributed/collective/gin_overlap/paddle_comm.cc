// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

// paddle_comm.cc —— 从 Paddle 已经建好的 ProcessGroup（按 ring id）直接取
// ncclComm_t，不再自己 ncclCommInitRank。 做法和 DeepEP
// 一样（paddle/fluid/distributed/collective/deep_ep/deep_ep.cpp 里取 comm
// 的那几行）：
//   ProcessGroupMapFromGid::getInstance()->get(ring_id)
//   （phi/core/distributed/collective/process_group.h:555-580，经
//                                                          fluid 的
//                                                          process_group.h:51
//                                                          re-export；
//                                                          在树外是靠这个
//                                                          静态局部变量的
//                                                          weak / unique
//                                                          符号和 libpaddle.so
//                                                          里共用同一份，搬进
//                                                          树里之后本来就是
//                                                          同一个库）
//   → ProcessGroupNCCL::GetOrCreateCommContext(place, CommType::ALLTOALL)
//   （process_group_nccl.h:193） → NCCLCommContext::GetNcclComm()
//   （nccl_comm_context.h:55）
// 在 PaddleFleet 的扩展里这是唯一包含 Paddle 头文件的 TU（里面有 thrust，外部
// 构建必须交给 nvcc），所以那边后缀是 .cu；搬进 Paddle 之后这些就是自己的头
// 文件，普通 .cc 即可。 拿不到就带上原因返回非
// 0（上层报错，不静默回退到自己建 comm）。
#include <exception>
#include <string>

#include <cuda_runtime.h>
#include <nccl.h>

#include "paddle/fluid/distributed/collective/process_group_nccl.h"
#include "paddle/phi/core/distributed/nccl_comm_context.h"

namespace gin_overlap {

int paddle_nccl_comm(int ring_id,
                     int device_id,
                     ncclComm_t* out_comm,
                     int* out_size,
                     int* out_rank,
                     std::string* out_err) {
  *out_comm = nullptr;
  *out_size = 0;
  *out_rank = 0;
  auto process_groups =
      paddle::distributed::ProcessGroupMapFromGid::getInstance();
  if (process_groups == nullptr) {
    *out_err =
        "paddle::distributed::ProcessGroupMapFromGid::getInstance() 返回空";
    return -1;
  }
  paddle::distributed::ProcessGroup* process_group =
      process_groups->get(ring_id);
  if (process_group == nullptr) {
    *out_err =
        "Paddle 的 ProcessGroupMapFromGid 里没有 ring id " +
        std::to_string(ring_id) +
        "（这个通信组还没在 C++ 侧注册？确认传的是 paddle 通信组对象的 .id）";
    return -1;
  }
  auto* nccl_process_group =
      dynamic_cast<paddle::distributed::ProcessGroupNCCL*>(process_group);
  if (nccl_process_group == nullptr) {
    *out_err = "ring id " + std::to_string(ring_id) +
               " 对应的 ProcessGroup 不是 ProcessGroupNCCL（backend=" +
               process_group->GetBackendName() + "）";
    return -1;
  }
  *out_size = process_group->GetSize();
  *out_rank = process_group->GetRank();
  phi::distributed::NCCLCommContext* comm_context = nullptr;
  try {
    comm_context = nccl_process_group->GetOrCreateCommContext(
        phi::GPUPlace(device_id), phi::distributed::CommType::ALLTOALL);
  } catch (const std::exception& ex) {
    *out_err = "ring id " + std::to_string(ring_id) +
               " 的 GetOrCreateCommContext(GPUPlace(" +
               std::to_string(device_id) + ")) 抛异常：" + ex.what();
    return -1;
  }
  if (comm_context == nullptr) {
    *out_err = "ring id " + std::to_string(ring_id) +
               " 的 GetOrCreateCommContext 返回空";
    return -1;
  }
  ncclComm_t comm = comm_context->GetNcclComm();
  if (comm == nullptr) {
    *out_err = "ring id " + std::to_string(ring_id) +
               " 的 NCCLCommContext::GetNcclComm() 返回空";
    return -1;
  }
  *out_comm = comm;
  return 0;
}

}  // namespace gin_overlap
