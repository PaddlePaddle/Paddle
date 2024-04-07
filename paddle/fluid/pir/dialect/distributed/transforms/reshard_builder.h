// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include "paddle/common/reshard_function_desc.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/reshard_utils.h"
#include "paddle/phi/core/distributed/comm_context_manager.h"
#include "paddle/phi/core/distributed/store/store_utils.h"

namespace phi {
class DeviceContext;

namespace distributed {

static CommContext* GetCommContext(const std::vector<int64_t>& process_ids) {
  std::string unique_comm_key = GenUniqueCommKey(process_ids);

  if (!CommContextManager::GetInstance().Has(unique_comm_key)) {
    int64_t world_size = static_cast<int64_t>(process_ids.size());
    int64_t rank = GetLocalRankInParticipate(process_ids);
    VLOG(3) << "local world size: " << world_size << " local rank: " << rank;

    auto store = CreateOrGetGlobalTCPStore();

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    CommContextManager::SetDeviceId(rank);
    VLOG(0) << "init for unique_comm_key: " << unique_comm_key;
    CommContextManager::CreateNCCLCommContext(store,
                                              unique_comm_key,
                                              static_cast<int>(rank),
                                              static_cast<int>(world_size));
#elif defined(PADDLE_WITH_CUSTOM_DEVICE)
    VLOG(0) << "init for unique_comm_key: " << unique_comm_key;
    CommContextManager::CreateXCCLCommContext(
        store, unique_comm_key, dev_ctx.GetPlace(), rank, world_size);
#elif defined(PADDLE_WITH_GLOO)
    VLOG(0) << "init for unique_comm_key: " << unique_comm_key;
    CommContextManager::CreateGlooCommContext(store,
                                              unique_comm_key,
                                              static_cast<int>(rank),
                                              static_cast<int>(world_size));
#endif
  }

  auto* comm_context = CommContextManager::GetInstance().Get(unique_comm_key);
  return comm_context;
}

inline pir::Operation* Build(const ReshardFuncDesc* base_desc,
                             const pir::Builder& builder,
                             const std::vector<pir::Value>& inputs) {
  if (base_desc->name == "AllReduce") {
    AllReduceOpDesc* desc = dynamic_cast<AllReduceOpDesc*>(base_desc);
    GetCommContext(desc->process_ids);

    std::string ring_id = GenUniqueCommKey(desc->process_ids);
    return builder.Build<paddle::dialect::AllReduceOp>(
        inputs[0], ring_id, desc->reduce_type);
  } else if (base_desc->name == "Send") {
    SendOpDesc* desc = dynamic_cast<SendOpDesc*>(base_desc);
    GetCommContext(desc->process_ids);

    std::string ring_id = GenUniqueCommKey(desc->process_ids);
    return builder.Build<paddle::dialect::PSendOp>(
        inputs[0], ring_id, desc->peer, desc->dynamic_shape);
  } else if (base_desc->name == "Recv") {
    RecvOpDesc* desc = dynamic_cast<RecvOpDesc*>(base_desc);
    GetCommContext(desc->process_ids);

    std::string ring_id = GenUniqueCommKey(desc->process_ids);
    return builder.Build<paddle::dialect::PRecvOp>(
        ring_id, desc->peer, desc->dtype, desc->dynamic_shape);
  }
  return nullptr;
}

}  // namespace distributed
}  // namespace phi
