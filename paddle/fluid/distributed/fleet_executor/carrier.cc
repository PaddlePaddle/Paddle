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

#include "paddle/fluid/distributed/fleet_executor/carrier.h"
#include "paddle/fluid/distributed/fleet_executor/interceptor.h"
#include "paddle/fluid/distributed/fleet_executor/interceptor_message_service.h"
#include "paddle/fluid/distributed/fleet_executor/task_node.h"

namespace paddle {
namespace distributed {

void Carrier::Init(
    const std::unordered_map<int64_t, TaskNode*>& interceptor_id_to_node) {
  interceptor_id_to_node_ = interceptor_id_to_node;
  CreateInterceptors();
}

bool Carrier::EnqueueInterceptorMessage(
    const InterceptorMessage& interceptor_message) {
  // enqueue message to interceptor
  if (interceptor_message.ctrl_message()) {
    // handle control message
    return true;
  } else {
    int64_t dst_id = interceptor_message.dst_id();
    Interceptor* dst_interceptor = GetInterceptor(dst_id);
    bool rst =
        dst_interceptor->EnqueueRemoteInterceptorMessage(interceptor_message);
    if (rst) {
      std::condition_variable& interceptor_cond_var =
          dst_interceptor->GetCondVar();
      interceptor_cond_var.notify_all();
    }
    return rst;
  }
}

Interceptor* Carrier::GetInterceptor(int64_t interceptor_id) {
  auto iter = interceptor_idx_to_interceptor_.find(interceptor_id);
  PADDLE_ENFORCE_NE(iter, interceptor_idx_to_interceptor_.end(),
                    platform::errors::InvalidArgument(
                        "Cannot find interceptor instance for interceptor "
                        "id %lld. Wrong dst? Call before init?",
                        interceptor_id));
  return iter->second.get();
}

Interceptor* Carrier::SetInterceptor(int64_t interceptor_id,
                                     std::unique_ptr<Interceptor> interceptor) {
  auto iter = interceptor_idx_to_interceptor_.find(interceptor_id);
  PADDLE_ENFORCE_EQ(iter, interceptor_idx_to_interceptor_.end(),
                    platform::errors::AlreadyExists(
                        "The interceptor id %lld has already been created! "
                        "The interceptor is should be unique.",
                        interceptor_id));
  auto* ptr = interceptor.get();
  interceptor_idx_to_interceptor_.insert(
      std::make_pair(interceptor_id, std::move(interceptor)));
  return ptr;
}

void Carrier::CreateInterceptors() {
  // create each Interceptor
  for (const auto& item : interceptor_id_to_node_) {
    int64_t interceptor_id = item.first;
    TaskNode* task_node = item.second;

    // TODO(wangxi): use node_type to select different Interceptor
    auto interceptor = std::make_unique<Interceptor>(interceptor_id, task_node);
    SetInterceptor(interceptor_id, std::move(interceptor));
    VLOG(3) << "Create Interceptor for " << interceptor_id;
  }
}

}  // namespace distributed
}  // namespace paddle
