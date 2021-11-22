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

#include "paddle/fluid/distributed/fleet_executor/compute_interceptor.h"

#include "paddle/fluid/distributed/fleet_executor/task_node.h"

namespace paddle {
namespace distributed {

ComputeInterceptor::ComputeInterceptor(int64_t interceptor_id, TaskNode* node)
    : Interceptor(interceptor_id, node) {
  PrepareDeps();
  RegisterMsgHandle([this](const InterceptorMessage& msg) { Compute(msg); });
}

void ComputeInterceptor::PrepareDeps() {
  auto& upstream = GetTaskNode()->upstream();
  upstream_deps_.insert(upstream.begin(), upstream.end());
}

void ComputeInterceptor::SendDataReadyToDownStream() {
  auto& downstream = GetTaskNode()->downstream();
  for (auto dst_id : downstream) {
    InterceptorMessage dst_msg;
    dst_msg.set_message_type(DATA_IS_READY);
    VLOG(3) << "ComputeInterceptor Send msg to " << dst_id;
    Send(dst_id, dst_msg);
  }
}

void ComputeInterceptor::Compute(const InterceptorMessage& msg) {
  if (msg.message_type() == DATA_IS_READY) {
    auto src_id = msg.src_id();
    upstream_deps_.erase(src_id);

    // all input is ready
    if (upstream_deps_.empty()) {
      // TODO(wangxi): op run
      VLOG(3) << "id=" << GetInterceptorId() << " ComputeInterceptor running";
      SendDataReadyToDownStream();
      PrepareDeps();
    }
  }
}

REGISTER_INTERCEPTOR(Compute, ComputeInterceptor);

}  // namespace distributed
}  // namespace paddle
