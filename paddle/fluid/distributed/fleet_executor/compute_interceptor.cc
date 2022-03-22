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
#include "paddle/fluid/framework/executor_gc_helper.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace distributed {

ComputeInterceptor::ComputeInterceptor(int64_t interceptor_id, TaskNode* node)
    : Interceptor(interceptor_id, node) {
  PrepareDeps();
  RegisterMsgHandle([this](const InterceptorMessage& msg) { Compute(msg); });
}

void ComputeInterceptor::RunOps() {
  VLOG(3) << "ComputeInterceptor " << interceptor_id_
          << " running ops in scope " << scope_idx_;
  for (auto op : node_->ops()) {
    PADDLE_ENFORCE_LT(scope_idx_, microbatch_scopes_.size(),
                      platform::errors::InvalidArgument(
                          "Scope index out of range. There are %ld "
                          "microbatch_scopes, but recevice scope index %ld",
                          microbatch_scopes_.size(), scope_idx_));
    op->Run(*microbatch_scopes_[scope_idx_], place_);
    if (gc_) {
      framework::DeleteUnusedTensors(*microbatch_scopes_[scope_idx_], op,
                                     node_->unused_vars(), gc_.get());
    }
  }
}

void ComputeInterceptor::PrepareDeps() {
  auto& upstream = node_->upstream();
  auto& downstream = node_->downstream();
  for (auto up : upstream) {
    in_readys_.emplace(up.first, std::make_pair(up.second, 0));
  }
  for (auto down : downstream) {
    out_buffs_.emplace(down.first, std::make_pair(down.second, 0));
  }
}

void ComputeInterceptor::Run() {
  while (IsInputReady() && CanWriteOutput()) {
    VLOG(3) << "id=" << GetInterceptorId() << " ComputeInterceptor running";
    scope_idx_ = ready_scope_idxs_.front();
    ready_scope_idxs_.pop();
    PADDLE_ENFORCE_GE(scope_idx_, 0,
                      platform::errors::InvalidArgument(
                          "The scope index should greater or equal to 0."));
    RunOps();
    // send to downstream and increase buff used
    SendDataReadyToDownStream();
    // reply to upstream and decrease ready data
    ReplyCompletedToUpStream();
  }
}

void ComputeInterceptor::Compute(const InterceptorMessage& msg) {
  if (msg.message_type() == DATA_IS_READY) {
    ready_scope_idxs_.push(msg.scope_idx());
    IncreaseReady(msg.src_id());
    Run();
  } else if (msg.message_type() == DATA_IS_USELESS) {
    DecreaseBuff(msg.src_id());
    Run();
  }
}

REGISTER_INTERCEPTOR(Compute, ComputeInterceptor);

}  // namespace distributed
}  // namespace paddle
