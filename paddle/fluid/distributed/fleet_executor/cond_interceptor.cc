// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/distributed/fleet_executor/cond_interceptor.h"
#include <algorithm>
#include "paddle/fluid/distributed/fleet_executor/task_node.h"
#include "paddle/fluid/framework/executor_gc_helper.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/errors.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/errors.h"

namespace paddle {
namespace distributed {

CondInterceptor::CondInterceptor(int64_t interceptor_id, TaskNode* node)
    : Interceptor(interceptor_id, node) {
  PrepareDeps();
  RegisterMsgHandle([this](const InterceptorMessage& msg) { Run(msg); });
}

void CondInterceptor::PrepareDeps() {
  auto& upstream = node_->upstream();
  auto& downstream = node_->downstream();
  auto& id_to_dep_type = node_->id_to_dep_type();

  for (const auto& up : upstream) {
    if (id_to_dep_type.at(up.first) == DependType::NORMAL) {
      normal_in_id_.insert(up.first);
    } else if (id_to_dep_type.at(up.first) == DependType::LOOP) {
      loop_id_ = up.first;
    }
  }

  for (const auto& down : downstream) {
    if (id_to_dep_type.at(down.first) == DependType::NORMAL) {
      normal_out_id_.insert(down.first);
    } else if (id_to_dep_type.at(down.first) == DependType::STOP_LOOP) {
      stop_loop_id_ = down.first;
    }
  }
}

bool CondInterceptor::GetCondResult() {
  PADDLE_ENFORCE_LT(cur_scope_id_,
                    microbatch_scopes_.size(),
                    platform::errors::InvalidArgument(
                        "Step out of range. There are %ld "
                        "microbatch_scopes, but recevice scope index %ld",
                        microbatch_scopes_.size(),
                        cur_scope_id_));
  auto* cond_var =
      microbatch_scopes_[cur_scope_id_]->FindVar(node_->cond_var());
  PADDLE_ENFORCE(cond_var,
                 platform::errors::NotFound(
                     "Condition variable %s not exists in scope %ld",
                     node_->cond_var(),
                     cur_scope_id_));
  const auto& cond_tensor = cond_var->Get<phi::DenseTensor>();
  bool res = false;
  if (platform::is_gpu_place(cond_tensor.place())) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    phi::DenseTensor cpu_tensor;
    framework::TensorCopy(cond_tensor, platform::CPUPlace(), &cpu_tensor);
    platform::DeviceContextPool::Instance().Get(cond_tensor.place())->Wait();
    res = cpu_tensor.data<bool>()[0];
#endif
  } else if (platform::is_cpu_place(cond_tensor.place())) {
    res = cond_tensor.data<bool>()[0];
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Unsupport device for cond interceptor."));
  }
  return res;
}

void CondInterceptor::SendDataReady(int64_t down_id) {
  InterceptorMessage ready_msg;
  ready_msg.set_message_type(DATA_IS_READY);
  ready_msg.set_scope_idx(cur_scope_id_);
  Send(down_id, ready_msg);
}

void CondInterceptor::SendStartLoop(int64_t down_id) {
  InterceptorMessage ready_msg;
  ready_msg.set_message_type(START_LOOP);
  ready_msg.set_scope_idx(cur_scope_id_);
  Send(down_id, ready_msg);
}

void CondInterceptor::ReplyDataIsUseless(int64_t up_id) {
  InterceptorMessage ready_msg;
  ready_msg.set_message_type(DATA_IS_USELESS);
  ready_msg.set_scope_idx(cur_scope_id_);
  Send(up_id, ready_msg);
}

void CondInterceptor::Compute() {
  bool cond = GetCondResult();
  VLOG(3) << "Cond interceptor get condition var " << node_->cond_var()
          << " with value " << cond;
  if (cond) {
    VLOG(3) << "Loop again in scope " << cur_scope_id_;
    for (auto& down_id : normal_out_id_) {
      SendStartLoop(down_id);
    }
    ++num_of_scopes_;
  } else {
    VLOG(3) << "Finish loop in scope " << cur_scope_id_;
    SendDataReady(stop_loop_id_);
  }
}

void CondInterceptor::Run(const InterceptorMessage& msg) {
  if (msg.message_type() == DATA_IS_READY ||
      msg.message_type() == DATA_WITH_VARS) {
    if (msg.src_id() == loop_id_) {
      --num_of_scopes_;
      VLOG(3) << "Receving loop again message from " << msg.src_id()
              << " waiting other " << num_of_scopes_ << " scopes ready";
      ready_scope_id_.emplace_back(msg.scope_idx());
      if (num_of_scopes_ == 0) {
        std::sort(ready_scope_id_.begin(), ready_scope_id_.end());
        for (auto scope_id : ready_scope_id_) {
          VLOG(3) << "Start a new loop in scope " << scope_id;
          cur_scope_id_ = scope_id;
          Compute();
        }
        ready_scope_id_.clear();
      }
    } else {
      cur_scope_id_ = msg.scope_idx();
      Compute();
    }
  } else if (msg.message_type() == DATA_IS_USELESS) {
    if (node_->id_to_dep_type().at(msg.src_id()) == DependType::STOP_LOOP) {
      for (auto& up_id : normal_in_id_) {
        ReplyDataIsUseless(up_id);
      }
      // Gc the variable in while block
      int64_t scope_id = msg.scope_idx();
      if (gc_) {
        VLOG(3) << "Release vars in while block in scope " << scope_id;
        framework::DeleteUnusedTensors(*microbatch_scopes_[scope_id],
                                       node_->while_block_vars(),
                                       gc_.get());
      }
    }
  }
}

REGISTER_INTERCEPTOR(Cond, CondInterceptor);

}  // namespace distributed
}  // namespace paddle
