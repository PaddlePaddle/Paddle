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
#include "paddle/common/errors.h"
#include "paddle/fluid/distributed/fleet_executor/task_node.h"
#include "paddle/fluid/framework/executor_gc_helper.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"

namespace paddle::distributed {

CondInterceptor::CondInterceptor(int64_t interceptor_id, TaskNode* node)
    : Interceptor(interceptor_id, node),
      cur_scope_id_(0),
      normal_in_id_(),
      normal_out_id_(),
      stop_loop_id_(0),
      loop_id_(0),
      scope_id_to_gen_step_(),
      start_micro_step_(0),
      num_micro_step_(0) {
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
                    phi::errors::InvalidArgument(
                        "Step out of range. There are %ld "
                        "microbatch_scopes, but receive scope index %ld",
                        microbatch_scopes_.size(),
                        cur_scope_id_));
  auto* cond_var =
      microbatch_scopes_[cur_scope_id_]->FindVar(node_->cond_var());
  PADDLE_ENFORCE(
      cond_var,
      phi::errors::NotFound("Condition variable %s not exists in scope %ld",
                            node_->cond_var(),
                            cur_scope_id_));
  const auto& cond_tensor = cond_var->Get<phi::DenseTensor>();
  bool res = false;
  if (phi::is_gpu_place(cond_tensor.place())) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    phi::DenseTensor cpu_tensor;
    framework::TensorCopy(cond_tensor, phi::CPUPlace(), &cpu_tensor);
    phi::DeviceContextPool::Instance().Get(cond_tensor.place())->Wait();
    res = cpu_tensor.data<bool>()[0];
#endif
  } else if (phi::is_custom_place(cond_tensor.place())) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
    phi::DenseTensor cpu_tensor;
    framework::TensorCopy(cond_tensor, phi::CPUPlace(), &cpu_tensor);
    phi::DeviceContextPool::Instance().Get(cond_tensor.place())->Wait();
    res = cpu_tensor.data<bool>()[0];
#endif
  } else if (phi::is_cpu_place(cond_tensor.place())) {
    res = cond_tensor.data<bool>()[0];
  } else {
    PADDLE_THROW(
        phi::errors::Unimplemented("Unsupport device for cond interceptor."));
  }
  return res;
}

void CondInterceptor::SendDataReady(int64_t down_id) {
  InterceptorMessage ready_msg;
  ready_msg.set_message_type(DATA_IS_READY);
  ready_msg.set_scope_idx(cur_scope_id_);
  ready_msg.set_start_micro_step(start_micro_step_);
  ready_msg.set_num_micro_step(num_micro_step_);
  Send(down_id, ready_msg);
}

void CondInterceptor::SendStartLoop(int64_t down_id, int64_t gen_step) {
  InterceptorMessage ready_msg;
  ready_msg.set_message_type(START_LOOP);
  ready_msg.set_scope_idx(cur_scope_id_);
  ready_msg.set_gen_step(gen_step);
  ready_msg.set_start_micro_step(start_micro_step_);
  ready_msg.set_num_micro_step(num_micro_step_);
  Send(down_id, ready_msg);
}

void CondInterceptor::ReplyDataIsUseless(int64_t up_id) {
  InterceptorMessage ready_msg;
  ready_msg.set_message_type(DATA_IS_USELESS);
  ready_msg.set_scope_idx(cur_scope_id_);
  Send(up_id, ready_msg);
}

void CondInterceptor::Compute(int64_t gen_step) {
  bool cond = GetCondResult();
  VLOG(3) << "Cond interceptor get condition var " << node_->cond_var()
          << " with value " << cond;
  if (cond) {
    VLOG(3) << "Loop again in scope " << cur_scope_id_ << " gen_step "
            << gen_step;
    for (auto& down_id : normal_out_id_) {
      SendStartLoop(down_id, gen_step);
    }
  } else {
    PADDLE_ENFORCE_NE(scope_id_to_gen_step_.find(cur_scope_id_),
                      scope_id_to_gen_step_.end(),
                      phi::errors::InvalidArgument(
                          "Can not find scope id %ld in scope_id_to_gen_step",
                          cur_scope_id_));
    VLOG(3) << "Finish loop in scope " << cur_scope_id_ << " with "
            << scope_id_to_gen_step_.at(cur_scope_id_) << " generation steps.";
    scope_id_to_gen_step_.erase(cur_scope_id_);
    SendDataReady(stop_loop_id_);
  }
}

void CondInterceptor::Run(const InterceptorMessage& msg) {
  if (msg.message_type() == DATA_IS_READY) {
    cur_scope_id_ = msg.scope_idx();
    start_micro_step_ = msg.start_micro_step();
    num_micro_step_ = msg.num_micro_step();
    scope_id_to_gen_step_.emplace(cur_scope_id_, 0);
    Compute(/*gen_step=*/0);
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
  } else if (msg.message_type() == DATA_WITH_VARS) {
    int64_t scope_id = msg.scope_idx();
    PADDLE_ENFORCE_NE(
        scope_id_to_gen_step_.find(scope_id),
        scope_id_to_gen_step_.end(),
        phi::errors::InvalidArgument(
            "Can not find scope id %ld in scope_id_to_gen_step", scope_id));
    // Keep the message in order with scope_id
    // message with scope 3 never send before scope 1.
    int64_t gen_step = scope_id_to_gen_step_.at(scope_id) + 1;
    bool wait_prev_scope = false;
    // If the previous scope gen_step less than cur scope
    // means: the previous scope doesn't finish last step generation, should
    // wait.
    auto iter = scope_id_to_gen_step_.begin();
    while (iter != scope_id_to_gen_step_.end()) {
      if (iter->first == scope_id) {
        break;
      }
      if (iter->second < gen_step) {
        wait_prev_scope = true;
        break;
      }
      ++iter;
    }
    scope_id_to_gen_step_.at(scope_id) = gen_step;
    if (!wait_prev_scope) {
      // Start send message to all scopes gen_step equal to cur_scope
      std::vector<int64_t> ready_scope_ids;
      while (iter != scope_id_to_gen_step_.end()) {
        if (iter->second == gen_step) {
          ready_scope_ids.emplace_back(iter->first);
        } else if (iter->second > gen_step) {
          PADDLE_THROW(
              phi::errors::Fatal("Some error may occur. Scope %ld's "
                                 "gen_step is much larger than previous.",
                                 iter->first));
        } else {
          break;
        }
        ++iter;
      }
      for (auto& scope_id : ready_scope_ids) {
        cur_scope_id_ = scope_id;
        Compute(gen_step);
      }
    }
  }
}

REGISTER_INTERCEPTOR(Cond, CondInterceptor);

}  // namespace paddle::distributed
