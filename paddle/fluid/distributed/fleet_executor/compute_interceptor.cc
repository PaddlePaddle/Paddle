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

#include "paddle/fluid/distributed/fleet_executor/carrier.h"
#include "paddle/fluid/distributed/fleet_executor/task_node.h"
#include "paddle/fluid/framework/executor_gc_helper.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/jit/serializer.h"
#include "paddle/phi/core/errors.h"

namespace paddle {
namespace distributed {

ComputeInterceptor::ComputeInterceptor(int64_t interceptor_id, TaskNode* node)
    : Interceptor(interceptor_id, node) {
  PrepareDeps();
  RegisterMsgHandle([this](const InterceptorMessage& msg) { Compute(msg); });
}

void ComputeInterceptor::PrepareDeps() {
  auto& upstream = node_->upstream();
  auto& downstream = node_->downstream();

  for (auto up : upstream) {
    std::map<int64_t, int64_t> ready_size_map;
    for (int64_t i = 0; i < node_->max_run_times(); ++i) {
      ready_size_map.emplace(i, 0);
    }
    in_readys_.emplace(up.first, std::make_pair(up.second, ready_size_map));
  }
  for (auto down : downstream) {
    out_buffs_.emplace(down.first, std::make_pair(down.second, 0));
  }
}

void ComputeInterceptor::DecodeMsgVars(const InterceptorMessage& msg) {
  int64_t scope_id = msg.scope_idx();
  PADDLE_ENFORCE_LT(scope_id,
                    microbatch_scopes_.size(),
                    platform::errors::InvalidArgument(
                        "Step out of range. There are %ld "
                        "microbatch_scopes, but recevice scope index %ld",
                        microbatch_scopes_.size(),
                        scope_id));
  auto* scope = microbatch_scopes_[scope_id];
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  for (const auto& var_iter : msg.vars_list()) {
    const std::string& name = var_iter.name();
    auto& dev_ctx = *pool.Get(place_);
    std::istringstream ss(var_iter.stensor());
    auto* var = scope->Var(name);
    auto* tensor = var->GetMutable<phi::DenseTensor>();
    framework::DeserializeFromStream(ss, tensor, dev_ctx);

    VLOG(3) << "Set vars " << name << " with value in scope " << scope_id
            << " with dims " << tensor->dims() << " with dtype "
            << tensor->dtype();
  }
}

InterceptorMessage ComputeInterceptor::PrepareVarsMsg() {
  PADDLE_ENFORCE_LT(cur_scope_id_,
                    microbatch_scopes_.size(),
                    platform::errors::InvalidArgument(
                        "Step out of range. There are %ld "
                        "microbatch_scopes, but recevice scope index %ld",
                        microbatch_scopes_.size(),
                        cur_scope_id_));
  auto* scope = microbatch_scopes_[cur_scope_id_];

  InterceptorMessage ready_msg;
  ready_msg.set_message_type(DATA_WITH_VARS);
  ready_msg.set_scope_idx(cur_scope_id_);
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  for (auto iter : node_->vars_to_dtype()) {
    VarList* vars = ready_msg.add_vars_list();
    const auto& var_name = iter.first;
    vars->set_name(var_name);
    std::ostringstream ss;
    auto& dev_ctx = *pool.Get(place_);
    auto* var = scope->FindVar(var_name);
    PADDLE_ENFORCE(
        var,
        platform::errors::NotFound(
            "Variable %s not exists in scope %ld", var_name, cur_scope_id_));
    const auto& tensor = var->Get<phi::DenseTensor>();
    framework::SerializeToStream(ss, tensor, dev_ctx);
    vars->set_stensor(ss.str());
    VLOG(3) << "Prepare vars msg " << var_name << " with dimension "
            << tensor.dims() << " dtype " << tensor.dtype();
  }
  return ready_msg;
}

void ComputeInterceptor::IncreaseReady(int64_t up_id, int64_t scope_id) {
  auto it = in_readys_.find(up_id);
  PADDLE_ENFORCE_NE(it,
                    in_readys_.end(),
                    platform::errors::NotFound(
                        "Cannot find upstream=%lld in in_readys.", up_id));

  auto max_ready_size = it->second.first;
  const auto& ready_scope_map = it->second.second;
  int64_t ready_size = 0;
  for (auto& scope_iter : ready_scope_map) {
    ready_size += scope_iter.second;
  }
  if (max_ready_size != INFINITE_BUFFER_SIZE) {
    PADDLE_ENFORCE_LE(
        ready_size,
        max_ready_size,
        platform::errors::OutOfRange(
            "upstream=%lld ready_size must <= max_ready_size, but "
            "now ready_size=%lld, max_ready_size=%lld",
            up_id,
            ready_size,
            max_ready_size));
  }
  PADDLE_ENFORCE_NE(
      it->second.second.find(scope_id),
      it->second.second.end(),
      platform::errors::OutOfRange(
          "Interceptor %lld can not find scope %lld in upstream ready map",
          interceptor_id_,
          scope_id));
  it->second.second.at(scope_id) = ready_scope_map.at(scope_id) + 1;
}

void ComputeInterceptor::DecreaseBuff(int64_t down_id) {
  auto it = out_buffs_.find(down_id);
  PADDLE_ENFORCE_NE(it,
                    out_buffs_.end(),
                    platform::errors::NotFound(
                        "Cannot find downstream=%lld in out_buffs.", down_id));
  auto used_size = it->second.second;
  used_size -= 1;
  PADDLE_ENFORCE_GE(
      used_size,
      0,
      platform::errors::OutOfRange(
          "downstream=%lld used buff size must >= 0, but now equal %lld",
          down_id,
          used_size));
  it->second.second = used_size;
}

bool ComputeInterceptor::IsInputReady() {
  std::map<int64_t, bool> scope_id_to_finish_flag;
  if (!gen_step_to_scope_id_to_finish_flag_.empty()) {
    scope_id_to_finish_flag =
        gen_step_to_scope_id_to_finish_flag_.begin()->second;
    VLOG(3) << "Is Input Ready in gen step "
            << gen_step_to_scope_id_to_finish_flag_.begin()->first;
  }
  int64_t num_micro_step =
      (num_micro_step_ == -1 ? node_->max_run_times() : num_micro_step_);
  int64_t start_micro_step = (start_micro_step_ == -1 ? 0 : start_micro_step_);
  for (int64_t i = start_micro_step; i < start_micro_step + num_micro_step;
       ++i) {
    bool flag = true;
    for (auto& ins : in_readys_) {
      auto ready_size_map = ins.second.second;
      flag = flag && (ready_size_map.at(i) != 0);
    }
    if (flag) {
      if (scope_id_to_finish_flag.empty()) {
        cur_scope_id_ = i;
        return true;
      } else if (scope_id_to_finish_flag.find(i) !=
                 scope_id_to_finish_flag.end()) {
        for (auto iter : scope_id_to_finish_flag) {
          if (iter.first == i) {
            break;
          } else if (!iter.second) {
            VLOG(3) << "The previous scope is not ready, waiting for the "
                       "previous scope "
                    << iter.first << " in gen_step "
                    << gen_step_to_scope_id_to_finish_flag_.begin()->first;
            return false;
          }
        }
        cur_scope_id_ = i;
        return true;
      } else {
        VLOG(3) << "Interceptor " << GetInterceptorId() << " in scope " << i
                << " is larger than gen_step "
                << gen_step_to_scope_id_to_finish_flag_.begin()->first;
      }
    } else {
      VLOG(3) << "Interceptor " << GetInterceptorId() << " in scope " << i
              << "'s upstreams aren't all ready.";
    }
  }
  return false;
}

bool ComputeInterceptor::CanWriteOutput() {
  for (auto& outs : out_buffs_) {
    auto max_buffer_size = outs.second.first;
    auto used_size = outs.second.second;
    if (max_buffer_size == INFINITE_BUFFER_SIZE) {
      continue;
    }
    // full, return false
    if (used_size == max_buffer_size) {
      VLOG(3) << "Interceptor " << GetInterceptorId()
              << "'s out buffer is full.";
      return false;
    }
  }
  return true;
}

void ComputeInterceptor::SendDataReadyToDownStream() {
  bool need_send_vars = !(node_->vars_to_dtype().empty());
  InterceptorMessage ready_msg;
  ready_msg.set_start_micro_step(start_micro_step_);
  ready_msg.set_num_micro_step(num_micro_step_);
  if (need_send_vars) {
    ready_msg = PrepareVarsMsg();
  } else {
    ready_msg.set_message_type(DATA_IS_READY);
    ready_msg.set_scope_idx(cur_scope_id_);
  }
  for (auto& outs : out_buffs_) {
    auto down_id = outs.first;
    auto max_buff_size = outs.second.first;
    auto used_size = outs.second.second;
    used_size += 1;
    if (max_buff_size != INFINITE_BUFFER_SIZE) {
      PADDLE_ENFORCE_LE(
          used_size,
          max_buff_size,
          platform::errors::OutOfRange("downstream=%lld used buff size must <= "
                                       "max_buff_size, but now used_size=%lld, "
                                       "max_buff_size=%lld",
                                       down_id,
                                       used_size,
                                       max_buff_size));
    }
    outs.second.second = used_size;

    if (need_send_vars) {
      VLOG(3) << "ComputeInterceptor " << interceptor_id_
              << " Send data_with_vars msg to " << down_id
              << " in scope: " << cur_scope_id_;
      Send(down_id, ready_msg);
    } else {
      VLOG(3) << "ComputeInterceptor " << interceptor_id_
              << " Send data_is_ready msg to " << down_id
              << " in scope: " << cur_scope_id_;
      Send(down_id, ready_msg);
    }
  }
}

void ComputeInterceptor::ReplyCompletedToUpStream() {
  for (auto& ins : in_readys_) {
    auto up_id = ins.first;
    auto ready_size = ins.second.second.at(cur_scope_id_);
    ready_size -= 1;
    PADDLE_ENFORCE_GE(
        ready_size,
        0,
        platform::errors::OutOfRange(
            "upstream=%lld ready_size must >= 0, but now got %lld",
            up_id,
            ready_size));
    ins.second.second[cur_scope_id_] = ready_size;

    VLOG(3) << "ComputeInterceptor " << interceptor_id_
            << " Reply data_is_useless msg to " << up_id
            << " in scope: " << cur_scope_id_;

    InterceptorMessage reply_msg;
    reply_msg.set_message_type(DATA_IS_USELESS);
    reply_msg.set_scope_idx(cur_scope_id_);
    Send(up_id, reply_msg);
  }
}

void ComputeInterceptor::RunOps() {
  if (!cores_.empty() || !node_->ops().empty()) {
    PADDLE_ENFORCE_LT(cur_scope_id_,
                      microbatch_scopes_.size(),
                      platform::errors::InvalidArgument(
                          "Step out of range. There are %ld "
                          "microbatch_scopes, but receive scope index %ld",
                          microbatch_scopes_.size(),
                          cur_scope_id_));
  }

  if (!cores_.empty()) {
    cores_[cur_scope_id_]->Run(/*feed_names=*/{}, /*need_fetch=*/false);
  } else {
    for (auto op : node_->ops()) {
      op->Run(*microbatch_scopes_[cur_scope_id_], place_);
      if (gc_) {
        framework::DeleteUnusedTensors(*microbatch_scopes_[cur_scope_id_],
                                       op,
                                       node_->unused_vars(),
                                       gc_.get());
      }
    }
  }
}

void ComputeInterceptor::Run() {
  while (IsInputReady() && CanWriteOutput()) {
    VLOG(3) << "id=" << GetInterceptorId()
            << " ComputeInterceptor running in scope " << cur_scope_id_;

    RunOps();

    if (!gen_step_to_scope_id_to_finish_flag_.empty()) {
      auto iter = gen_step_to_scope_id_to_finish_flag_.begin();
      VLOG(3) << "id=" << GetInterceptorId()
              << " ComputeInterceptor running in scope " << cur_scope_id_
              << " with gen_step " << iter->first;
      auto& scope_id_to_finish_flag = iter->second;
      PADDLE_ENFORCE_NE(
          scope_id_to_finish_flag.find(cur_scope_id_),
          scope_id_to_finish_flag.end(),
          platform::errors::NotFound(
              "Can not find scope %ld in scope_id_to_finish", cur_scope_id_));
      scope_id_to_finish_flag.erase(cur_scope_id_);
      if (scope_id_to_finish_flag.empty()) {
        gen_step_to_scope_id_to_finish_flag_.erase(iter);
      }
    }

    // send to downstream and increase buff used
    SendDataReadyToDownStream();
    // reply to upstream and decrease ready data
    ReplyCompletedToUpStream();
    // clear TensorArray
    auto vars_names = microbatch_scopes_[cur_scope_id_]->LocalVarNames();
    for (auto var_name : vars_names) {
      if (var_name == "feed" || var_name == "fetch") continue;
      auto* var = microbatch_scopes_[cur_scope_id_]->Var(var_name);
      if (var != nullptr && var->IsType<framework::LoDTensorArray>()) {
        auto* lod_tensor_arr = var->GetMutable<framework::LoDTensorArray>();
        lod_tensor_arr->clear();
      }
    }
  }
}

void ComputeInterceptor::Compute(const InterceptorMessage& msg) {
  if (msg.message_type() == DATA_IS_READY) {
    VLOG(3) << "Compute interceptor " << interceptor_id_
            << " receive data_is_ready " << msg.src_id() << " "
            << msg.scope_idx() << " ";
    start_micro_step_ = msg.start_micro_step();
    num_micro_step_ = msg.num_micro_step();
    IncreaseReady(msg.src_id(), msg.scope_idx());
    Run();
  } else if (msg.message_type() == DATA_IS_USELESS) {
    VLOG(3) << "Compute interceptor " << interceptor_id_
            << " receive data_is_useless " << msg.src_id() << " "
            << msg.scope_idx() << " ";
    DecreaseBuff(msg.src_id());
    Run();
  } else if (msg.message_type() == DATA_WITH_VARS) {
    VLOG(3) << "Compute interceptor " << interceptor_id_
            << " receive data_with_vars " << msg.src_id() << " "
            << msg.scope_idx() << " ";
    DecodeMsgVars(msg);
    IncreaseReady(msg.src_id(), msg.scope_idx());
    Run();
  } else if (msg.message_type() == START_LOOP) {
    VLOG(3) << "Compute interceptor " << interceptor_id_
            << " receive start_loop " << msg.src_id() << " in scope "
            << msg.scope_idx() << " with gen_step " << msg.gen_step();
    start_micro_step_ = msg.start_micro_step();
    num_micro_step_ = msg.num_micro_step();
    IncreaseReady(msg.src_id(), msg.scope_idx());
    int64_t gen_step = msg.gen_step();
    gen_step_to_scope_id_to_finish_flag_[gen_step].emplace(msg.scope_idx(),
                                                           false);
    Run();
  }
}

REGISTER_INTERCEPTOR(Compute, ComputeInterceptor);

}  // namespace distributed
}  // namespace paddle
