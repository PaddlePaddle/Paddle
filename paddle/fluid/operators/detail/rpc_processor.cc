// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <iostream>
#include <string>
#include <vector>

#include "grpc++/grpc++.h"
#include "grpc++/support/byte_buffer.h"
#include "grpc/support/log.h"
#include "paddle/fluid/framework/blocking_queue.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/operators/detail/rpc_processor.h"
#include "paddle/fluid/operators/detail/sendrecvop_utils.h"

namespace paddle {
namespace operators {
namespace detail {

void GRPCProcessorCtx::SetExit() {
  LOG(WARNING) << "GRPCProcessorCtx SetExit!";
  std::unique_lock<std::mutex> lock(mutex_);
  exit_flag_ = true;
  condition_send_.notify_all();
  condition_get_.notify_all();
}

void GRPCProcessorCtx::IncreaseBatchBarrierSend() {
  std::unique_lock<std::mutex> lock(mutex_);
  batch_barrier_send_++;
  if (batch_barrier_send_ >= fan_in_) {
    condition_send_.notify_all();
  }
}

void GRPCProcessorCtx::IncreaseBatchBarrierGet() {
  std::unique_lock<std::mutex> lock(mutex_);
  batch_barrier_get_++;
  if (batch_barrier_get_ >= fan_in_) {
    condition_get_.notify_all();
  }
}

bool GRPCProcessorCtx::ProcessSendImpl(const std::string& msg_name,
                                       framework::Variable* var,
                                       framework::Scope* scope) {
  if (msg_name == LISTEN_TERMINATE_MESSAGE) {
    LOG(INFO) << "received terminate message and exit";
    SetExit();
    return true;
  }

  // Async
  if (!sync_mode_) {
    try {
      executor_->RunPreparedContext((*grad_to_prepared_ctx_)[msg_name].get(),
                                    scope);
    } catch (std::exception& e) {
      LOG(ERROR) << "async: run sub program error " << e.what();
    }
    return true;
  }

  PADDLE_ENFORCE((fan_in_ > 0), "please set fan_in first!");
  // Sync
  if (msg_name == BATCH_BARRIER_MESSAGE) {
    VLOG(3) << "sync: recv batch barrier message";
    IncreaseBatchBarrierSend();
  } else {
    VLOG(3) << "sync: received var_name: " << msg_name;

    if (var == nullptr) {
      LOG(ERROR) << "sync: Can not find server side var: " << msg_name;
      PADDLE_THROW("sync: Can not find server side var");
    }

    if (var->IsType<framework::SelectedRows>()) {
      std::unique_lock<std::mutex> lock(mutex_);
      sparse_vars_.push_back(var);
    }
  }

  return true;
}

bool GRPCProcessorCtx::ProcessGetImpl(const std::string& msg_name,
                                      framework::Variable** var) {
  if (msg_name != FETCH_BARRIER_MESSAGE) {
    *var = scope_->FindVar(msg_name);
    return true;
  }

  sendrecv::VariableMessage msg;
  if (sync_mode_) {
    PADDLE_ENFORCE((fan_in_ > 0), "please set fan_in first!");
    IncreaseBatchBarrierGet();
  }
  return true;
}

bool GRPCProcessorCtx::ProcessPrefetchImpl(const std::string& msg_name,
                                           framework::Scope* scope,
                                           framework::Variable** var) {
  auto var_desc = program_->Block(0).FindVar(msg_name);
  *var = scope->FindVar(msg_name);

  InitializeVariable(*var, var_desc->GetType());
  executor_->RunPreparedContext(prefetch_ctx_.get(), scope_);

  return true;
}

}  // namespace detail
}  // namespace operators
}  // namespace paddle
