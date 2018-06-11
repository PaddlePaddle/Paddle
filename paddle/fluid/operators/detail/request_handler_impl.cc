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

#include "paddle/fluid/framework/blocking_queue.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/operators/detail/request_handler_impl.h"
#include "paddle/fluid/operators/detail/rpc_server.h"
#include "paddle/fluid/operators/detail/sendrecvop_utils.h"
#include "paddle/fluid/operators/detail/variable_response.h"

namespace paddle {
namespace operators {
namespace detail {

bool RequestSendHandler::Handle(const std::string& varname,
                                framework::Scope* scope,
                                framework::Variable* invar,
                                framework::Variable** outvar) {
  VLOG(4) << "RequestSendHandler:" << varname;

  // Async
  if (!sync_mode_) {
    try {
      executor_->RunPreparedContext((*grad_to_prepared_ctx_)[varname].get(),
                                    scope);
    } catch (std::exception& e) {
      LOG(ERROR) << "async: run sub program error " << e.what();
      return false;
    }
    return true;
  }

  // Sync
  if (varname == BATCH_BARRIER_MESSAGE) {
    VLOG(3) << "sync: recv batch barrier message";
    rpc_server_->IncreaseBatchBarrier(kRequestSend);
  } else {
    VLOG(3) << "sync: received var_name: " << varname;
    if (sync_mode_) {
      rpc_server_->WaitCond(kRequestSend);
    }

    if (invar == nullptr) {
      LOG(ERROR) << "sync: Can not find server side var: " << varname;
      PADDLE_THROW("sync: Can not find server side var");
      return false;
    }
    if (invar->IsType<framework::SelectedRows>()) {
      std::unique_lock<std::mutex> lock(mutex_sparse_vars_);
      sparse_vars_.push_back(invar);
    }
  }
  return true;
}

void RequestSendHandler::ResetSparseVarRecorder() {
  std::unique_lock<std::mutex> lock(mutex_sparse_vars_);
  for (auto* var : sparse_vars_) {
    var->GetMutable<framework::SelectedRows>()->mutable_rows()->clear();
  }
  sparse_vars_.clear();
}

bool RequestGetHandler::Handle(const std::string& varname,
                               framework::Scope* scope,
                               framework::Variable* invar,
                               framework::Variable** outvar) {
  VLOG(4) << "RequestGetHandler:" << varname;

  if (varname != FETCH_BARRIER_MESSAGE) {
    if (sync_mode_) {
      rpc_server_->WaitCond(kRequestGet);
    }
    *outvar = scope_->FindVar(varname);
    return true;
  }

  // FETCH_BARRIER_MESSAGE
  if (sync_mode_) {
    VLOG(3) << "sync: recv fetch barrier message";
    rpc_server_->IncreaseBatchBarrier(kRequestGet);
  }

  return true;
}

bool RequestPrefetchHandler::Handle(const std::string& varname,
                                    framework::Scope* scope,
                                    framework::Variable* invar,
                                    framework::Variable** outvar) {
  VLOG(4) << "RequestPrefetchHandler " << varname;

  auto var_desc = program_->Block(0).FindVar(varname);
  *outvar = scope->FindVar(varname);
  InitializeVariable(*outvar, var_desc->GetType());
  executor_->RunPreparedContext(prefetch_ctx_.get(), scope);

  return true;
}

}  // namespace detail
}  // namespace operators
}  // namespace paddle
