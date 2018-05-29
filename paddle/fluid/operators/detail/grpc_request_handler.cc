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
#include "paddle/fluid/operators/detail/grpc_request_handler.h"
#include "paddle/fluid/operators/detail/grpc_server.h"
#include "paddle/fluid/operators/detail/grpc_service.h"
#include "paddle/fluid/operators/detail/sendrecvop_utils.h"
#include "paddle/fluid/operators/detail/variable_response.h"

namespace paddle {
namespace operators {
namespace detail {

bool GrpcRequestSendHandler::Handle(void* input, void* output) {
  VariableResponse* req = static_cast<VariableResponse*>(input);

  auto scope = req->GetMutableLocalScope();
  std::string msg_name = req->Varname();

  // Async
  if (!sync_mode_) {
    try {
      executor_->RunPreparedContext((*grad_to_prepared_ctx_)[msg_name].get(),
                                    scope);
    } catch (std::exception& e) {
      LOG(ERROR) << "async: run sub program error " << e.what();
      return false;
    }
    return true;
  }

  // Sync
  if (msg_name == BATCH_BARRIER_MESSAGE) {
    VLOG(3) << "sync: recv batch barrier message";
    rpc_server_->IncreaseBatchBarrier(kRequestSend);
  } else {
    VLOG(3) << "sync: received var_name: " << msg_name;
    if (sync_mode_) {
      rpc_server_->WaitCond(kRequestSend);
    }

    framework::Variable* var = req->GetVar();
    if (var == nullptr) {
      LOG(ERROR) << "sync: Can not find server side var: " << msg_name;
      PADDLE_THROW("sync: Can not find server side var");
      return false;
    }

    if (var->IsType<framework::SelectedRows>()) {
      std::unique_lock<std::mutex> lock(sparse_var_mutex_);
      sparse_vars_.push_back(var);
    }
  }

  return true;
}

bool GrpcRequestGetHandler::Handle(void* input, void* output) {
  sendrecv::VariableMessage* req =
      static_cast<sendrecv::VariableMessage*>(input);
  ::grpc::ByteBuffer* reply = static_cast<::grpc::ByteBuffer*>(output);
  std::string msg_name = req->varname();

  VLOG(3) << "ProcessGetImpl:" << msg_name;

  if (msg_name != FETCH_BARRIER_MESSAGE) {
    if (sync_mode_) {
      rpc_server_->WaitCond(kRequestGet);
    }
    framework::Variable* var = scope_->FindVar(msg_name);
    SerializeToByteBuffer(msg_name, var, *dev_ctx_, reply);
    return true;
  }

  // FETCH_BARRIER_MESSAGE
  if (sync_mode_) {
    VLOG(3) << "sync: recv fetch barrier message";
    rpc_server_->IncreaseBatchBarrier(kRequestGet);
  }

  return true;
}

bool GrpcRequestPrefetchHandler::Handle(void* input, void* output) {
  VariableResponse* req = static_cast<VariableResponse*>(input);
  ::grpc::ByteBuffer* reply = static_cast<::grpc::ByteBuffer*>(output);

  std::string var_name = req->OutVarname();
  VLOG(3) << "RequestPrefetchHandler " << var_name;

  auto var_desc = program_->Block(0).FindVar(var_name);
  framework::Scope* local_scope = &scope_->NewScope();
  auto* var = local_scope->FindVar(var_name);
  InitializeVariable(var, var_desc->GetType());
  executor_->RunPreparedContext(prefetch_ctx_.get(), scope_);

  SerializeToByteBuffer(var_name, var, *dev_ctx_, reply);

  return true;
}

}  // namespace detail
}  // namespace operators
}  // namespace paddle
