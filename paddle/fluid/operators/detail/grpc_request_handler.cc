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
#include "paddle/fluid/operators/detail/grpc_service.h"
#include "paddle/fluid/operators/detail/sendrecvop_utils.h"
#include "paddle/fluid/operators/detail/variable_response.h"

namespace paddle {
namespace operators {
namespace detail {

bool GRPCRequestHandler::HandlerImpl(int method_id, void* input, void* output) {
  switch (method_id) {
    case static_cast<int>(GrpcMethod::kSendVariable):
      return RequestSendHandler(input, output);
    case static_cast<int>(GrpcMethod::kGetVariable):
      return RequestGetHandler(input, output);
    case static_cast<int>(GrpcMethod::kPrefetchVariable):
      return RequestPrefetchHandler(input, output);
    default:
      PADDLE_ENFORCE(false, "not surpported method_id");
      return false;
  }
}

bool GRPCRequestHandler::RequestSendHandler(void* input, void* output) {
  VariableResponse* req = static_cast<VariableResponse*>(input);

  auto scope = req->GetMutableLocalScope();
  std::string msg_name = req->Varname();

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
      return false;
    }
    return true;
  }

  // Sync
  if (msg_name == BATCH_BARRIER_MESSAGE) {
    VLOG(3) << "sync: recv batch barrier message";
    IncreaseBatchBarrier();
  } else {
    VLOG(3) << "sync: received var_name: " << msg_name;

    framework::Variable* var = req->GetVar();
    if (var == nullptr) {
      LOG(ERROR) << "sync: Can not find server side var: " << msg_name;
      PADDLE_THROW("sync: Can not find server side var");
      return false;
    }

    if (var->IsType<framework::SelectedRows>()) {
      std::unique_lock<std::mutex> lock(mutex_);
      sparse_vars_.push_back(var);
    }
  }

  return true;
}

bool GRPCRequestHandler::RequestGetHandler(void* input, void* output) {
  sendrecv::VariableMessage* req =
      static_cast<sendrecv::VariableMessage*>(input);
  ::grpc::ByteBuffer* reply = static_cast<::grpc::ByteBuffer*>(output);
  std::string msg_name = req->varname();

  VLOG(3) << "ProcessGetImpl:" << msg_name;

  if (msg_name != FETCH_BARRIER_MESSAGE) {
    framework::Variable* var = scope_->FindVar(msg_name);
    SerializeToByteBuffer(msg_name, var, *dev_ctx_, reply);
  }

  // FETCH_BARRIER_MESSAGE
  if (sync_mode_) {
    VLOG(3) << "sync: recv fetch barrier message";
    IncreaseBatchBarrier();
  }

  return true;
}

bool GRPCRequestHandler::RequestPrefetchHandler(void* input, void* output) {
  VariableResponse* req = static_cast<VariableResponse*>(input);
  ::grpc::ByteBuffer* reply = static_cast<::grpc::ByteBuffer*>(output);

  std::string var_name = req->OutVarname();
  VLOG(3) << "RequestPrefetchHandler " << var_name;

  auto var_desc = program_->Block(0).FindVar(var_name);
  // FIXME(gongwb):need delete.
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
