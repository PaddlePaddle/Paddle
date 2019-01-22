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

#include "paddle/fluid/operators/distributed/handlers/send_handler.h"
#include "paddle/fluid/operators/distributed/handlers/signal_handler.h"
#include "paddle/fluid/operators/distributed/rpc_server.h"

namespace paddle {
namespace operators {
namespace distributed {

framework::Variable* SendHandlerSync::GetOrCreateRequestVar(
    const std::string& varname, RPCRequest* request) {
  return scope_->FindVar(varname);
}

bool SendHandlerSync::Handle(RPCRequest* request) {
  VLOG(3) << "sync: processing received var: " << request->varname_;
  rpc_server_->WaitState(RPCServerState::STATE_SEND);
  if (HandleSignal(request, scope_, rpc_server_)) {
    return true;
  }

  if (request->var_ == nullptr) {
    LOG(FATAL) << "sync: Can not find server side var: " << request->varname_;
    return false;
  }
  return true;
}

framework::Variable* SendHandlerAsync::GetOrCreateRequestVar(
    const std::string& varname, RPCRequest* request) {
  request->scope_ = &scope_->NewScope();
  // create new var for each async request because the var name from
  // every worker is the same.
  return request->scope_->Var(varname);
}

bool SendHandlerAsync::Handle(RPCRequest* request) {
  VLOG(3) << "async process var: " << request->varname_;
  if (HandleSignal(request, scope_, rpc_server_)) {
    return true;
  }
  bool ret = true;
  try {
    executor_->RunPreparedContext(
        (*grad_to_prepared_ctx_)[request->varname_].get(), request->scope_);
  } catch (std::exception& e) {
    LOG(ERROR) << "async: run sub program error " << e.what();
    ret = false;
  }
  return ret;
}

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
