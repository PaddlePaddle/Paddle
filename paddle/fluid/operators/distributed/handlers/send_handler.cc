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

bool Handle(std::function<void(framework::Scope *)> start,
            std::function<void()> finish) {
  auto request = start(scope_);
  if (HandleSignal(request, scope, rpc_server_)) {
    finish();
    return true;
  }
  rpc_server_->WaitState(RPCServerState::STATE_SEND);
  VLOG(3) << "sync: processing received var: " << request->varname_;

  if (request->var_ == nullptr) {
    LOG(FATAL) << "sync: Can not find server side var: " << request->varname_;
    finish();
    return false;
  }
  finish();
  return true;
}

bool Handle(std::function<void(framework::Scope *)> start,
            std::function<void()> finish) {
  auto *local_scope = scope_->NewScope();
  auto request = start(local_scope);
  VLOG(3) << "async process var: " << request->varname_;
  if (HandleSignal(request, local_scope, rpc_server_)) {
    finish();
    return true;
  }
  try {
    executor_->RunPreparedContext(
        (*grad_to_prepared_ctx_)[request->varname_].get(), local_scope);
  } catch (std::exception &e) {
    LOG(ERROR) << "async: run sub program error " << e.what();
    finish();
    return false;
  }
  finish();
  return true;
}

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
