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

#include "paddle/fluid/operators/distributed/handlers/get_monomer_handler.h"

#include <string>
#include "paddle/fluid/operators/distributed/handlers/signal_handler.h"
#include "paddle/fluid/operators/distributed/rpc_server.h"

namespace paddle {
namespace operators {
namespace distributed {

bool GetMonomerHandler::Handle(RPCRequest *request, Scope *scope) {
  if (HandleSignal(request, scope, rpc_server_)) {
    return true;
  }
  // NOTE: get monomer do not depend on server state (barriers).
  rpc_server_->WaitVarReady(request->varname_);
  *(request->out_var_) = scope->FindVar(request->varname_);
  return true;
}

bool GetMonomerBarrierHandler::Handle(RPCRequest *request, Scope *scope) {
  if (HandleSignal(request, scope, rpc_server_)) {
    return true;
  }
  // FIXME(typhoonzero): need wait?
  rpc_server_->WaitVarReady(request->varname_);
  auto *bar = rpc_server_->VarReadyBarrier(request->varname_);
  if (bar) bar->Increase();
  return true;
}

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
