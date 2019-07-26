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

#include "paddle/fluid/operators/distributed/handlers/signal_handler.h"

namespace paddle {
namespace operators {
namespace distributed {

using Scope = paddle::framework::Scope;

bool HandleSignal(RPCRequest *request, Scope *scope, RPCServer *rpc_server) {
  if (request->varname_ == FETCH_BARRIER_MESSAGE) {
    VLOG(4) << "server got FETCH_BARRIER_MESSAGE";
    rpc_server->RecvBarrier()->Increase();
    return true;
  } else if (request->varname_ == BATCH_BARRIER_MESSAGE) {
    VLOG(4) << "server got BATCH_BARRIER_MESSAGE";
    rpc_server->SendBarrier()->Increase();
    return true;
  } else if (request->varname_ == COMPLETE_MESSAGE) {
    VLOG(4) << "server got COMPLETE_MESSAGE";
    rpc_server->Complete();
    return true;
  }
  // no signal detected
  return false;
}

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
