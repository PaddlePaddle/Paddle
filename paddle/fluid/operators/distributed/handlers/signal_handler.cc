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

bool SignalHandler::Handle(RPCRequest *request, Scope *scope) {
  if (request->varname_ == FETCH_BARRIER_MESSAGE) {
    VLOG(3) << "recv fetch barrier message";
    rpc_server_->IncreaseBatchBarrier(kRequestGet);
  } else if (request->varname_ == BATCH_BARRIER_MESSAGE) {
    VLOG(3) << "recv BATCH_BARRIER_MESSAGE";
    rpc_server_->IncreaseBatchBarrier(kRequestSend);
  } else if (request->varname_ == COMPLETE_MESSAGE) {
    VLOG(3) << "recv complete message";
    rpc_server_->Complete();
  }
  return true;
}

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
