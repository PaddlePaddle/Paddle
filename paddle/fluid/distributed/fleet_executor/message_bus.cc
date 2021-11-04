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

#include "paddle/fluid/distributed/fleet_executor/message_bus.h"
#include "paddle/fluid/distributed/fleet_executor/carrier.h"

namespace paddle {
namespace distributed {

MessageBus::~MessageBus() {
  // destroy
}

bool MessageBus::Send(const InterceptorMessage& interceptor_message) {
  // called by Interceptor, send InterceptorMessage to dst
  return true;
}

void MessageBus::ListenPort() {
  // function keep listen the port and handle the message
}

bool MessageBus::IsSameRank(int64_t src_id, int64_t dst_id) {
  // check whether the dst is the same rank or different rank with src
  return true;
}

#ifndef PADDLE_WITH_ASCEND_CL
#ifdef PADDLE_WITH_DISTRIBUTE
bool MessageBus::SendInterRank(const InterceptorMessage& interceptor_message) {
  // send the message inter rank (dst is different rank with src)
  return true;
}
#endif
#endif

bool MessageBus::SendIntraRank(const InterceptorMessage& interceptor_message) {
  // send the message intra rank (dst is the same rank with src)
  return true;
}

}  // namespace distributed
}  // namespace paddle
