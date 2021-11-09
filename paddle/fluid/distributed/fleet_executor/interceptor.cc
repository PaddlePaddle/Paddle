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

#include "paddle/fluid/distributed/fleet_executor/interceptor.h"

namespace paddle {
namespace distributed {

Interceptor::Interceptor(int64_t interceptor_id_, TaskNode* node) {
  // init
}

std::condition_variable& Interceptor::GetCondVar() {
  // get the conditional var
  return cond_var_;
}

int64_t Interceptor::GetInterceptorId() const {
  // return the interceptor id
  return 0;
}

bool Interceptor::EnqueueRemoteInterceptorMessage(
    const InterceptorMessage& interceptor_message) {
  // Called by Carrier, enqueue an InterceptorMessage to remote mailbox
  return true;
}

void Interceptor::PoolTheMailbox() {
  // pool the local mailbox, parse the Message
}

bool Interceptor::FetchRemoteMailbox() {
  // fetch all Message from remote mailbox to local mailbox
  // return true if remote mailbox not empty, otherwise return false
  return true;
}

}  // namespace distributed
}  // namespace paddle
