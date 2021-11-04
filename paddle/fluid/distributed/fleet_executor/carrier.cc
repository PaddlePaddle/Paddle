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

#include "paddle/fluid/distributed/fleet_executor/carrier.h"
#include "paddle/fluid/distributed/fleet_executor/interceptor.h"
#include "paddle/fluid/distributed/fleet_executor/interceptor_message_service.h"
#include "paddle/fluid/distributed/fleet_executor/task_node.h"

namespace paddle {
namespace distributed {

Carrier::Carrier(
    const std::unordered_map<int64_t, TaskNode*>& interceptor_id_to_node) {
  // init
}

Carrier::~Carrier() {
  // destroy
}

bool Carrier::EnqueueInterceptorMessage(
    const InterceptorMessage& interceptor_message) {
  // enqueue message to interceptor
  return true;
}

void Carrier::CreateInterceptors() {
  // create each Interceptor
}

}  // namespace distributed
}  // namespace paddle
