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

#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "paddle/fluid/distributed/fleet_executor/interceptor_message.pb.h"

namespace paddle {
namespace distributed {

class Interceptor;
class TaskNode;
class InterceptorMessageServiceImpl;

class Carrier final {
 public:
  Carrier() = delete;

  Carrier(
      const std::unordered_map<int64_t, TaskNode*>& interceptor_idx_to_node);

  ~Carrier();

  // Enqueue a message to corresponding interceptor id
  bool EnqueueInterceptorMessage(const InterceptorMessage& interceptor_message);

  DISABLE_COPY_AND_ASSIGN(Carrier);

 private:
  // create each Interceptor
  void CreateInterceptor();

  // get interceptor based on the interceptor id
  Interceptor* GetInterceptor(int64_t interceptor_id);

  // interceptor logic id to the Nodes info
  std::unordered_map<int64_t, TaskNode*> interceptor_idx_to_node_;

  // interceptor logic id to actually interceptor
  std::unordered_map<int64_t, std::unique_ptr<Interceptor>>
      interceptor_idx_to_interceptor_;
};

}  // namespace distributed
}  // namespace paddle
