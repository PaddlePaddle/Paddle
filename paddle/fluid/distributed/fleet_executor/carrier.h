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
#include <vector>

#include "paddle/fluid/distributed/fleet_executor/interceptor.h"
#include "paddle/fluid/distributed/fleet_executor/interceptor_message.pb.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/errors.h"
#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace distributed {

class TaskNode;
class InterceptorMessageServiceImpl;

// A singleton MessageBus
class Carrier final {
 public:
  static Carrier& Instance() {
    static Carrier carrier;
    return carrier;
  }

  void Init(
      const std::unordered_map<int64_t, TaskNode*>& interceptor_id_to_node);

  ~Carrier() = default;

  // Enqueue a message to corresponding interceptor id
  bool EnqueueInterceptorMessage(const InterceptorMessage& interceptor_message);

  // get interceptor based on the interceptor id
  Interceptor* GetInterceptor(int64_t interceptor_id);

  // set interceptor with interceptor id
  Interceptor* SetInterceptor(int64_t interceptor_id,
                              std::unique_ptr<Interceptor>);

  void SetCreatingFlag(bool flag);

  void Start();

  bool IsInit() const;

  DISABLE_COPY_AND_ASSIGN(Carrier);

 private:
  Carrier() = default;

  // create each Interceptor
  void CreateInterceptors();

  void HandleTmpMessages();

  // interceptor logic id to the Nodes info
  std::unordered_map<int64_t, TaskNode*> interceptor_id_to_node_;

  // interceptor logic id to actually interceptor
  std::unordered_map<int64_t, std::unique_ptr<Interceptor>>
      interceptor_idx_to_interceptor_;

  std::vector<InterceptorMessage> message_tmp_{};
  bool creating_interceptors_{true};
  bool is_init_{false};
};

}  // namespace distributed
}  // namespace paddle
