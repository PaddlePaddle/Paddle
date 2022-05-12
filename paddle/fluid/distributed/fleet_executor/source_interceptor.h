// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/distributed/fleet_executor/interceptor.h"

namespace paddle {
namespace distributed {

/*
 * Source interceptor
 * There is only one source in the runtime graph
 * Take charge of:
 *   1. receive `start` message from carrier
 *   2. send num_of_steps `data_is_ready` message to downstream
 */
class SourceInterceptor : public Interceptor {
 public:
  SourceInterceptor(int64_t interceptor_id, TaskNode* node);

 private:
  void SendDataReadyToDownStream(int64_t down_id);
  void Run(const InterceptorMessage& msg);
  int64_t max_run_times_;
  // downstream_id->cur_step
  std::map<int64_t, int64_t> downstream_step_;
};

}  // namespace distributed
}  // namespace paddle
