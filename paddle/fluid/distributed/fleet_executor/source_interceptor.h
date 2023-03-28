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
#include "paddle/fluid/distributed/fleet_executor/compute_interceptor.h"

namespace paddle {
namespace distributed {

/*
 * Source interceptor
 * There is only one source in the runtime graph
 * Take charge of:
 *   1. receive `start` message from carrier
 *   2. send num_of_steps `data_is_ready` message to downstream
 */
class SourceInterceptor final : public ComputeInterceptor {
 public:
  SourceInterceptor(int64_t interceptor_id, TaskNode* node);

 private:
  void Compute(const InterceptorMessage& msg) override;
  void SendDataReadyToDownStream() override;
  void PrepareDeps();
  void Run();
  int64_t max_run_times_;
  // cur_step
  int64_t step_{0};
  // carrier_id-->(scope_id, is_ready)
  // std::map<int32_t, std::vector<std::pair<int32_t, bool>>>
  // carriers_scopes_ready_;
};

}  // namespace distributed
}  // namespace paddle
