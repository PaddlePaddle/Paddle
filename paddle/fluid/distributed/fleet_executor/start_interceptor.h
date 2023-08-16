// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <chrono>
#include <utility>

#include "paddle/fluid/distributed/fleet_executor/compute_interceptor.h"

namespace paddle {
namespace distributed {

class StartInterceptor final : public ComputeInterceptor {
 public:
  StartInterceptor(int64_t interceptor_id, TaskNode* node);

 private:
  void SendDataReadyToDownStream() override;
  void RunOps() override;
  void Compute(const InterceptorMessage& msg) override;

  int64_t batch_size_{0};
  int64_t finish_count_{0};
  int64_t step_{0};
  std::chrono::time_point<std::chrono::system_clock> start_time_{
      std::chrono::system_clock::now()};
};

}  // namespace distributed
}  // namespace paddle
