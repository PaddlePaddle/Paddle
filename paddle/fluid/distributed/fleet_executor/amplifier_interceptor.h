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

#include <utility>

#include "paddle/fluid/distributed/fleet_executor/compute_interceptor.h"

namespace paddle {
namespace distributed {

class AmplifierInterceptor : public ComputeInterceptor {
 public:
  AmplifierInterceptor(int64_t interceptor_id, TaskNode* node);

 private:
  void RunOps() override;
  void SendDataReadyToDownStream() override;
  void ReplyCompletedToUpStream() override;

  int64_t run_per_steps_{1};
  int64_t run_at_offset_{0};

  // one input produces multi times output
  int64_t reply_up_per_steps_{1};
  // one output need multi times input
  int64_t send_down_per_steps_{1};
};

}  // namespace distributed
}  // namespace paddle
