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
 * Sink interceptor
 * There is only one sink in the runtime graph
 * Take charge of:
 *   1. record the num of micro-step
 *   2. check whether to notify carrier the current step is finished
 */
class SinkInterceptor : public Interceptor {
 public:
  SinkInterceptor(int64_t interceptor_id, TaskNode* node);

 private:
  void ReplyCompletedToUpStream(int64_t up_id);
  void Run(const InterceptorMessage& msg);
  void StopCarrierIfComplete();
  int64_t max_run_times_;
  // upstream_id->cur_step
  std::map<int64_t, int64_t> upstream_step_;
};
}  // namespace distributed
}  // namespace paddle
