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

#include <queue>
#include <utility>

#include "paddle/fluid/distributed/fleet_executor/interceptor.h"

namespace paddle {
namespace distributed {

const int64_t INFINITE_BUFFER_SIZE = -1;

class ComputeInterceptor : public Interceptor {
 public:
  ComputeInterceptor(int64_t interceptor_id, TaskNode* node);

 protected:
  virtual void RunOps();
  virtual void SendDataReadyToDownStream();
  virtual void ReplyCompletedToUpStream();

  std::queue<int64_t> ready_queue_;
  int64_t cur_scope_id_;

 private:
  void PrepareDeps();

  void IncreaseReady(int64_t up_id);
  void DecreaseBuff(int64_t down_id);
  bool IsInputReady();
  bool CanWriteOutput();

  void Run();
  void Compute(const InterceptorMessage& msg);

  // upstream_id-->(max_ready_size, ready_size)
  std::map<int64_t, std::pair<int64_t, int64_t>> in_readys_{};
  // downstream_id-->(max_buffer_size, used_size)
  std::map<int64_t, std::pair<int64_t, int64_t>> out_buffs_{};
};

}  // namespace distributed
}  // namespace paddle
