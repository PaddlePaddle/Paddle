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

#include <iomanip>
#include <queue>
#include "paddle/fluid/distributed/fleet_executor/interceptor.h"

namespace paddle {
namespace distributed {

/* Condition Interceptor
 * This is a special interceptor and only one condition op in the task node.
 * This interceptor has two downstreams,
 *  1. If the program result is true, select one of the downstreams, otherwise
 * select another.
 *  2. Used to implement while op in program.
 */
class CondInterceptor final : public Interceptor {
 public:
  CondInterceptor(int64_t interceptor_id, TaskNode* node);

 private:
  void PrepareDeps();
  void Run(const InterceptorMessage& msg);
  void Compute(int64_t gen_step);
  bool GetCondResult();
  void SendDataReady(int64_t down_id);
  void SendStartLoop(int64_t down_id, int64_t gen_step);
  void ReplyDataIsUseless(int64_t up_id);

  int64_t cur_scope_id_;

  std::set<int64_t> normal_in_id_;
  std::set<int64_t> normal_out_id_;
  int64_t stop_loop_id_;
  int64_t loop_id_;
  std::map<int64_t, int64_t> scope_id_to_gen_step_;
  int64_t start_micro_step_;
  int64_t num_micro_step_;
};

}  // namespace distributed
}  // namespace paddle
