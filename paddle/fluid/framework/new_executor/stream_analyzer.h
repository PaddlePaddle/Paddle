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
#include <future>
#include <memory>
#include <vector>

#include "paddle/fluid/framework/new_executor/new_executor_defs.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/device_event.h"

namespace paddle {
namespace framework {

class StreamAnalyzer {
 public:
  using Place = platform::Place;
  using DeviceContext = platform::DeviceContext;

  explicit StreamAnalyzer(const Place& place);

  ~StreamAnalyzer() {}

  void Schedule(const std::vector<size_t>& downstream_ops,
                std::vector<Instruction>* instructions,
                size_t op_index);

  DeviceContext* ParseDeviceContext(const OpFuncNode& op_func_node);

 private:
  std::vector<size_t> GetNeedEventVarIds(const Instruction& cur_instr,
                                         const Instruction& next_instr);

  void ConstructEventForVar(const std::vector<size_t>& new_event_var_id,
                            Instruction* next_instr,
                            platform::DeviceType waiter_type,
                            const Place& place);

  bool IsDirectRun(Instruction& cur_instr,  // NOLINT
                   const Instruction& next_instr);

  platform::DeviceType GetWaiterType(const Instruction& instr);

  const Place place_;
  std::shared_future<std::unique_ptr<platform::DeviceContext>> d2h_ctx_;
  std::shared_future<std::unique_ptr<platform::DeviceContext>> h2d_ctx_;
  std::map<size_t, std::shared_ptr<platform::DeviceEvent>> var_id2event_;
};

}  // namespace framework
}  // namespace paddle
