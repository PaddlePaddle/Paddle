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
#include <future>
#include <memory>
#include <vector>

#include "paddle/fluid/framework/new_executor/interpreter/dependency_builder.h"
#include "paddle/fluid/framework/new_executor/new_executor_defs.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/device_event.h"

namespace paddle {
namespace framework {
namespace interpreter {

enum DownstreamRunType { kDirectRun, kEventRun };

class StreamAnalyzer {
 public:
  using DeviceContext = platform::DeviceContext;
  using Place = platform::Place;

  explicit StreamAnalyzer(const Place& place) : place_(place) {}

  ~StreamAnalyzer() {}

  void ConstructEvents(std::vector<Instruction>* instructions) const;

  platform::DeviceContext* ParseDeviceContext(
      const OpFuncNode& op_func_node) const;

  platform::DeviceType GetWaiterType(const Instruction& instr) const;

 private:
  bool HasDataDependency(const Instruction& cur_instr,
                         const Instruction& next_instr) const;

  void AnalyseAllEventInfo(
      const std::vector<Instruction>& instructions,
      const std::vector<std::vector<std::vector<size_t>>>& run_type_info,
      std::map<const DeviceContext*, std::map<size_t, std::set<size_t>>>*
          event_info) const;

  void AnalyseAllRunType(
      const std::vector<Instruction>& instructions,
      const std::map<size_t, std::set<size_t>>& downstream_map,
      std::vector<std::vector<std::vector<size_t>>>* run_type_info) const;

  void AnalyseEventInfoForTwoInstructions(
      const std::vector<Instruction>& instructions,
      const std::vector<std::vector<std::vector<size_t>>>& run_type_info,
      const size_t cur_instr_id,
      const size_t next_instr_id,
      std::set<size_t>* waiter_instr_ids) const;

  void ShrinkEventInfo(
      const DependencyBuilder& dependency_builder,
      std::map<const DeviceContext*, std::map<size_t, std::set<size_t>>>*
          event_info_map) const;

  DownstreamRunType AnalyseRunTypeForTwoInstructions(
      const Instruction& cur_instr, const Instruction& next_instr) const;

  const Place place_;
};

}  // namespace interpreter
}  // namespace framework
}  // namespace paddle
