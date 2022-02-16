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

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/platform/profiler.h"
#include "paddle/fluid/platform/profiler/event_tracing.h"
#include "paddle/fluid/platform/variant.h"

namespace paddle {
namespace framework {

class CostData {
 public:
  CostData() {}

  ~CostData();

  // Support global block only
  // TODO(zhhsplendid): add support for sub-block
  double GetOpTimeMs(int op_id) const;
  double GetOpMemoryBytes(int op_id) const;
  double GetWholeTimeMs() const;
  double GetWholeMemoryBytes() const;

  const ir::Graph* GetGraph() const;
  const ProgramDesc* GetProgram() const;

  // Support Time Event only
  // TODO(zhhsplendid): add memory
  bool SetCostData(
      const ProgramDesc& program,
      const std::vector<std::vector<platform::Event>>& time_events);

  static const double NOT_MEASURED;

 private:
  ir::Graph* graph_{nullptr};
  ProgramDesc* program_{nullptr};
  std::map<int, double> op_time_ms_;  // from Op Node id to time
  std::map<int, double>
      op_memory_bytes_;         // from Op Node id to total memory bytes
  std::map<int, double> comm_;  // from Op Node id to communicate cost
  double whole_time_ms_{
      NOT_MEASURED};  // time cost of the whole program or graph
  double whole_memory_bytes_{
      NOT_MEASURED};  // memory cost of the whole program or graph
  double whole_comm_{
      NOT_MEASURED};  // communication cost of the whole program or graph
};

class CostModel {
 public:
  CostModel() {}
  ~CostModel() {}

  CostData ProfileMeasure(
      const ProgramDesc& main_program, const ProgramDesc& startup_program,
      const std::string& device,
      const std::vector<std::string>& fetch_cost_list) const;
};

}  // namespace framework
}  // namespace paddle
