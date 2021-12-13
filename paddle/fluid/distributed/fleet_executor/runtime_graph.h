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
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/distributed/fleet_executor/fleet_executor_desc.pb.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace framework {
class ProgramDesc;
class OperatorBase;
}

namespace distributed {
class TaskNode;

class RuntimeGraph final {
 public:
  using ProgramDesc = paddle::framework::ProgramDesc;
  using OperatorBase = paddle::framework::OperatorBase;
  RuntimeGraph() = default;
  explicit RuntimeGraph(const ProgramDesc& program,
                        const FleetExecutorDesc& exe_desc);
  ~RuntimeGraph() = default;
  const std::unordered_map<int64_t, TaskNode*>& intercepter_id_to_node() const {
    return intercepter_id_to_node_;
  }
  const std::unordered_map<int64_t, int64_t>& intercepter_id_to_rank() const {
    return intercepter_id_to_rank_;
  }
  std::string DebugString() const;

 private:
  DISABLE_COPY_AND_ASSIGN(RuntimeGraph);
  void SplitProgramBasedFunctionality(const ProgramDesc& program);
  void FakeDependence();
  void AssignTaskToIntercepter();
  void FakeRuntimeInfo();
  void OriginProgramCompile(const ProgramDesc& program);
  // LRSched, Forward, Backward, Optimize
  static std::vector<paddle::framework::OpRole> functionality_order;
  std::vector<std::unique_ptr<TaskNode>> task_nodes_;
  std::vector<std::unique_ptr<OperatorBase>> ops_;
  std::unordered_map<int64_t, TaskNode*> intercepter_id_to_node_;
  std::unordered_map<int64_t, int64_t> intercepter_id_to_rank_;
  FleetExecutorDesc exe_desc_;
};

}  // namespace distributed
}  // namespace paddle
