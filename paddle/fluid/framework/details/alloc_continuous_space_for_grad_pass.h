//   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include <algorithm>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/details/build_strategy.h"
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/ir/graph_helper.h"

namespace paddle {
namespace framework {
namespace details {

void SetFuseParameterGroupsSize(int group_size);
int GetFuseParameterGroupsSize();

void SetFuseParameterMemorySize(uint64_t memory_size);
uint64_t GetFuseParameterMemorySize();

class AllocContinuousSpaceForGradPass : public ir::Pass {
 protected:
  void ApplyImpl(ir::Graph *graph) const override;

  template <typename AttrType>
  void ResetAttribute(const std::string &attr_name, ir::Graph *graph) const;

  void SetGroupGradsAndParams(
      const std::unordered_map<std::string, ir::Node *> &var_nodes,
      const ParamsAndGrads &params_grads,
      GroupGradsAndParams *group_grads_params) const;

  void SetGroupAccordingToLayers(
      const std::unordered_map<std::string, ir::Node *> &var_nodes,
      const ParamsAndGrads &params_grads,
      GroupGradsAndParams *group_grads_params) const;

  void SetGroupAccordingToMemorySize(
      const std::unordered_map<std::string, ir::Node *> &var_nodes,
      GroupGradsAndParams *group_grads_params) const;

  void SetGroupAccordingToGroupSize(
      const std::unordered_map<std::string, ir::Node *> &var_nodes,
      GroupGradsAndParams *group_grads_params) const;

 private:
  bool IsSupportedVarType(const proto::VarType::Type &type) const;

  void RecordParamsAndGrads(ir::Node *node, ParamsAndGrads *params_grads) const;

  void InitFusedVarsAndAllocSpaceForVars(
      const std::vector<platform::Place> &places,
      const std::vector<Scope *> &local_scopes,
      const std::unordered_map<std::string, ir::Node *> &vars,
      const std::string &fused_var_name,
      const ParamsAndGrads &params_grads) const;

  void AppendAllocSpaceForVarsOp(const std::vector<std::string> &params_name,
                                 const std::vector<std::string> &grads_name,
                                 const std::string &fused_var_name,
                                 BlockDesc *global_block) const;
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
