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

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/details/build_strategy.h"
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/ir/graph.h"

namespace paddle {
namespace framework {
namespace ir {

constexpr char kGrad[] = "Grad";
constexpr char kParam[] = "Param";
constexpr char kLearningRate[] = "LearningRate";

class FuseOptimizerOpPass : public ir::Pass {
 protected:
  void ApplyImpl(ir::Graph *graph) const override;

 protected:
  virtual void SortParametersAndAuxVars(
      const std::vector<std::pair<std::string, std::string>> &params_grads,
      std::unordered_map<std::string, std::vector<std::string>> *aux_var_set,
      std::vector<ir::Node *> *ops) const;

  void InserInputAndOutputForOptOps(const std::vector<ir::Node *> &opt_ops,
                                    ir::Node *opt_node) const;

 private:
  virtual const std::string GetOpType() const = 0;

  virtual const std::vector<std::string> GetAuxiliaryVarNames() const = 0;

  virtual void FuseOptimizerOps(
      const std::unordered_map<std::string, std::vector<std::string>> &vars_set,
      const std::unordered_map<std::string, std::string> &fused_vars_name,
      const std::vector<ir::Node *> &adam_ops, ir::Graph *graph) const = 0;

  void GetSpecifiedOpsAndVars(
      const std::string &op_type, const std::vector<std::string> &aux_vars_name,
      ir::Node *node, std::vector<ir::Node *> *ops,
      std::unordered_map<std::string, std::vector<std::string>> *aux_args_name)
      const;

  void AppendAllocContinuousSpace(const std::vector<std::string> &in_args,
                                  const std::vector<std::string> &out_args,
                                  const std::string &fused_out_arg,
                                  BlockDesc *global_block, bool copy_data,
                                  bool check_name = true) const;

  void InitFusedGradsAndAllocSpaceForGrads(
      const std::vector<platform::Place> &places,
      const std::vector<Scope *> &local_scopes,
      const std::vector<std::string> &params,
      const std::vector<std::string> &grads, const std::string &fused_grad_name,
      ir::Graph *result) const;

  void InitFusedVarsAndAllocSpaceForVars(
      const std::vector<platform::Place> &places,
      const std::vector<Scope *> &local_scopes,
      const std::vector<std::string> &aux_var_names,
      const std::unordered_map<std::string, std::vector<std::string>>
          &aux_var_set,
      const std::unordered_map<std::string, std::string> &fused_vars_name)
      const;

  void RunInitOps(const std::vector<platform::Place> &places,
                  const std::vector<Scope *> &local_scopes,
                  const BlockDesc &global_block) const;

  void InitVars(const std::vector<Scope *> &local_scopes,
                const std::string &fused_var_name) const;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
