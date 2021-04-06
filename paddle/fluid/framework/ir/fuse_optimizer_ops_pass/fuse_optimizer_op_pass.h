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
class BlockDesc;
class VarDesc;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace ir {

class Graph;
class Node;

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

  void InsertInputAndOutputForFusedOpNode(
      const std::vector<ir::Node *> &opt_ops, ir::Graph *graph,
      ir::Node *opt_node) const;

 private:
  virtual const std::string GetOpType() const = 0;

  virtual const std::vector<std::string> GetAuxiliaryVarNames() const = 0;

  virtual ir::Node *FuseOptimizerOps(
      const std::unordered_map<std::string, std::vector<std::string>> &vars_set,
      const std::unordered_map<std::string, std::string> &fused_vars_name,
      const std::vector<ir::Node *> &adam_ops, ir::Graph *graph) const = 0;

  void GetFusingVarNamesMap(
      const std::vector<std::string> &aux_vars_name,
      const std::vector<ir::Node *> &opt_nodes,
      std::unordered_map<std::string, std::vector<std::string>> *aux_args_name)
      const;

  void AppendCoalesceTensorOp(const std::vector<std::string> &in_args,
                              const std::vector<std::string> &out_args,
                              const std::string &fused_out_arg,
                              const proto::VarType::Type &dtype,
                              BlockDesc *global_block, bool copy_data,
                              bool check_name = true) const;

  void FuseGradientsToContinuousSpace(const std::vector<std::string> &params,
                                      const std::vector<std::string> &grads,
                                      const std::string &fused_grad_name,
                                      const proto::VarType::Type &dtype,
                                      ir::Graph *result) const;

  void FuseVarsToContinuousSpace(
      const std::vector<std::string> &aux_var_names,
      const std::unordered_map<std::string, std::vector<std::string>>
          &aux_var_set,
      const std::unordered_map<std::string, std::string> &fused_vars_name,
      const proto::VarType::Type &dtype, ir::Graph *result) const;

  std::unordered_map<std::string, std::vector<Node *>> GetVarInfo(
      const Graph &result) const;

  bool OpWithKernelSupportCPUAndGPU(const std::string &op_type) const;

  bool GradGeneratedOpKernelCheck(
      const std::unordered_map<std::string, std::vector<ir::Node *>> &vars_info,
      const std::string &grad_var_name) const;

  proto::VarType::Type GetDtypeOfVar(
      const std::unordered_map<std::string, std::vector<ir::Node *>> &vars_info,
      const std::string &name) const;

  proto::VarType::Type GetTypeOfVar(
      const std::unordered_map<std::string, std::vector<Node *>> &var_nodes,
      const std::string &name) const;

  const VarDesc *GetVarDescFromVarsInfo(
      const std::unordered_map<std::string, std::vector<Node *>> &vars_info,
      const std::string &var_name) const;

  void GradientsFilter(const std::vector<size_t> &new_grad_idx,
                       std::vector<Node *> *opt_nodes,
                       std::unordered_map<std::string, std::vector<std::string>>
                           *aux_var_set) const;

  bool IsLoDTensorType(const proto::VarType::Type &type) const;

  bool HasVarDepsBetweenOps(const std::vector<Node *> &topo_nodes,
                            const std::vector<Node *> &opt_nodes) const;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
