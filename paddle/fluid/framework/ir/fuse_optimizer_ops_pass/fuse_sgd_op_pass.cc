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
#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/ir/fuse_optimizer_ops_pass/fuse_optimizer_op_pass.h"
#include "paddle/fluid/framework/op_registry.h"
namespace paddle {
namespace framework {
namespace ir {

class FuseSgdOpPass : public FuseOptimizerOpPass {
 private:
  virtual const std::string GetOpType() const { return "sgd"; }

  virtual const std::vector<std::string> GetAuxiliaryVarNames() const {
    return {};
  }

  // Fuse Sgd Ops
  virtual ir::Node *FuseOptimizerOps(
      const std::unordered_map<std::string, std::vector<std::string>> &vars_set,
      const std::unordered_map<std::string, std::string> &fused_vars_name,
      const std::vector<ir::Node *> &sgd_ops, ir::Graph *graph) const {
    PADDLE_ENFORCE_GT(sgd_ops.size(), static_cast<size_t>(0));

    // NOTE: fused_var is only exist in scope, so the graph doesn't have
    // fused_var node.

    int op_role = boost::get<int>(
        sgd_ops[0]->Op()->GetAttr(OpProtoAndCheckerMaker::OpRoleAttrName()));
    VLOG(6) << "Insert sgd to graph.";
    // Add fused scale
    OpDesc Sgd_desc(sgd_ops[0]->Op()->Block());
    Sgd_desc.SetType("sgd");
    Sgd_desc.SetInput(kParam, {fused_vars_name.at(kParam)});
    Sgd_desc.SetInput(kGrad, {fused_vars_name.at(kGrad)});
    Sgd_desc.SetOutput("ParamOut", {fused_vars_name.at(kParam)});

    // TODO(zcd): The LearningRate should be equal.
    Sgd_desc.SetInput(kLearningRate, sgd_ops[0]->Op()->Input(kLearningRate));

    // NOTE: multi_devices_pass requires that every op should have a role.
    Sgd_desc.SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(), op_role);

    return graph->CreateOpNode(&Sgd_desc);
  }
};
}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fuse_sgd_op_pass, paddle::framework::ir::FuseSgdOpPass);
