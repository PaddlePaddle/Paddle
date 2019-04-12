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
#include "paddle/fluid/framework/details/build_strategy.h"
#include "paddle/fluid/framework/details/fuse_optimizer_op_pass.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace framework {
namespace details {

class FuseMomentumOpPass : public FuseOptimizerOpPass {
 private:
  virtual const std::string GetOpType() const { return "momentum"; }

  virtual const std::vector<std::string> GetAuxiliaryVarNames() const {
    return {"Velocity"};
  }

  // Fuse Momentum Ops
  virtual void FuseOptimizerOps(
      const std::unordered_map<std::string, std::vector<std::string>> &vars_set,
      const std::unordered_map<std::string, std::string> &fused_vars_name,
      const std::vector<ir::Node *> &momentum_ops, ir::Graph *graph) const {
    PADDLE_ENFORCE_GT(momentum_ops.size(), static_cast<size_t>(0));

    // Check attributions
    // NOTE: If new attribution is added, the following code maybe need change.
    int op_role = boost::get<int>(momentum_ops[0]->Op()->GetAttr(
        OpProtoAndCheckerMaker::OpRoleAttrName()));
    float mu = boost::get<float>(momentum_ops[0]->Op()->GetAttr("mu"));
    bool use_nesterov =
        boost::get<bool>(momentum_ops[0]->Op()->GetAttr("use_nesterov"));

    for (auto &momentum_op : momentum_ops) {
      PADDLE_ENFORCE_EQ(mu,
                        boost::get<float>(momentum_op->Op()->GetAttr("mu")));
      PADDLE_ENFORCE_EQ(
          use_nesterov,
          boost::get<bool>(momentum_op->Op()->GetAttr("use_nesterov")));
      PADDLE_ENFORCE_EQ(op_role,
                        boost::get<int>(momentum_op->Op()->GetAttr(
                            OpProtoAndCheckerMaker::OpRoleAttrName())));
    }

    // NOTE: fused_var is only exist in scope, so the graph doesn't have
    // fused_var node.

    VLOG(10) << "Insert momentum to graph ";
    OpDesc momentum_desc(momentum_ops[0]->Op()->Block());
    momentum_desc.SetType("momentum");
    momentum_desc.SetInput(kParam, {fused_vars_name.at(kParam)});
    momentum_desc.SetInput(kGrad, {fused_vars_name.at(kGrad)});
    momentum_desc.SetInput("Velocity", {fused_vars_name.at("Velocity")});
    // TODO(zcd): The LearningRate should be equal.
    momentum_desc.SetInput(kLearningRate,
                           momentum_ops[0]->Op()->Input(kLearningRate));

    momentum_desc.SetOutput("ParamOut", {fused_vars_name.at(kParam)});
    momentum_desc.SetOutput("VelocityOut", {fused_vars_name.at("Velocity")});
    momentum_desc.SetAttr("mu", mu);
    momentum_desc.SetAttr("use_nesterov", use_nesterov);
    momentum_desc.SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(), op_role);

    auto momentum_node = graph->CreateOpNode(&momentum_desc);

    InserInputAndOutputForOptOps(momentum_ops, momentum_node);
  }
};

}  // namespace details
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fuse_momentum_op_pass,
              paddle::framework::details::FuseMomentumOpPass)
    .RequirePassAttr(paddle::framework::details::kPlaces)
    .RequirePassAttr(paddle::framework::details::kLocalScopes);
