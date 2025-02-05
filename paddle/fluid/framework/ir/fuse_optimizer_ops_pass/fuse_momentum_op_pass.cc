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

#include <string>

#include "glog/logging.h"
#include "paddle/fluid/framework/ir/fuse_optimizer_ops_pass/fuse_optimizer_op_pass.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle::framework::ir {

class Node;

class FuseMomentumOpPass : public FuseOptimizerOpPass {
 private:
  const std::string GetOpType() const override { return "momentum"; }

  const std::vector<std::string> GetAuxiliaryVarNames() const override {
    return {"Velocity"};
  }

  // Fuse Momentum Ops
  ir::Node *FuseOptimizerOps(
      const std::unordered_map<std::string, std::vector<std::string>> &vars_set,
      const std::unordered_map<std::string, std::string> &fused_vars_name,
      const std::vector<ir::Node *> &momentum_ops,
      ir::Graph *graph) const override {
    PADDLE_ENFORCE_GT(
        momentum_ops.size(),
        static_cast<size_t>(0),
        common::errors::InvalidArgument("Momentum ops must not be empty."));

    // Check attributions
    // NOTE: If new attribution is added, the following code maybe need change.
    int op_role =
        PADDLE_GET_CONST(int,
                         momentum_ops[0]->Op()->GetAttr(
                             OpProtoAndCheckerMaker::OpRoleAttrName()));
    float mu = PADDLE_GET_CONST(float, momentum_ops[0]->Op()->GetAttr("mu"));
    bool use_nesterov =
        PADDLE_GET_CONST(bool, momentum_ops[0]->Op()->GetAttr("use_nesterov"));

    for (auto &momentum_op : momentum_ops) {
      PADDLE_ENFORCE_EQ(
          mu,
          PADDLE_GET_CONST(float, momentum_op->Op()->GetAttr("mu")),
          common::errors::InvalidArgument(
              "All momentum Op's attr(mu) must be same, but there are two "
              "different "
              "value: %f, %f.",
              mu,
              PADDLE_GET_CONST(float, momentum_op->Op()->GetAttr("mu"))));
      PADDLE_ENFORCE_EQ(
          use_nesterov,
          PADDLE_GET_CONST(bool, momentum_op->Op()->GetAttr("use_nesterov")),
          common::errors::InvalidArgument(
              "All momentum Op's attr(use_nesterov) must be same, but there "
              "are two different value: %d, %d.",
              use_nesterov,
              PADDLE_GET_CONST(bool,
                               momentum_op->Op()->GetAttr("use_nesterov"))));
      PADDLE_ENFORCE_EQ(
          op_role,
          PADDLE_GET_CONST(int,
                           momentum_op->Op()->GetAttr(
                               OpProtoAndCheckerMaker::OpRoleAttrName())),
          common::errors::InvalidArgument(
              "All momentum Op's attr(op_role) must be same, but there are two "
              "different "
              "value: %d, %d.",
              op_role,
              PADDLE_GET_CONST(int,
                               momentum_op->Op()->GetAttr(
                                   OpProtoAndCheckerMaker::OpRoleAttrName()))));
    }

    // NOTE: fused_var is only exist in scope, so the graph doesn't have
    // fused_var node.

    VLOG(6) << "Insert momentum to graph ";
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

    return graph->CreateOpNode(&momentum_desc);
  }
};

}  // namespace paddle::framework::ir

REGISTER_PASS(fuse_momentum_op_pass, paddle::framework::ir::FuseMomentumOpPass);
