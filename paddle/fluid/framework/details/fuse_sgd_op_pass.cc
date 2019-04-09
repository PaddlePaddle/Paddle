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
#include <utility>
#include <vector>

#include "paddle/fluid/framework/details/build_strategy.h"
#include "paddle/fluid/framework/details/fuse_optimizer_op_pass.h"
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace framework {
namespace details {

class FuseSgdOpPass : public FuseOptimizerOpPass {
 private:
  virtual const std::string GetOpType() const { return "sgd"; }

  virtual const std::vector<std::string> GetAuxiliaryVarNames() const {
    return {"Param"};
  }

  // Fuse Sgd Ops
  virtual void FuseOptimizerOps(
      const std::unordered_map<std::string, std::vector<std::string>> &vars_set,
      const std::unordered_map<std::string, std::string> &fused_vars_name,
      const std::vector<ir::Node *> &sgd_ops, ir::Graph *graph) const {
    PADDLE_ENFORCE_GT(sgd_ops.size(), static_cast<size_t>(0));

    // NOTE: fused_var is only exist in scope, so the graph doesn't have
    // fused_var
    // node.

    int op_role = boost::get<int>(
        sgd_ops[0]->Op()->GetAttr(OpProtoAndCheckerMaker::OpRoleAttrName()));
    VLOG(10) << "Insert sgd to graph ";
    // Add fused scale
    OpDesc Sgd_desc(sgd_ops[0]->Op()->Block());
    Sgd_desc.SetType("sgd");
    Sgd_desc.SetInput("Param", {fused_vars_name.at("Param")});
    Sgd_desc.SetInput("Grad", {fused_vars_name.at("Grad")});
    Sgd_desc.SetOutput("ParamOut", {fused_vars_name.at("Param")});

    // TODO(zcd): The LearningRate, Beta1Pow, Beta2Pow should be equal.
    Sgd_desc.SetInput("LearningRate", sgd_ops[0]->Op()->Input("LearningRate"));

    // NOTE: multi_devices_pass requires that every op should have a role.
    Sgd_desc.SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(), op_role);

    auto sgd_node = graph->CreateOpNode(&Sgd_desc);

    InserInputAndOutputForOptOps(sgd_ops, sgd_node);
  }
};

}  // namespace details
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fuse_sgd_op_pass, paddle::framework::details::FuseSgdOpPass)
    .RequirePassAttr(paddle::framework::details::kPlaces)
    .RequirePassAttr(paddle::framework::details::kLocalScopes);
