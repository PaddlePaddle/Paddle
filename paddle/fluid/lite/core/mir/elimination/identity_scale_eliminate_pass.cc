// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/lite/core/mir/pass.h"
#include "paddle/fluid/lite/core/mir/pass_registry.h"
#include "paddle/fluid/lite/core/mir/pattern_matcher_high_api.h"

namespace paddle {
namespace lite {
namespace mir {

namespace {

class Eliminator : public FuseBase {
 public:
  void BuildPattern() override {
    auto* pre_op = OpNode("preop");  // the previous op's output need update
    // TODO(Superjomn) check has only one output
    auto* x = VarNode("x")->assert_is_op_input("scale", "X");
    auto* scale_op = OpNode("scale", "scale")
                         ->assert_op_attr<float>("scale", 1.)
                         ->assert_op_attr<float>("bias", 0.);
    auto* out = VarNode("out")->assert_is_op_output("scale", "Out");

    *pre_op >> *x >> *scale_op >> *out;

    // The pre_op will be eliminated, and a new output-updated op will insert.
    x->AsIntermediate();  // x is pre_op's output, need to update
  }

 private:
  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    auto& pre_op = matched.at("preop")->AsStmt();
    auto op_info = *pre_op.op_info();

    op_info.UpdateAllOutputs(matched.at("x")->AsArg().name,
                             matched.at("out")->AsArg().name);
    pre_op.ResetOp(op_info, graph->valid_places());

    GraphSafeRemoveNodes(graph, {matched.at("scale")});

    IR_NODE_LINK_TO(matched.at("preop"), matched.at("out"));
  }
};

}  // namespace

class IdentityScaleEliminatePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    Eliminator eliminator;
    eliminator(graph.get());
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(identity_scale_eliminate_pass,
                  paddle::lite::mir::IdentityScaleEliminatePass);
