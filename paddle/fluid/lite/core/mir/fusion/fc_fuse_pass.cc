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

#include "paddle/fluid/lite/core/mir/fusion/fc_fuse_pass.h"
#include <memory>
#include <vector>
#include "paddle/fluid/lite/core/mir/pass_registry.h"
#include "paddle/fluid/lite/core/mir/pattern_matcher_high_api.h"

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

class FcFusePass::Impl : public FuseBase {
 public:
  void BuildPattern() override;
  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override;

 private:
  cpp::OpDesc GenOpDesc(const key2nodes_t& matched) override;
};

void FcFusePass::Impl::BuildPattern() {
  // create nodes.
  auto* x = VarNode("x");
  auto* W = VarNode("W");
  auto* b = VarNode("b");
  auto* mul = OpNode("mul", "mul");
  auto* mul_out = VarNode("mul_out");
  auto* add = OpNode("add", "elementwise_add");
  auto* Out = VarNode("Out");

  // create topology.
  std::vector<PMNode*> mul_inputs{W, x};
  std::vector<PMNode*> add_inputs{mul_out, b};
  mul_inputs >> *mul >> *mul_out;
  add_inputs >> *add >> *Out;

  // Some op specialities.
  mul_out->AsIntermediate();
  mul->AsIntermediate();
  add->AsIntermediate();
}

void FcFusePass::Impl::InsertNewNode(SSAGraph* graph,
                                     const key2nodes_t& matched) {
  auto op_desc = GenOpDesc(matched);
  auto fc_op = LiteOpRegistry::Global().Create("fc");
  auto mul = matched.at("mul")->stmt()->op;
  auto* scope = mul->scope();
  auto& valid_places = mul->valid_places();
  fc_op->Attach(op_desc, scope);

  auto* new_op_node = graph->GraphCreateInstructNode(fc_op, valid_places);

  IR_NODE_LINK_TO(matched.at("W"), new_op_node);
  IR_NODE_LINK_TO(matched.at("x"), new_op_node);
  IR_NODE_LINK_TO(matched.at("b"), new_op_node);
  IR_NODE_LINK_TO(new_op_node, matched.at("Out"));
}

cpp::OpDesc FcFusePass::Impl::GenOpDesc(const key2nodes_t& matched) {
  cpp::OpDesc op_desc;
  op_desc.SetType("fc");
  op_desc.SetInput("Input", {matched.at("x")->arg()->name});
  op_desc.SetInput("W", {matched.at("W")->arg()->name});
  op_desc.SetInput("Bias", {matched.at("b")->arg()->name});
  op_desc.SetOutput("Out", {matched.at("Out")->arg()->name});
  op_desc.SetAttr("in_num_col_dims", 1);
  return op_desc;
}

FcFusePass::FcFusePass() : impl_(new FcFusePass::Impl) {}

void FcFusePass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  impl_->operator()(graph.get());
}

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(lite_fc_fuse_pass, paddle::lite::mir::fusion::FcFusePass);
