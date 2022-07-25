/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/ir/delete_unsqueeze_pass.h"

#include <string>

#include "glog/logging.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

#include <vector>

#include "paddle/fluid/framework/convert_utils.h"

namespace paddle {
namespace framework {
namespace ir {
class Node;
}  // namespace ir
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {

struct DeleteUnsqueeze : public PatternBase {
  DeleteUnsqueeze(PDPattern *pattern, const std::string &name_scope)
      : PatternBase(pattern, name_scope, "delete_unsqueeze_pass") {}

  PDNode *operator()(PDNode *x);

  // declare operator node's name
  PATTERN_DECL_NODE(unsqz);
  // declare variable node's name
  PATTERN_DECL_NODE(unsqz_in);
  PATTERN_DECL_NODE(unsqz_out);
};

PDNode *DeleteUnsqueeze::operator()(PDNode *x) {
  x->assert_is_op_input("unsqueeze2", "X");

  auto *unsqz = pattern->NewNode(unsqz_repr())->assert_is_op("unsqueeze2");
  auto *unsqz_out = pattern->NewNode(unsqz_out_repr())
                        ->assert_is_op_output("unsqueeze2", "Out");
  unsqz->LinksFrom({x});
  unsqz_out->LinksFrom({unsqz});
  return unsqz_out;
}

}  // namespace patterns

DeleteUnsqueezePass::DeleteUnsqueezePass() {
  AddOpCompat(OpCompat("unsqueeze2"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("AxesTensor")
      .IsOptional()
      .IsTensor()
      .End()
      .AddInput("AxesTensorList")
      .IsOptional()
      .IsTensor()
      .End()
      .AddOutput("XShape")
      .IsOptional()
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("axes")
      .IsType<std::vector<int>>()
      .End();
}

void DeleteUnsqueezePass::ApplyImpl(ir::Graph *graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  FusePassBase::Init("delete_unsqueeze2", graph);
  int found_subgraph_count = 0;
  GraphPatternDetector gpd;

  auto *unsqueeze2_x = gpd.mutable_pattern()
                ->NewNode("unsqueeze2_x")
                ->AsInput()
                ->assert_is_op_input("unsqueeze2")
                ->assert_is_persistable_var();

  patterns::DeleteUnsqueeze fused_pattern(gpd.mutable_pattern(), "delete_unsqueeze2");
  fused_pattern(unsqueeze2_x);

  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *graph) {
    if (subgraph.count(unsqueeze2_x) <= 0) {
      LOG(WARNING) << "The subgraph is empty.";
      return;
    }
    if (!IsCompat(subgraph, graph)) {
      LOG(WARNING) << "Pass in op compat failed.";
      return;
    }
    VLOG(4) << "handle DeleteUnsqueeze fuse";

    GET_IR_NODE_FROM_SUBGRAPH(unsqz_op, unsqz, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(unsqz_out, unsqz_out, fused_pattern);

    auto x_shape = subgraph.at(unsqueeze2_x)->Var()->GetShape();
    auto out_shape = unsqz_out->Var()->GetShape();
    auto* scope = param_scope();
    auto* x_tensor = scope->FindVar(subgraph.at(unsqueeze2_x)->Name())->GetMutable<LoDTensor>();
    auto* x_ptr = x_tensor->data<float>();
    if(0) std::cout << x_ptr << std::endl;
    auto unsqz_out_desc = unsqz_out->Var();
    unsqz_out_desc->SetShape(out_shape);
    unsqz_out_desc->SetPersistable(true);
    auto* unsqz_out_tensor = scope->Var(unsqz_out_desc->Name())->GetMutable<LoDTensor>();
    //auto dtype = framework::TransToPhiDataType(unsqz_out_desc->GetDataType());
    unsqz_out_tensor->Resize(phi::make_ddim(out_shape));
    unsqz_out_tensor->mutable_data<float>(platform::CPUPlace());

    // Remove links in graph
    GraphSafeRemoveNodes(graph, {unsqz_op});
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(delete_unsqueeze_pass,
              paddle::framework::ir::DeleteUnsqueezePass);
