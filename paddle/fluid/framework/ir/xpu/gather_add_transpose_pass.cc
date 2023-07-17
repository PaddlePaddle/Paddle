// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/ir/xpu/pass_utils.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {

/*
add transpose when gather with axis = -1 + squeeze2
For example:
graph:
                      x
                      |
                    gather (axis = -1)
                      |
                   squeeze2
                      |
                    output
------------------------------------------------------
After the pass is applied:
                      x
                      |
                  transpose2
                      |
                    gather (axis = 0)
                      |
                   squeeze2
                      |
                    Output
*/
struct GatherAddTranspose : public PatternBase {
  GatherAddTranspose(PDPattern* pattern, const std::string& name_scope);

  // declare operator node's name
  PATTERN_DECL_NODE(gather);
  PATTERN_DECL_NODE(squeeze2);
  // declare variable node's name
  PATTERN_DECL_NODE(gather_in);
  PATTERN_DECL_NODE(gather_out);
};  // struct GatherAddTranspose

GatherAddTranspose::GatherAddTranspose(PDPattern* pattern,
                                       const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  auto* gather_in = pattern->NewNode(gather_in_repr())
                        ->assert_is_op_input("gather", "X")
                        ->assert_more([](Node* node) {
                          return node->Var()->GetShape().size() >= 3 &&
                                 node->Var()->GetShape().size() < 5;
                        });
  auto* gather =
      pattern->NewNode(gather_repr())
          ->assert_is_op("gather")
          ->assert_more([&](Node* node) {
            auto* op_desc = node->Op();
            auto in_var_node = node->inputs[1];
            auto var_size = in_var_node->Var()->GetShape().size();
            VLOG(3) << "GatherAddTranspose gather var_size" << var_size;
            return op_desc->GetAttrIfExists<int>("axis") == -1 ||
                   op_desc->GetAttrIfExists<int>("axis") == var_size - 1;
          });
  auto* gather_out =
      pattern->NewNode(gather_out_repr())->assert_is_op_output("gather", "Out");
  auto* squeeze2 =
      pattern->NewNode(squeeze2_repr())
          ->assert_is_op("squeeze2")
          ->assert_more([&](Node* node) {
            auto* op_desc = node->Op();
            auto in_var_node = node->inputs[0];
            auto var_size = in_var_node->Var()->GetShape().size();
            VLOG(3) << "GatherAddTranspose squeeze2 var_size" << var_size;
            return op_desc->GetAttrIfExists<std::vector<int>>("axes") ==
                       std::vector<int>{-1} ||
                   op_desc->GetAttrIfExists<std::vector<int>>("axes") ==
                       std::vector<int>{var_size - 1};
          });

  gather->LinksFrom({gather_in}).LinksTo({gather_out});
  gather_out->LinksTo({squeeze2});
}

}  // namespace patterns

class GatherAddTransposePass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  void GatherAddTranspose(ir::Graph* graph) const;
  const std::string name_scope_{"gather_add_transpose_pass"};
};

void GatherAddTransposePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);

  GatherAddTranspose(graph);
}

void GatherAddTransposePass::GatherAddTranspose(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  GraphPatternDetector gpd;
  patterns::GatherAddTranspose pattern(gpd.mutable_pattern(), name_scope_);

  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle GatherAddTransposePass";
    GET_IR_NODE(gather);
    GET_IR_NODE(gather_in);
    GET_IR_NODE(gather_out);
    GET_IR_NODE(squeeze2);

    gather->Op()->SetAttr("axis", 0);
    squeeze2->Op()->SetAttr("axes", std::vector<int>{0});
    auto gather_out_shape = gather_out->Var()->GetShape();
    if (gather_out_shape.size() == 3) {
      gather_out->Var()->SetShape(
          {gather_out_shape[2], gather_out_shape[0], gather_out_shape[1]});
    } else {
      gather_out->Var()->SetShape({gather_out_shape[3],
                                   gather_out_shape[0],
                                   gather_out_shape[1],
                                   gather_out_shape[2]});
    }

    std::string transpose2_out_name =
        patterns::PDNodeName("gather_add_transpose", "transpose2out");
    VarDesc transpose2_out_vardesc(transpose2_out_name);
    OpDesc transpose2_op_desc(gather->Op()->Block());
    auto gather_in_shape = gather_in->Var()->GetShape();
    transpose2_out_vardesc.SetDataType(gather_in->Var()->GetDataType());
    if (gather_in_shape.size() == 3) {
      transpose2_out_vardesc.SetShape(
          {gather_in_shape[2], gather_in_shape[0], gather_in_shape[1]});
      transpose2_op_desc.SetAttr("axis", std::vector<int>{2, 0, 1});
    } else {
      transpose2_out_vardesc.SetShape({gather_in_shape[3],
                                       gather_in_shape[0],
                                       gather_in_shape[1],
                                       gather_in_shape[2]});
      transpose2_op_desc.SetAttr("axis", std::vector<int>{3, 0, 1, 2});
    }

    auto* transpose2_out = graph->CreateVarNode(&transpose2_out_vardesc);

    transpose2_op_desc.SetType("transpose2");
    transpose2_op_desc.SetInput("X", {gather_in->Name()});
    transpose2_op_desc.SetOutput("Out", {transpose2_out->Name()});
    auto* transpose2 = graph->CreateOpNode(&transpose2_op_desc);

    gather->Op()->SetInput("X", {transpose2_out->Name()});

    IR_NODE_UNLINK(gather_in, gather);
    IR_NODE_LINK_TO(gather_in, transpose2);
    IR_NODE_LINK_TO(transpose2, transpose2_out);
    IR_NODE_LINK_TO(transpose2_out, gather);

    found_subgraph_count++;
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(gather_add_transpose_pass,
              paddle::framework::ir::GatherAddTransposePass);
