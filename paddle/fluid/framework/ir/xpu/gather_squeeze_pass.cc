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

#include "paddle/fluid/framework/ir/xpu/gather_squeeze_pass.h"
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

struct GatherSqueeze : public PatternBase {
  GatherSqueeze(PDPattern* pattern, const std::string& name_scope);

  // declare operator node's name
  PATTERN_DECL_NODE(gather);
  PATTERN_DECL_NODE(squeeze2);
  // declare variable node's name
  PATTERN_DECL_NODE(gather_in);
  PATTERN_DECL_NODE(gather_index);
  PATTERN_DECL_NODE(gather_out);
};  // struct GatherSqueeze

GatherSqueeze::GatherSqueeze(PDPattern* pattern, const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  auto* gather_in = pattern->NewNode(gather_in_repr())
                        ->assert_is_op_input("gather", "X")
                        ->assert_more([](Node* node) {
                          for (auto* op : node->outputs) {
                            if (op->Op()->Type() != "gather") {
                              return false;
                            }
                          }
                          return node->outputs.size() >= 2 &&
                                 node->Var()->GetShape().size() >= 2 &&
                                 node->Var()->GetShape().size() < 5;
                        });
  auto* gather_index = pattern->NewNode(gather_index_repr())
                           ->assert_is_op_input("gather", "Index")
                           ->assert_more([](Node* node) {
                             auto shape = node->Var()->GetShape();
                             return shape.size() == 1 && shape[0] == 1;
                           });
  auto* gather = pattern->NewNode(gather_repr())->assert_is_op("gather");
  auto* gather_out = pattern->NewNode(gather_out_repr())
                         ->assert_is_op_output("gather", "Out")
                         ->assert_is_op_input("squeeze2", "X");
  auto* squeeze2 = pattern->NewNode(squeeze2_repr())->assert_is_op("squeeze2");

  gather->LinksFrom({gather_in, gather_index}).LinksTo({gather_out});
  gather_out->LinksTo({squeeze2});
}

}  // namespace patterns

void GatherSqueezePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);

  AddTranspose(graph);
}

void GatherSqueezePass::AddTranspose(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  GraphPatternDetector gpd;
  patterns::GatherSqueeze pattern(gpd.mutable_pattern(), name_scope_);

  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle GatherSqueezePass";
    GET_IR_NODE(gather);
    GET_IR_NODE(gather_in);
    GET_IR_NODE(gather_out);
    GET_IR_NODE(squeeze2);

    bool flag = true;
    auto var_dims = static_cast<int32_t>(gather_in->Var()->GetShape().size());
    auto gather_axis = gather->Op()->GetAttrIfExists<int>("axis");
    auto squeeze_axis =
        squeeze2->Op()->GetAttrIfExists<std::vector<int>>("axes");

    flag = flag && gather_axis == var_dims - 1;
    flag = flag && (squeeze_axis == std::vector<int>{-1} ||
                    squeeze_axis == std::vector<int>{var_dims - 1});

    if (flag) {
      gather->Op()->SetAttr("axis", 0);
      squeeze2->Op()->SetAttr("axes", std::vector<int>{0});

      std::string transpose2_out_name =
          patterns::PDNodeName(name_scope_, "transpose2out");
      VarDesc transpose2_out_vardesc(transpose2_out_name);
      OpDesc transpose2_op_desc(gather->Op()->Block());
      auto gather_in_shape = gather_in->Var()->GetShape();
      auto gather_out_shape = gather_out->Var()->GetShape();
      transpose2_out_vardesc.SetDataType(gather_in->Var()->GetDataType());

      if (var_dims == 2) {
        gather_out->Var()->SetShape({gather_out_shape[1], gather_out_shape[0]});
        transpose2_out_vardesc.SetShape(
            {gather_in_shape[1], gather_in_shape[0]});
        transpose2_op_desc.SetAttr("axis", std::vector<int>{1, 0});
      } else if (var_dims == 3) {
        gather_out->Var()->SetShape(
            {gather_out_shape[2], gather_out_shape[0], gather_out_shape[1]});
        transpose2_out_vardesc.SetShape(
            {gather_in_shape[2], gather_in_shape[0], gather_in_shape[1]});
        transpose2_op_desc.SetAttr("axis", std::vector<int>{2, 0, 1});
      } else {
        gather_out->Var()->SetShape({gather_out_shape[3],
                                     gather_out_shape[0],
                                     gather_out_shape[1],
                                     gather_out_shape[2]});
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
    }
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(gather_squeeze_pass, paddle::framework::ir::GatherSqueezePass);

REGISTER_PASS_CAPABILITY(gather_squeeze_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .EQ("gather", 1)
            .EQ("squeeze2", 0));
