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

#include "paddle/fluid/framework/ir/xpu/matmul_weight_trans_pass.h"
#include <string>

#include "glog/logging.h"

#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/xpu/pass_utils.h"
#include "paddle/fluid/framework/ir/xpu/quant_utils.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

namespace patterns {
struct Reshape2MatmulV2Pattern : public PatternBase {
  Reshape2MatmulV2Pattern(PDPattern* pattern, const std::string& name_scope);

  // declare operator node's name
  PATTERN_DECL_NODE(reshape2);
  PATTERN_DECL_NODE(matmul_v2);
  // declare variable node's name
  PATTERN_DECL_NODE(reshape2_in);
  PATTERN_DECL_NODE(matmul_x);
  PATTERN_DECL_NODE(matmul_y);
  PATTERN_DECL_NODE(matmul_out);
};

Reshape2MatmulV2Pattern::Reshape2MatmulV2Pattern(PDPattern* pattern,
                                                 const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  auto* reshape2_in =
      pattern->NewNode(reshape2_in_repr())
          ->assert_is_op_input("reshape2", "X")
          ->AsInput()
          ->assert_more([](Node* node) {
            auto reshape2_in_x_shape = node->Var()->GetShape();
            size_t reshape2_in_rank = reshape2_in_x_shape.size();
            return (reshape2_in_rank == 4 && reshape2_in_x_shape[2] == 1 &&
                    reshape2_in_x_shape[3] == 1);
          });
  auto* reshape2 = pattern->NewNode(reshape2_repr())->assert_is_op("reshape2");
  auto matmul_x = pattern->NewNode(matmul_x_repr())
                      ->assert_is_op_output("reshape2", "Out")
                      ->assert_is_op_input("matmul_v2", "X")
                      ->assert_more([](Node* node) {
                        auto matmul_x_shape = node->Var()->GetShape();
                        size_t matmul_x_rank = matmul_x_shape.size();
                        return matmul_x_rank == 2;
                      });
  auto* matmul_y = pattern->NewNode(matmul_y_repr())
                       ->assert_is_op_input("matmul_v2", "Y")
                       ->assert_is_persistable_var()
                       ->assert_more([](Node* node) {
                         auto matmul_y_shape = node->Var()->GetShape();
                         size_t matmul_y_rank = matmul_y_shape.size();
                         return matmul_y_rank == 2;
                       });
  auto* matmul_v2 = pattern->NewNode(matmul_v2_repr())
                        ->assert_is_op("matmul_v2")
                        ->assert_op_attr<bool>("trans_x", false)
                        ->assert_op_attr<bool>("trans_y", true);
  auto* matmul_out = pattern->NewNode(matmul_out_repr())
                         ->assert_is_op_output("matmul_v2", "Out")
                         ->AsOutput();
  reshape2->LinksFrom({reshape2_in}).LinksTo({matmul_x});
  matmul_v2->LinksFrom({matmul_x, matmul_y}).LinksTo({matmul_out});
}

struct Transpose2MatmulV2Pattern : public PatternBase {
  Transpose2MatmulV2Pattern(PDPattern* pattern, const std::string& name_scope);

  // declare operator node's name
  PATTERN_DECL_NODE(transpose2);
  PATTERN_DECL_NODE(matmul);
  // declare variable node's name
  PATTERN_DECL_NODE(transpose2_in);
  PATTERN_DECL_NODE(matmul_x);
  PATTERN_DECL_NODE(matmul_y);
  PATTERN_DECL_NODE(matmul_out);
};

Transpose2MatmulV2Pattern::Transpose2MatmulV2Pattern(
    PDPattern* pattern, const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  auto* transpose2_in = pattern->NewNode(transpose2_in_repr())
                            ->assert_is_op_input("transpose2", "X")
                            ->AsInput();
  auto* transpose2 =
      pattern->NewNode(transpose2_repr())
          ->assert_is_op("transpose2")
          ->assert_more([](Node* node) {
            auto axis = node->Op()->GetAttrIfExists<std::vector<int>>("axis");
            return axis.size() == 3 && axis[0] == 0 && axis[1] == 2 &&
                   axis[2] == 1;  // axis == [0, 2, 1]
          });
  auto matmul_x =
      pattern->NewNode(matmul_x_repr())->assert_is_op_input("matmul_v2", "X");
  auto* matmul_y = pattern->NewNode(matmul_y_repr())
                       ->assert_is_op_input("matmul_v2", "Y")
                       ->assert_is_op_output("transpose2", "Out");
  auto* matmul = pattern->NewNode(matmul_repr())
                     ->assert_is_op("matmul_v2")
                     ->assert_op_attr<bool>("trans_y", false);
  auto* matmul_out = pattern->NewNode(matmul_out_repr())
                         ->assert_is_op_output("matmul_v2", "Out")
                         ->AsOutput();
  transpose2->LinksFrom({transpose2_in}).LinksTo({matmul_y});
  matmul->LinksFrom({matmul_x, matmul_y}).LinksTo({matmul_out});
}

}  // namespace patterns

void MatmulWeightTransPass::TransMatmulV2Weight(ir::Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::Reshape2MatmulV2Pattern pattern(gpd.mutable_pattern(), name_scope_);
  int found_subgraph_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle TransMatmulV2Weight";
    /* declare operator node's name */
    GET_IR_NODE(reshape2);
    GET_IR_NODE(matmul_v2);
    /* declare variable node's name*/
    GET_IR_NODE(reshape2_in);
    GET_IR_NODE(matmul_x);
    GET_IR_NODE(matmul_y);
    GET_IR_NODE(matmul_out);

    auto* scope = param_scope();
    PADDLE_ENFORCE_NOT_NULL(
        scope, common::errors::InvalidArgument("Scope cannot be nullptr."));

    auto* matmul_y_t =
        scope->GetVar(matmul_y->Name())->GetMutable<phi::DenseTensor>();
    Transpose2D(matmul_y_t);
    auto from_shape = matmul_y->Var()->GetShape();
    matmul_y->Var()->SetShape({from_shape[1], from_shape[0]});
    matmul_v2->Op()->SetAttr("trans_y", false);
    matmul_v2->Op()->Flush();
    // delete useless node
    found_subgraph_count++;
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

void MatmulWeightTransPass::FuseTranspose2MatmulV2(ir::Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::Transpose2MatmulV2Pattern pattern(gpd.mutable_pattern(),
                                              name_scope_);
  int found_subgraph_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle FuseTranspose2Matmul";
    /* declare operator node's name */
    GET_IR_NODE(transpose2);
    GET_IR_NODE(matmul);
    /* declare variable node's name*/
    GET_IR_NODE(transpose2_in);
    GET_IR_NODE(matmul_x);
    GET_IR_NODE(matmul_y);
    GET_IR_NODE(matmul_out);

    auto* scope = param_scope();
    PADDLE_ENFORCE_NOT_NULL(
        scope, common::errors::InvalidArgument("Scope cannot be nullptr."));

    matmul->Op()->RenameInput(matmul_y->Name(), transpose2_in->Name());
    matmul->Op()->SetAttr("trans_y", true);
    matmul->Op()->Flush();

    IR_NODE_LINK_TO(transpose2_in, matmul);
    // delete useless node
    std::unordered_set<const Node*> delete_nodes = {transpose2, matmul_y};
    GraphSafeRemoveNodes(graph, delete_nodes);
    // delete useless node
    found_subgraph_count++;
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

void MatmulWeightTransPass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);

  TransMatmulV2Weight(graph);
  FuseTranspose2MatmulV2(graph);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(matmul_weight_trans_pass,
              paddle::framework::ir::MatmulWeightTransPass);

REGISTER_PASS_CAPABILITY(matmul_weight_trans_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .EQ("reshape2", 0)
            .EQ("matmul_v2", 0));
