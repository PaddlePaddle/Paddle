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

#include "glog/logging.h"

#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/ir/xpu/pass_utils.h"
#include "paddle/fluid/framework/ir/xpu/quant_utils.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace phi {
class DenseTensor;
}  // namespace phi

namespace paddle {
namespace framework {
class Scope;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {

/*
fuse elementwise_mul + elementwise_add op to addcmul_xpu op
For example:
graph:
              x         y
               \       /
                \     /
            elementwise_mul    w
                    \         /
                     \       /
                  elementwise_add
                        |
                        |
                      output
------------------------------------------------------
After the pass is applied:
               x      y      w
                \     |     /
                 \    |    /
                 addcmul_xpu
                      |
                      |
                    output
*/
struct ElementwiseMulAddFusePass : public PatternBase {
  ElementwiseMulAddFusePass(PDPattern* pattern, const std::string& name_scope);
  // declare operator node's name
  PATTERN_DECL_NODE(elementwise_mul);
  PATTERN_DECL_NODE(elementwise_add);
  // declare variable node's name
  PATTERN_DECL_NODE(mul_x);
  PATTERN_DECL_NODE(mul_y);
  PATTERN_DECL_NODE(mul_out);
  PATTERN_DECL_NODE(add_w);
  PATTERN_DECL_NODE(add_out);
};

ElementwiseMulAddFusePass::ElementwiseMulAddFusePass(
    PDPattern* pattern, const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  auto elementwise_mul =
      pattern->NewNode(elementwise_mul_repr())->assert_is_op("elementwise_mul");
  auto elementwise_add =
      pattern->NewNode(elementwise_add_repr())->assert_is_op("elementwise_add");
  auto mul_x = pattern->NewNode(mul_x_repr())
                   ->AsInput()
                   ->assert_is_op_input("elementwise_mul", "X");
  auto mul_y = pattern->NewNode(mul_y_repr())
                   ->AsInput()
                   ->assert_is_op_input("elementwise_mul", "Y");
  auto mul_out = pattern->NewNode(mul_out_repr())
                     ->AsOutput()
                     ->assert_is_op_output("elementwise_mul", "Out")
                     ->assert_is_op_input("elementwise_add", "X")
                     ->assert_has_n_outputs(1);
  elementwise_mul->LinksFrom({mul_x, mul_y}).LinksTo({mul_out});
  auto add_w = pattern->NewNode(add_w_repr())
                   ->AsInput()
                   ->assert_is_op_input("elementwise_add", "Y");
  auto add_out = pattern->NewNode(add_out_repr())
                     ->AsOutput()
                     ->assert_is_op_output("elementwise_add", "Out");
  elementwise_add->LinksFrom({mul_out, add_w}).LinksTo({add_out});
}

/*
special case for addcmul_xpu op:
graph:
              x         y
               \       /
                \     /
            elementwise_mul    x
                    \         /
                     \       /
                  elementwise_add
                        |
                        |
                      output
------------------------------------------------------
After the pass is applied:
               x             y
                \           /
                 \         /
                 addcmul_xpu
                      |
                      |
                    output
*/
struct ElementwiseMulAddFuseXYPattern : public PatternBase {
  ElementwiseMulAddFuseXYPattern(PDPattern* pattern,
                                 const std::string& name_scope);
  // declare operator node's name
  PATTERN_DECL_NODE(elementwise_mul);
  PATTERN_DECL_NODE(elementwise_add);
  // declare variable node's name
  PATTERN_DECL_NODE(mul_x);
  PATTERN_DECL_NODE(mul_y);
  PATTERN_DECL_NODE(mul_out);
  PATTERN_DECL_NODE(add_out);
};

ElementwiseMulAddFuseXYPattern::ElementwiseMulAddFuseXYPattern(
    PDPattern* pattern, const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  auto elementwise_mul =
      pattern->NewNode(elementwise_mul_repr())->assert_is_op("elementwise_mul");
  auto elementwise_add =
      pattern->NewNode(elementwise_add_repr())->assert_is_op("elementwise_add");
  auto mul_x = pattern->NewNode(mul_x_repr())
                   ->AsInput()
                   ->assert_is_op_input("elementwise_mul", "X")
                   ->assert_is_op_input("elementwise_add", "Y");
  auto mul_y = pattern->NewNode(mul_y_repr())
                   ->AsInput()
                   ->assert_is_op_input("elementwise_mul", "Y");
  auto mul_out = pattern->NewNode(mul_out_repr())
                     ->AsOutput()
                     ->assert_is_op_output("elementwise_mul", "Out")
                     ->assert_is_op_input("elementwise_add", "X");
  elementwise_mul->LinksFrom({mul_x, mul_y}).LinksTo({mul_out});
  auto add_out = pattern->NewNode(add_out_repr())
                     ->AsOutput()
                     ->assert_is_op_output("elementwise_add", "Out");
  elementwise_add->LinksFrom({mul_out, mul_x}).LinksTo({add_out});
}
}  // namespace patterns

class ElementwiseMulAddFusePass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  void FuseElementwiseMulAdd(ir::Graph* graph) const;
  void FuseElementwiseMulAddWithOnlyXY(ir::Graph* graph) const;

  const std::string name_scope_{"elementwise_mul_add_fuse_pass"};
};

void ElementwiseMulAddFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);

  FuseElementwiseMulAdd(graph);
  FuseElementwiseMulAddWithOnlyXY(graph);
}

void ElementwiseMulAddFusePass::FuseElementwiseMulAdd(ir::Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::ElementwiseMulAddFusePass pattern(gpd.mutable_pattern(),
                                              name_scope_);

  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle ElementwiseMulAddFusePass";
    // declare operator node's name
    GET_IR_NODE(elementwise_mul);
    GET_IR_NODE(elementwise_add);
    // declare variable node's name
    GET_IR_NODE(mul_x);
    GET_IR_NODE(mul_y);
    GET_IR_NODE(mul_out);
    GET_IR_NODE(add_w);
    GET_IR_NODE(add_out);

    bool flag = true;
    auto var_type = mul_x->Var()->GetDataType();
    if (var_type != proto::VarType::FP16 && var_type != proto::VarType::FP32) {
      flag = false;
    }

    auto x_shape = mul_x->Var()->GetShape();
    auto y_shape = mul_y->Var()->GetShape();
    auto w_shape = add_w->Var()->GetShape();
    if (x_shape.size() == y_shape.size() && x_shape.size() == w_shape.size()) {
      for (size_t i = 0; i < x_shape.size(); ++i) {
        if (x_shape[i] != y_shape[i] || x_shape[i] != w_shape[i] ||
            x_shape[i] == -1) {
          flag = false;
        }
      }
    } else {
      flag = false;
    }

    if (flag) {
      auto* block = elementwise_mul->Op()->Block();

      // delete useless node
      std::unordered_set<const Node*> delete_nodes;

      // Generate addcmul_xpu op
      framework::OpDesc fused_op_desc(block);
      fused_op_desc.SetType("addcmul_xpu");
      fused_op_desc.SetInput("x", {mul_x->Name()});
      fused_op_desc.SetInput("y", {mul_y->Name()});
      fused_op_desc.SetInput("w", {add_w->Name()});
      fused_op_desc.SetOutput("out", {add_out->Name()});
      auto* fused_op = graph->CreateOpNode(&fused_op_desc);
      IR_NODE_LINK_TO(mul_x, fused_op);
      IR_NODE_LINK_TO(mul_y, fused_op);
      IR_NODE_LINK_TO(add_w, fused_op);
      IR_NODE_LINK_TO(fused_op, add_out);
      delete_nodes.insert({elementwise_mul, elementwise_add, mul_out});
      GraphSafeRemoveNodes(graph, delete_nodes);
      found_subgraph_count++;
    }
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

void ElementwiseMulAddFusePass::FuseElementwiseMulAddWithOnlyXY(
    ir::Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::ElementwiseMulAddFuseXYPattern pattern(gpd.mutable_pattern(),
                                                   name_scope_);

  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle ElementwiseMulAddFusePass";
    // declare operator node's name
    GET_IR_NODE(elementwise_mul);
    GET_IR_NODE(elementwise_add);
    // declare variable node's name
    GET_IR_NODE(mul_x);
    GET_IR_NODE(mul_y);
    GET_IR_NODE(mul_out);
    GET_IR_NODE(add_out);

    bool flag = true;
    auto var_type = mul_x->Var()->GetDataType();
    if (var_type != proto::VarType::FP16 && var_type != proto::VarType::FP32) {
      flag = false;
    }

    auto x_shape = mul_x->Var()->GetShape();
    auto y_shape = mul_y->Var()->GetShape();
    if (x_shape.size() == y_shape.size()) {
      for (size_t i = 0; i < x_shape.size(); ++i) {
        if (x_shape[i] != y_shape[i] || x_shape[i] == -1) {
          flag = false;
        }
      }
    } else {
      flag = false;
    }

    if (flag) {
      auto* block = elementwise_mul->Op()->Block();

      // delete useless node
      std::unordered_set<const Node*> delete_nodes;

      // Generate addcmul_xpu op
      framework::OpDesc fused_op_desc(block);
      fused_op_desc.SetType("addcmul_xpu");
      fused_op_desc.SetInput("x", {mul_x->Name()});
      fused_op_desc.SetInput("y", {mul_y->Name()});
      fused_op_desc.SetInput("w", {mul_x->Name()});
      fused_op_desc.SetOutput("out", {add_out->Name()});
      auto* fused_op = graph->CreateOpNode(&fused_op_desc);
      IR_NODE_LINK_TO(mul_x, fused_op);
      IR_NODE_LINK_TO(mul_y, fused_op);
      IR_NODE_LINK_TO(fused_op, add_out);
      delete_nodes.insert({elementwise_mul, elementwise_add, mul_out});
      GraphSafeRemoveNodes(graph, delete_nodes);
      found_subgraph_count++;
    }
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(elementwise_mul_add_fuse_pass,
              paddle::framework::ir::ElementwiseMulAddFusePass);

REGISTER_PASS_CAPABILITY(elementwise_mul_add_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .GE("elementwise_add", 0)
            .LE("elementwise_add", 1)
            .GE("elementwise_mul", 0)
            .LE("elementwise_mul", 1));
