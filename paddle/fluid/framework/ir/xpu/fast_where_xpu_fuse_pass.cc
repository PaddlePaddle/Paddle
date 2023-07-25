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
fuse cast+scale+mul+mul+add ops to fast_where_xpu op reduce memory access.
For example:
            condition
               |
              cast
               |
            /      \
           /        \
        scale        \
  x      /      y     \
    \   /        \    /
    mul0           mul1
        \          /
         \        /
          \      /
            add

After the pass is applied,

condition    x       y
     \       |      /
      \      |     /
       \     |    /
      fast_where_xpu

or,

          condition
              |
            cast
              |
            /    \
           /    scale
          /         \
   x     /      y    \
    \   /        \    /
      mul0         mul1
        \          /
         \        /
          \      /
            add

After the pass is applied,

condition    y       x
     \       |      /
      \      |     /
       \     |    /
      fast_where_xpu

or,

          condition
             |
            cast
             |
          /    \
      scale     \
       /         \
      /    x      \      y
      \   /        \    /
       mul0         mul1
          \          /
           \        /
            \      /
              add

After the pass is applied,

condition    x       y
     \       |      /
      \      |     /
       \     |    /
      fast_where_xpu

*/
struct OneFastWhereXPUPattern : public PatternBase {
  OneFastWhereXPUPattern(PDPattern* pattern, const std::string& name_scope);
  // declare operator node's name
  PATTERN_DECL_NODE(cast);
  PATTERN_DECL_NODE(scale);
  PATTERN_DECL_NODE(mul0);
  PATTERN_DECL_NODE(mul1);
  PATTERN_DECL_NODE(add);
  // declare variable node's name
  // cast
  PATTERN_DECL_NODE(cast_in);
  PATTERN_DECL_NODE(cast_out);
  // scale
  PATTERN_DECL_NODE(scale_out);
  // mul0
  PATTERN_DECL_NODE(mul0_x);
  PATTERN_DECL_NODE(mul0_out);
  // mul1
  PATTERN_DECL_NODE(mul1_x);
  PATTERN_DECL_NODE(mul1_out);
  // add
  PATTERN_DECL_NODE(add_out);
};

OneFastWhereXPUPattern::OneFastWhereXPUPattern(PDPattern* pattern,
                                               const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  // cast
  auto cast_in =
      pattern->NewNode(cast_in_repr())->assert_is_op_input("cast", "X");
  auto cast_out = pattern->NewNode(cast_out_repr())
                      ->assert_is_op_output("cast", "Out")
                      ->assert_is_op_input("scale", "X")
                      ->assert_is_op_input("elementwise_mul");
  auto cast = pattern->NewNode(cast_repr())
                  ->assert_is_op("cast")
                  ->assert_more([](Node* n) {
                    auto in_dtype_val =
                        PADDLE_GET_CONST(int, n->Op()->GetAttr("in_dtype"));
                    auto out_dtype_val =
                        PADDLE_GET_CONST(int, n->Op()->GetAttr("out_dtype"));
                    return in_dtype_val == 0 &&
                           (out_dtype_val == 4 || out_dtype_val == 5);
                  });
  // scale
  auto scale_out = pattern->NewNode(scale_out_repr())
                       ->assert_is_op_output("scale", "Out")
                       ->assert_is_op_input("elementwise_mul");
  auto scale =
      pattern->NewNode(scale_repr())
          ->assert_is_op("scale")
          ->assert_more([](Node* n) {
            auto bias_val = PADDLE_GET_CONST(float, n->Op()->GetAttr("bias"));
            auto scale_val = PADDLE_GET_CONST(float, n->Op()->GetAttr("scale"));
            return bias_val == 1 && scale_val == -1;
          });
  // mul0
  auto mul0_x =
      pattern->NewNode(mul0_x_repr())->assert_is_op_input("elementwise_mul");
  auto mul0_out = pattern->NewNode(mul0_out_repr())
                      ->assert_is_op_output("elementwise_mul", "Out")
                      ->assert_is_op_input("elementwise_add");
  auto mul0 = pattern->NewNode(mul0_repr())
                  ->assert_is_op("elementwise_mul")
                  ->assert_more([](Node* node) {
                    auto node1 = node->inputs[0];
                    auto node2 = node->inputs[1];
                    auto node1_shape = node1->Var()->GetShape();
                    auto node2_shape = node2->Var()->GetShape();
                    if (node1_shape.size() != node2_shape.size() &&
                        node1_shape.size() != 2)
                      return false;
                    for (size_t i = 0; i < node1_shape.size(); i++) {
                      if (node1_shape[i] != node2_shape[i] &&
                          (node1_shape[i] != 1 && node2_shape[i] != 1)) {
                        return false;
                      }
                    }
                    return true;
                  });
  // mul1
  auto mul1_x =
      pattern->NewNode(mul1_x_repr())->assert_is_op_input("elementwise_mul");
  auto mul1_out = pattern->NewNode(mul1_out_repr())
                      ->assert_is_op_output("elementwise_mul", "Out")
                      ->assert_is_op_input("elementwise_add");
  auto mul1 = pattern->NewNode(mul1_repr())
                  ->assert_is_op("elementwise_mul")
                  ->assert_more([](Node* node) {
                    auto node1 = node->inputs[0];
                    auto node2 = node->inputs[1];
                    auto node1_shape = node1->Var()->GetShape();
                    auto node2_shape = node2->Var()->GetShape();
                    if (node1_shape.size() != node2_shape.size() &&
                        node1_shape.size() != 2)
                      return false;
                    for (size_t i = 0; i < node1_shape.size(); i++) {
                      if (node1_shape[i] != node2_shape[i] &&
                          (node1_shape[i] != 1 && node2_shape[i] != 1)) {
                        return false;
                      }
                    }
                    return true;
                  });
  // add
  auto add_out = pattern->NewNode(add_out_repr())
                     ->assert_is_op_output("elementwise_add", "Out");
  auto add = pattern->NewNode(add_repr())
                 ->assert_is_op("elementwise_add")
                 ->assert_more([](Node* node) {
                   // add -> in -> mul -> in
                   auto node_in1 = node->inputs[0];
                   auto node_in2 = node->inputs[1];
                   if (node_in1->inputs.size() == 1 &&
                       node_in1->inputs[0]->Op()->Type() == "elementwise_mul" &&
                       node_in2->inputs.size() == 1 &&
                       node_in2->inputs[0]->Op()->Type() == "elementwise_mul") {
                     auto mul1 = node_in1->inputs[0];
                     auto mul2 = node_in2->inputs[0];
                     auto shape1 = mul1->inputs[1]->Var()->GetShape();
                     auto shape2 = mul2->inputs[1]->Var()->GetShape();
                     return shape1 == shape2;
                   }
                   return false;
                 });
  cast->LinksFrom({cast_in}).LinksTo({cast_out});
  scale->LinksFrom({cast_out}).LinksTo({scale_out});
  mul0->LinksFrom({mul0_x, cast_out}).LinksTo({mul0_out});
  mul1->LinksFrom({mul1_x, scale_out}).LinksTo({mul1_out});
  add->LinksFrom({mul0_out, mul1_out}).LinksTo({add_out});
}

/*
fuse cascade fast_where_xpu ops to one fast_where_xpu op reduce memory access.
For example:
when mode = 0,

              x--------------
              |              |
              | condition1   |      y
              |      \       |      /
              |       \      |     /
              |        \     |    /
condition2    |       fast_where_xpu
     \        |         /
      \       |        /
       \      |       /
        fast_where_xpu

After the pass is applied,

condition1  condition2
     \          /
      \        /
          or
            \        x       y
             \       |      /
              \      |     /
              fast_where_xpu

or mode = 1,

  condition1   x        y
       \       |      /   |
        \      |     /    |
         \     |    /     |
        fast_where_xpu    |
               |          |
 condition2    |          |
      \        |         /
       \       |        /
        \      |       /
        fast_where_xpu

After the pass is applied,

condition1  condition2
     \          /
      \        /
       \      /
          and
            \        x       y
             \       |      /
              \      |     /
              fast_where_xpu

*/
struct CascadeFastWhereXPUPattern : public PatternBase {
  CascadeFastWhereXPUPattern(PDPattern* pattern,
                             const std::string& name_scope,
                             int mode);
  // declare operator node's name
  PATTERN_DECL_NODE(fast_where_xpu0);
  PATTERN_DECL_NODE(fast_where_xpu1);
  // declare variable node's name
  PATTERN_DECL_NODE(condition0);
  PATTERN_DECL_NODE(condition1);
  PATTERN_DECL_NODE(x);
  PATTERN_DECL_NODE(y);
  PATTERN_DECL_NODE(fast_where_xpu0_out);
  PATTERN_DECL_NODE(fast_where_xpu1_out);

 private:
  int mode_{0};
};

CascadeFastWhereXPUPattern::CascadeFastWhereXPUPattern(
    PDPattern* pattern, const std::string& name_scope, int mode)
    : PatternBase(pattern, name_scope, name_scope), mode_(mode) {
  // declare operator nodes
  auto fast_where_xpu0 =
      pattern->NewNode(fast_where_xpu0_repr())->assert_is_op("fast_where_xpu");
  auto fast_where_xpu1 =
      pattern->NewNode(fast_where_xpu0_repr())->assert_is_op("fast_where_xpu");
  // declare vairable nodes
  auto condition0 = pattern->NewNode(condition0_repr())
                        ->assert_is_op_input("fast_where_xpu", "condition");
  auto condition1 = pattern->NewNode(condition1_repr())
                        ->assert_is_op_input("fast_where_xpu", "condition");
  auto fast_where_xpu0_out = pattern->NewNode(fast_where_xpu0_out_repr())
                                 ->assert_is_op_output("fast_where_xpu", "Out");
  auto fast_where_xpu1_out = pattern->NewNode(fast_where_xpu1_out_repr())
                                 ->assert_is_op_output("fast_where_xpu", "Out");
  auto x = pattern->NewNode(x_repr())->assert_is_op_input("fast_where_xpu");
  auto y = pattern->NewNode(y_repr())->assert_is_op_input("fast_where_xpu");
  fast_where_xpu0->LinksFrom({condition0, x, y}).LinksTo({fast_where_xpu0_out});
  PADDLE_ENFORCE_GE(mode,
                    1,
                    platform::errors::InvalidArgument(
                        "cascade mode(%d) is not supported.", mode));
  if (mode == 0) {
    x->assert_is_op_input("fast_where_xpu", "x");
    y->assert_is_op_input("fast_where_xpu", "y");
    fast_where_xpu0_out->assert_is_op_input("fast_where_xpu", "y");
    fast_where_xpu1->LinksFrom({condition1, x, fast_where_xpu0_out})
        .LinksTo({fast_where_xpu1_out});
  } else if (mode == 1) {
    x->assert_is_op_input("fast_where_xpu", "x");
    y->assert_is_op_input("fast_where_xpu", "y");
    fast_where_xpu0_out->assert_is_op_input("fast_where_xpu", "x");
    fast_where_xpu1->LinksFrom({condition1, fast_where_xpu0_out, y})
        .LinksTo({fast_where_xpu1_out});
  }
}

}  // namespace patterns

class OneFastWhereXPUFusePass : public FusePassBase {
 public:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  int ApplySubgraph(ir::Graph* graph) const;

  const std::string name_scope_{"one_fast_where_xpu_fuse_pass"};
};

int OneFastWhereXPUFusePass::ApplySubgraph(ir::Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::OneFastWhereXPUPattern pattern(gpd.mutable_pattern(), name_scope_);
  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle FastWhereXPUFusePass fuse";
    // declare operator node's name
    GET_IR_NODE(cast);
    GET_IR_NODE(scale);
    GET_IR_NODE(mul0);
    GET_IR_NODE(mul1);
    GET_IR_NODE(add);
    // declare variable node's name
    // scale
    GET_IR_NODE(cast_in);
    GET_IR_NODE(cast_out);
    GET_IR_NODE(scale_out);
    // mul0
    GET_IR_NODE(mul0_x);
    GET_IR_NODE(mul0_out);
    // mul1
    GET_IR_NODE(mul1_x);
    GET_IR_NODE(mul1_out);
    // add
    GET_IR_NODE(add_out);

    auto* block = add->Op()->Block();
    auto* scope = param_scope();
    PADDLE_ENFORCE_NOT_NULL(
        scope, platform::errors::InvalidArgument("Scope cannot be nullptr."));
    framework::OpDesc fast_where_xpu_op_desc(block);
    fast_where_xpu_op_desc.SetType("fast_where_xpu");
    fast_where_xpu_op_desc.SetInput("condition", {cast_in->Name()});
    fast_where_xpu_op_desc.SetInput("x", {mul0_x->Name()});
    fast_where_xpu_op_desc.SetInput("y", {mul1_x->Name()});
    fast_where_xpu_op_desc.SetOutput("out", {add_out->Name()});
    auto fast_where_xpu_op_node = graph->CreateOpNode(&fast_where_xpu_op_desc);
    IR_NODE_LINK_TO(mul0_x, fast_where_xpu_op_node);
    IR_NODE_LINK_TO(mul1_x, fast_where_xpu_op_node);
    IR_NODE_LINK_TO(cast_in, fast_where_xpu_op_node);
    IR_NODE_LINK_TO(fast_where_xpu_op_node, add_out);
    std::unordered_set<const Node*> delete_nodes = {
        cast, cast_out, scale, scale_out, mul0, mul0_out, mul1, mul1_out, add};
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  return found_subgraph_count;
}

void OneFastWhereXPUFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);
  int found_subgraph_count = 0;
  found_subgraph_count += ApplySubgraph(graph);
  AddStatis(found_subgraph_count);
}

class CascadeFastWhereXPUFusePass : public FusePassBase {
 public:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  int ApplySubgraph(ir::Graph* graph, int mode) const;

  const std::string name_scope_{"cascade_fast_where_xpu_fuse_pass"};
};

int CascadeFastWhereXPUFusePass::ApplySubgraph(ir::Graph* graph,
                                               int mode) const {
  GraphPatternDetector gpd;
  patterns::CascadeFastWhereXPUPattern pattern(
      gpd.mutable_pattern(), name_scope_, mode);
  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle FastWhereXPUFusePass fuse";
    // declare operator node's name
    GET_IR_NODE(fast_where_xpu0);
    GET_IR_NODE(fast_where_xpu1);
    // declare variable node's name
    GET_IR_NODE(condition0);
    GET_IR_NODE(condition1);
    GET_IR_NODE(x);
    GET_IR_NODE(y);
    GET_IR_NODE(fast_where_xpu0_out);
    GET_IR_NODE(fast_where_xpu1_out);

    auto* block = fast_where_xpu0->Op()->Block();
    auto* scope = param_scope();
    PADDLE_ENFORCE_NOT_NULL(
        scope, platform::errors::InvalidArgument("Scope cannot be nullptr."));
    // Reuse variables
    fast_where_xpu0_out->Var()->SetShape(condition0->Var()->GetShape());
    fast_where_xpu0_out->Var()->SetDataType(condition0->Var()->GetDataType());
    // Change the first fast_where_xpu op to logical op
    fast_where_xpu0->Op()->SetType("logical_and");
    fast_where_xpu0->Op()->SetInput(
        "X", std::vector<std::string>({condition0->Name()}));
    fast_where_xpu0->Op()->SetInput(
        "Y", std::vector<std::string>({condition1->Name()}));
    // Reserve the second first_where_xpu but change its inputs
    fast_where_xpu1->Op()->SetInput(
        "condition", std::vector<std::string>({fast_where_xpu0_out->Name()}));
    fast_where_xpu1->Op()->SetInput("x", std::vector<std::string>({x->Name()}));
    fast_where_xpu1->Op()->SetInput("y", std::vector<std::string>({y->Name()}));
    IR_NODE_UNLINK(x, fast_where_xpu0);
    IR_NODE_UNLINK(y, fast_where_xpu0);
    IR_NODE_LINK_TO(condition1, fast_where_xpu0);
    IR_NODE_UNLINK(condition1, fast_where_xpu1);
    IR_NODE_LINK_TO(x, fast_where_xpu1);
    IR_NODE_LINK_TO(y, fast_where_xpu1);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  return found_subgraph_count;
}

void CascadeFastWhereXPUFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);
  int found_subgraph_count = 0;
  for (auto mode : {0, 1}) {
    found_subgraph_count += ApplySubgraph(graph, mode);
  }
  AddStatis(found_subgraph_count);
}

class FastWhereXPUFusePass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  const std::string name_scope_{"fast_where_xpu_fuse_pass"};
};

void FastWhereXPUFusePass::ApplyImpl(ir::Graph* graph) const {
  VLOG(4) << "handle fast_where_xpu op fusion.";
  OneFastWhereXPUFusePass one_fast_where_xpu_fuse_pass;
  one_fast_where_xpu_fuse_pass.ApplyImpl(graph);
  // CascadeFastWhereXPUFusePass cascade_fast_where_xpu_fuse_pass;
  // cascade_fast_where_xpu_fuse_pass.ApplyImpl(graph);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fast_where_xpu_fuse_pass,
              paddle::framework::ir::FastWhereXPUFusePass);

REGISTER_PASS_CAPABILITY(fast_where_xpu_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "fast_where_xpu_fuse_pass", 0));
