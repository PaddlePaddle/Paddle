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

#include "paddle/fluid/framework/ir/xpu/conv2d_bias_fuse_pass.h"

#include "glog/logging.h"

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

struct FcBiasPattern : public PatternBase {
  FcBiasPattern(PDPattern* pattern,
                const std::string& name_scope,
                const std::string& mul_type);

  // declare operator node's name
  PATTERN_DECL_NODE(ew_bias_add);
  // declare variable node's name
  PATTERN_DECL_NODE(mul_out);
  PATTERN_DECL_NODE(ew_bias_add_x);
  PATTERN_DECL_NODE(ew_bias_add_out);

 private:
  std::string mul_type_;
};

FcBiasPattern::FcBiasPattern(PDPattern* pattern,
                             const std::string& name_scope,
                             const std::string& mul_type)
    : PatternBase(pattern, name_scope, name_scope), mul_type_(mul_type) {
  auto* mul_out = pattern->NewNode(mul_out_repr())
                      ->assert_is_op_output(mul_type_, "Out")
                      ->assert_is_op_input("elementwise_add", "Y")
                      ->assert_has_n_outputs(1);
  auto* ew_bias_add = pattern->NewNode(ew_bias_add_repr())
                          ->assert_is_op("elementwise_add")
                          ->assert_more([](Node* node) {
                            auto* op_desc = node->Op();
                            auto axis = op_desc->GetAttrIfExists<int>("axis");
                            return axis == -1;
                          });
  auto* ew_bias_add_x = pattern->NewNode(ew_bias_add_x_repr())
                            ->assert_is_op_input("elementwise_add", "X")
                            ->assert_is_persistable_var()
                            ->assert_has_n_outputs(1);
  auto* ew_bias_add_out = pattern->NewNode(ew_bias_add_out_repr())
                              ->assert_is_op_output("elementwise_add", "Out");
  ew_bias_add->LinksFrom({mul_out, ew_bias_add_x}).LinksTo({ew_bias_add_out});
}

struct Conv2dBiasPattern : public PatternBase {
  Conv2dBiasPattern(PDPattern* pattern, const std::string& name_scope);

  // declare operator node's name
  PATTERN_DECL_NODE(ew_bias_add);
  // declare variable node's name
  PATTERN_DECL_NODE(x);
  PATTERN_DECL_NODE(ew_bias_add_y);
  PATTERN_DECL_NODE(ew_bias_add_out);
};

Conv2dBiasPattern::Conv2dBiasPattern(PDPattern* pattern,
                                     const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  auto* x = pattern->NewNode(x_repr())
                ->assert_is_op_output("conv2d", "Output")
                ->assert_is_op_input("elementwise_add", "X")
                ->assert_has_n_outputs(1);
  auto* ew_bias_add = pattern->NewNode(ew_bias_add_repr())
                          ->assert_is_op("elementwise_add")
                          ->assert_more([](Node* node) {
                            auto* op_desc = node->Op();
                            auto axis = op_desc->GetAttrIfExists<int>("axis");
                            return axis == -1;
                          });
  auto* ew_bias_add_y = pattern->NewNode(ew_bias_add_y_repr())
                            ->assert_is_op_input("elementwise_add", "Y")
                            ->assert_is_persistable_var()
                            ->assert_has_n_outputs(1)
                            ->assert_more([](Node* node) {
                              auto y_shape = node->Var()->GetShape();
                              size_t y_rank = y_shape.size();
                              return y_rank == 4 && y_shape[0] == 1 &&
                                     y_shape[2] == 1 && y_shape[3] == 1;
                            });
  auto* ew_bias_add_out = pattern->NewNode(ew_bias_add_out_repr())
                              ->assert_is_op_output("elementwise_add", "Out");
  ew_bias_add->LinksFrom({x, ew_bias_add_y}).LinksTo({ew_bias_add_out});
}

struct ScaleFusePattern : public PatternBase {
  ScaleFusePattern(PDPattern* pattern, const std::string& name_scope);
  // declare operator node's name
  PATTERN_DECL_NODE(ele_mul);
  PATTERN_DECL_NODE(ele_add);
  // declare variable node's name
  PATTERN_DECL_NODE(x);
  PATTERN_DECL_NODE(ele_mul_y);
  PATTERN_DECL_NODE(ele_mul_out);
  PATTERN_DECL_NODE(ele_add_y);
  PATTERN_DECL_NODE(ele_add_out);
};

ScaleFusePattern::ScaleFusePattern(PDPattern* pattern,
                                   const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  // ele_mul op
  auto ele_mul =
      pattern->NewNode(ele_mul_repr())->assert_is_op("elementwise_mul");
  auto x = pattern->NewNode(x_repr())
               ->assert_is_op_input("elementwise_mul", "X")
               ->AsInput();
  auto ele_mul_y = pattern->NewNode(ele_mul_y_repr())
                       ->assert_is_op_input("elementwise_mul", "Y")
                       ->assert_is_persistable_var()
                       ->assert_has_n_outputs(1)
                       ->assert_more([](Node* node) {
                         return node->Var()->GetShape().size() == 1;
                       });
  auto ele_mul_out = pattern->NewNode(ele_mul_out_repr())
                         ->assert_is_op_output("elementwise_mul", "Out")
                         ->assert_is_op_input("elementwise_add", "X")
                         ->assert_has_n_outputs(1);
  ele_mul->LinksFrom({x, ele_mul_y}).LinksTo({ele_mul_out});
  // ele_add op
  auto ele_add =
      pattern->NewNode(ele_add_repr())->assert_is_op("elementwise_add");
  auto ele_add_y = pattern->NewNode(ele_add_y_repr())
                       ->assert_is_op_input("elementwise_add", "Y")
                       ->assert_is_persistable_var()
                       ->assert_has_n_outputs(1)
                       ->assert_more([](Node* node) {
                         return node->Var()->GetShape().size() == 1;
                       });
  auto ele_add_out = pattern->NewNode(ele_add_out_repr())
                         ->assert_is_op_output("elementwise_add", "Out");
  ele_add->LinksFrom({ele_mul_out, ele_add_y}).LinksTo({ele_add_out});
}

}  // namespace patterns

void Conv2dBiasFusePass::TransFcBias(ir::Graph* graph,
                                     const std::string& mul_type) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);
  GraphPatternDetector gpd;
  patterns::FcBiasPattern pattern(gpd.mutable_pattern(), name_scope_, mul_type);

  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle TransFcBias fuse";
    // declare operator node's name
    GET_IR_NODE(ew_bias_add);
    // declare variable node's name
    GET_IR_NODE(mul_out);
    GET_IR_NODE(ew_bias_add_x);
    GET_IR_NODE(ew_bias_add_out);

    // trans link order of x && y for ew_bias_add op
    auto ew_bias_add_desc = ew_bias_add->Op();
    IR_NODE_UNLINK(mul_out, ew_bias_add);
    IR_NODE_UNLINK(ew_bias_add_x, ew_bias_add);
    ew_bias_add_desc->RemoveInput("X");
    ew_bias_add_desc->RemoveInput("Y");
    ew_bias_add_desc->Flush();
    ew_bias_add_desc->SetInput("X", {mul_out->Name()});
    ew_bias_add_desc->SetInput("Y", {ew_bias_add_x->Name()});
    IR_OP_VAR_LINK(mul_out, ew_bias_add);
    IR_OP_VAR_LINK(ew_bias_add_x, ew_bias_add);

    found_subgraph_count++;
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

void Conv2dBiasFusePass::FoldConv2dBias(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);
  GraphPatternDetector gpd;
  patterns::Conv2dBiasPattern pattern(gpd.mutable_pattern(), name_scope_);

  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle TransEwBiasAdd fuse";
    // declare operator node's name
    GET_IR_NODE(ew_bias_add);
    // declare variable node's name
    GET_IR_NODE(x);
    GET_IR_NODE(ew_bias_add_y);
    GET_IR_NODE(ew_bias_add_out);

    auto* scope = param_scope();
    // resize 4D dims of ew_bias_add_y to 1-D dim
    auto ew_bias_add_desc = ew_bias_add->Op();
    ew_bias_add_desc->SetAttr("axis", 1);
    auto* ew_bias_add_y_desc = ew_bias_add_y->Var();
    auto y_shape = ew_bias_add_y_desc->GetShape();
    ew_bias_add_y_desc->SetShape({y_shape[1]});
    auto* ew_bias_add_y_tensor =
        scope->GetVar(ew_bias_add_y->Name())->GetMutable<phi::DenseTensor>();
    ew_bias_add_y_tensor->Resize(common::make_ddim({y_shape[1]}));
    ew_bias_add_desc->Flush();

    found_subgraph_count++;
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

void Conv2dBiasFusePass::FuseScaleOps(ir::Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::ScaleFusePattern pattern(gpd.mutable_pattern(), name_scope_);
  int found_subgraph_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle FuseScaleOps";
    /* declare operator node's name */
    GET_IR_NODE(ele_mul);
    GET_IR_NODE(ele_add);
    // declare variable node's name
    GET_IR_NODE(x);
    GET_IR_NODE(ele_mul_y);
    GET_IR_NODE(ele_mul_out);
    GET_IR_NODE(ele_add_y);
    GET_IR_NODE(ele_add_out);

    auto* scope = param_scope();
    PADDLE_ENFORCE_NOT_NULL(
        scope, common::errors::InvalidArgument("Scope cannot be nullptr."));
    // get attrs of scale from ele_mul && ele_add
    const auto& ele_mul_y_t =
        scope->GetVar(ele_mul_y->Name())->GetMutable<phi::DenseTensor>();
    auto ele_mul_y_t_len = ele_mul_y_t->numel();
    PADDLE_ENFORCE_EQ(
        ele_mul_y_t_len,
        1,
        common::errors::InvalidArgument("the size(%ld) of ele_mul y tensor "
                                        "must equal 1",
                                        ele_mul_y_t_len));
    const auto& ele_add_y_t =
        scope->GetVar(ele_add_y->Name())->GetMutable<phi::DenseTensor>();
    auto ele_add_y_t_len = ele_add_y_t->numel();
    PADDLE_ENFORCE_EQ(
        ele_add_y_t_len,
        1,
        common::errors::InvalidArgument("the size(%ld) of ele_add y tensor "
                                        "must equal 1",
                                        ele_mul_y_t_len));
    auto tensor_type = ele_mul_y_t->dtype();
    float scale_val_ = 1.f;
    float bias_val_ = 0.f;
    if (tensor_type == phi::DataType::FLOAT16) {
      CastToFp32(ele_mul_y_t, nullptr);
      CastToFp32(ele_add_y_t, nullptr);
    }
    float* ele_mul_y_ptr = ele_mul_y_t->mutable_data<float>(phi::CPUPlace());
    float* ele_add_y_ptr = ele_add_y_t->mutable_data<float>(phi::CPUPlace());
    scale_val_ = ele_mul_y_ptr[0];
    bias_val_ = ele_add_y_ptr[0];
    // replace ele_mul+ele_add with scale
    OpDesc new_desc;
    new_desc.SetType("scale");
    new_desc.SetAttr("bias_after_scale", true);
    new_desc.SetAttr("scale", scale_val_);
    new_desc.SetAttr("bias", bias_val_);
    new_desc.SetInput("X", {x->Name()});
    new_desc.SetOutput("Out", {ele_add_out->Name()});
    new_desc.Flush();

    auto fused_node = graph->CreateOpNode(&new_desc);
    IR_NODE_LINK_TO(x, fused_node);
    IR_NODE_LINK_TO(fused_node, ele_add_out);

    std::unordered_set<const Node*> del_node_set = {
        ele_mul, ele_mul_y, ele_mul_out, ele_add, ele_add_y};
    GraphSafeRemoveNodes(graph, del_node_set);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

void Conv2dBiasFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);
  // for conv2d + scale fuse
  FuseScaleOps(graph);
  // for conv2d + ew_bias_add + scale fuse
  FoldConv2dBias(graph);
  // for matmul + ew_bias_add fuse
  for (auto mul_type : {"mul", "matmul", "matmul_v2"}) {
    TransFcBias(graph, mul_type);
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(conv2d_bias_fuse_pass, paddle::framework::ir::Conv2dBiasFusePass);

REGISTER_PASS_CAPABILITY(conv2d_bias_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .EQ("conv2d", 0)
            .EQ("mul", 0)
            .LE("elementwise_add", 1));
