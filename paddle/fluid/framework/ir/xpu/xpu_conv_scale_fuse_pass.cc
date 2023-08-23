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

#include "paddle/fluid/framework/ir/xpu/xpu_conv_scale_fuse_pass.h"

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

struct XPUConvScaleFusePattern : public PatternBase {
  XPUConvScaleFusePattern(PDPattern* pattern,
                          const std::string& name_scope,
                          const std::string& conv_type,
                          bool conv_with_bias);
  // declare operator node's name
  PATTERN_DECL_NODE(xpu_conv);
  PATTERN_DECL_NODE(scale);
  // declare variable node's name
  PATTERN_DECL_NODE(conv_input);
  PATTERN_DECL_NODE(conv_filter);
  PATTERN_DECL_NODE(conv_filter_max);
  PATTERN_DECL_NODE(conv_bias);
  PATTERN_DECL_NODE(conv_out);
  PATTERN_DECL_NODE(scale_out);

 private:
  std::string conv_type_;
  bool conv_with_bias_{false};
};

XPUConvScaleFusePattern::XPUConvScaleFusePattern(PDPattern* pattern,
                                                 const std::string& name_scope,
                                                 const std::string& conv_type,
                                                 bool conv_with_bias)
    : PatternBase(pattern, name_scope, name_scope),
      conv_type_(conv_type),
      conv_with_bias_(conv_with_bias) {
  auto xpu_conv = pattern->NewNode(xpu_conv_repr())->assert_is_op(conv_type_);
  auto conv_input = pattern->NewNode(conv_input_repr())
                        ->assert_is_op_input(conv_type_, "x")
                        ->AsInput()
                        ->assert_more([](Node* node) {
                          return node->Var()->GetShape().size() == 4;
                        });
  auto conv_filter = pattern->NewNode(conv_filter_repr())
                         ->assert_is_op_input(conv_type_, "filter")
                         ->AsInput();
  auto conv_filter_max = pattern->NewNode(conv_filter_max_repr())
                             ->assert_is_op_input(conv_type_, "filter_max")
                             ->AsInput();
  auto conv_out = pattern->NewNode(conv_out_repr())
                      ->assert_is_op_output(conv_type_, "out")
                      ->assert_is_op_input("scale", "X")
                      ->assert_has_n_outputs(1);
  PDNode* conv_bias = nullptr;
  if (conv_with_bias_) {
    conv_bias = pattern->NewNode(conv_bias_repr())
                    ->assert_is_op_input(conv_type_, "bias")
                    ->AsInput();
    xpu_conv->LinksFrom({conv_input, conv_filter, conv_filter_max, conv_bias})
        .LinksTo({conv_out});
  } else {
    xpu_conv->LinksFrom({conv_input, conv_filter, conv_filter_max})
        .LinksTo({conv_out});
  }
  // scale op
  auto scale = pattern->NewNode(scale_repr())->assert_is_op("scale");
  auto scale_out = pattern->NewNode(scale_out_repr())
                       ->assert_is_op_output("scale", "Out")
                       ->AsOutput();
  scale->LinksFrom({conv_out}).LinksTo({scale_out});
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

void XPUConvScaleFusePass::FuseScaleOps(ir::Graph* graph) const {
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

    VLOG(1) << "scale fuse done";
    auto* scope = param_scope();
    PADDLE_ENFORCE_NOT_NULL(
        scope, platform::errors::InvalidArgument("Scope cannot be nullptr."));
    // get attrs of scale from ele_mul && ele_add
    const auto& ele_mul_y_t =
        scope->GetVar(ele_mul_y->Name())->GetMutable<phi::DenseTensor>();
    auto ele_mul_y_t_dims = ele_mul_y_t->dims();
    PADDLE_ENFORCE_EQ(
        ele_mul_y_t_dims.size(),
        1,
        platform::errors::InvalidArgument("the size(%d) of ele_mul y tensor "
                                          "must equal 1",
                                          ele_mul_y_t_dims.size()));
    const auto& ele_add_y_t =
        scope->GetVar(ele_add_y->Name())->GetMutable<phi::DenseTensor>();
    auto ele_add_y_t_dims = ele_add_y_t->dims();
    PADDLE_ENFORCE_EQ(
        ele_add_y_t_dims.size(),
        1,
        platform::errors::InvalidArgument("the size(%d) of ele_add y tensor "
                                          "must equal 1",
                                          ele_add_y_t_dims.size()));
    auto tensor_type = ele_mul_y_t->dtype();
    float scale_val_ = 1.f;
    float bias_val_ = 0.f;
    if (tensor_type == phi::DataType::FLOAT16) {
      CastToFp32(ele_mul_y_t, nullptr);
      CastToFp32(ele_add_y_t, nullptr);
    }
    float* ele_mul_y_ptr =
        ele_mul_y_t->mutable_data<float>(paddle::platform::CPUPlace());
    float* ele_add_y_ptr =
        ele_add_y_t->mutable_data<float>(paddle::platform::CPUPlace());
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

void XPUConvScaleFusePass::FuseConvScale(ir::Graph* graph,
                                         const std::string& conv_type,
                                         bool conv_with_bias) const {
  GraphPatternDetector gpd;
  patterns::XPUConvScaleFusePattern pattern(
      gpd.mutable_pattern(), name_scope_, conv_type, conv_with_bias);
  int found_subgraph_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle FuseConvScale";
    /* declare operator node's name */
    GET_IR_NODE(xpu_conv);
    GET_IR_NODE(scale);
    // declare variable node's name
    GET_IR_NODE(conv_input);
    GET_IR_NODE(conv_filter);
    GET_IR_NODE(conv_filter_max);
    GET_IR_NODE(conv_bias);
    GET_IR_NODE(conv_out);
    GET_IR_NODE(scale_out);

    VLOG(1) << "conv scale fuse";
    auto* block = xpu_conv->Op()->Block();
    auto* scope = param_scope();
    PADDLE_ENFORCE_NOT_NULL(
        scope, platform::errors::InvalidArgument("Scope cannot be nullptr."));
    // get attrs of scale
    float scale_val = 1.f;
    float bias_val = 0.f;
    scale_val = PADDLE_GET_CONST(float, scale->Op()->GetAttr("scale"));
    bias_val = PADDLE_GET_CONST(float, scale->Op()->GetAttr("bias"));
    bool bias_after_scale_ =
        PADDLE_GET_CONST(bool, scale->Op()->GetAttr("bias_after_scale"));
    // recompute weight for conv2d_xpu op
    Node* conv_filter_node = nullptr;
    // trans conv_filter from int16 to fp32
    // PrepareBias(graph, scope, block, conv_filter, &conv_filter_node);
    auto src_name = conv_filter->Name();
    auto* src_tensor = scope->Var(src_name)->GetMutable<phi::DenseTensor>();
    auto filter_dims = src_tensor->dims();
    phi::DenseTensor dst_tensor;
    CastToFp32(src_tensor, &dst_tensor);
    std::string pre_name = GetPrefixWithoutHash(src_name);
    std::string dst_name = pre_name + "_fusion_scale_filter";
    conv_filter_node = FindNodeWithName(graph, dst_name);
    if (conv_filter_node == nullptr) {
      // Create dst node
      // Update dst var_desc in block
      VarDesc dst_desc(dst_name);
      dst_desc.SetPersistable(true);
      dst_desc.SetShape(vectorize(dst_tensor.dims()));
      dst_desc.SetDataType(framework::TransToProtoVarType(dst_tensor.dtype()));
      conv_filter_node = graph->CreateVarNode(&dst_desc);
      auto* block_dst_desc = block->Var(dst_name);
      block_dst_desc->SetPersistable(dst_desc.Persistable());
      block_dst_desc->SetShape(dst_desc.GetShape());
      block_dst_desc->SetDataType(dst_desc.GetDataType());
      Assign(dst_tensor, scope->Var(dst_name)->GetMutable<phi::DenseTensor>());
    }
    // compute new value of conv_filter tensor
    auto* filter_t =
        scope->GetVar(conv_filter_node->Name())->GetMutable<phi::DenseTensor>();
    auto filter_len = filter_t->numel();
    for (int i = 0; i < filter_len; ++i) {
      float* filter_ptr =
          filter_t->mutable_data<float>(paddle::platform::CPUPlace());
      filter_ptr[i] *= scale_val;
    }
    // update filter && filter_max
    Node* filter_int16 = nullptr;
    Node* filter_max = nullptr;
    PrepareWeight<int16_t>(graph,
                           scope,
                           block,
                           conv_filter_node,
                           &filter_int16,
                           &filter_max,
                           false);
    // Generate conv2d_xpu op
    VLOG(1) << "filter name :" << conv_filter_node->Name();
    VLOG(1) << "filter name :" << filter_int16->Name();
    VLOG(1) << "filter_max name :" << filter_max->Name();
    // recompute bias for conv2d_xpu op
    // Node* conv_bias_node = nullptr;
    Node* tmp_bias_node = nullptr;
    phi::DenseTensor* tmp_bias_tensor;
    auto bias_len = filter_dims[1];
    VLOG(1) << "conv_with_bias is : " << (conv_with_bias ? "true" : "false");
    VLOG(1) << "bias dims is: " << bias_len;
    if (!conv_with_bias) {
      // create conv2d_xpu bias node
      std::string tmp_bias_name = scale_out->Name() + "_conv_scale_fusion_bias";
      tmp_bias_node = FindNodeWithName(graph, tmp_bias_name);
      if (tmp_bias_node == nullptr) {
        VarDesc tmp_bias_desc(tmp_bias_name);
        tmp_bias_desc.SetPersistable(true);
        tmp_bias_desc.SetShape(std::vector<int64_t>({bias_len}));
        tmp_bias_desc.SetDataType(proto::VarType::FP32);
        tmp_bias_node = graph->CreateVarNode(&tmp_bias_desc);
        tmp_bias_tensor =
            scope->Var(tmp_bias_name)->GetMutable<phi::DenseTensor>();
        // Initialize tmp_bias
        tmp_bias_tensor->Resize(phi::make_ddim({bias_len}));
        std::fill_n(tmp_bias_tensor->mutable_data<float>(platform::CPUPlace()),
                    tmp_bias_tensor->numel(),
                    0.0f);
      }
      VLOG(1) << "conv bias name is :" << tmp_bias_node->Name();
      auto conv_bias_t =
          scope->GetVar(tmp_bias_node->Name())->GetMutable<phi::DenseTensor>();
      float* conv_bias_ptr =
          conv_bias_t->mutable_data<float>(paddle::platform::CPUPlace());
      for (int i = 0; i < bias_len; ++i) {
        if (bias_after_scale_) {
          conv_bias_ptr[i] = conv_bias_ptr[i] * scale_val + bias_val;
        } else {
          conv_bias_ptr[i] = (conv_bias_ptr[i] + bias_val) * scale_val;
        }
      }
    } else {
      auto conv_bias_t =
          scope->GetVar(conv_bias->Name())->GetMutable<phi::DenseTensor>();
      float* conv_bias_ptr =
          conv_bias_t->mutable_data<float>(paddle::platform::CPUPlace());
      for (int i = 0; i < bias_len; ++i) {
        if (bias_after_scale_) {
          conv_bias_ptr[i] = conv_bias_ptr[i] * scale_val + bias_val;
        } else {
          conv_bias_ptr[i] = (conv_bias_ptr[i] + bias_val) * scale_val;
        }
      }
    }
    // update conv2d_xpu op
    auto* conv2d_xpu_op_desc = xpu_conv->Op();
    IR_NODE_UNLINK(conv_filter, xpu_conv);
    IR_NODE_UNLINK(conv_filter_max, xpu_conv);
    conv2d_xpu_op_desc->SetInput("filter", {filter_int16->Name()});
    conv2d_xpu_op_desc->SetInput("filter_max", {filter_max->Name()});
    IR_NODE_LINK_TO(filter_int16, xpu_conv);
    IR_NODE_LINK_TO(filter_max, xpu_conv);
    if (!conv_with_bias) {
      conv2d_xpu_op_desc->SetInput("bias", {tmp_bias_node->Name()});
      IR_NODE_LINK_TO(tmp_bias_node, xpu_conv);
    }
    // link xpu_conv to ops which are behind scale
    auto scale_out_link_nodes = scale_out->outputs;
    for (auto out_link_node : scale_out_link_nodes) {
      auto op_desc = out_link_node->Op();
      op_desc->RenameInput(scale_out->Var()->Name(), conv_out->Var()->Name());
      op_desc->Flush();
      IR_NODE_LINK_TO(conv_out, out_link_node);
    }
    // delete useless node
    std::unordered_set<const Node*> delete_nodes = {
        conv_filter, conv_filter_max, conv_filter_node, scale, scale_out};
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

void XPUConvScaleFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);

  FuseScaleOps(graph);
  for (auto conv_type : {"conv2d_xpu"}) {
    for (auto conv_with_bias : {true, false}) {
      FuseConvScale(graph, conv_type, conv_with_bias);
    }
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(xpu_conv_scale_fuse_pass,
              paddle::framework::ir::XPUConvScaleFusePass);

REGISTER_PASS_CAPABILITY(xpu_conv_scale_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "conv2d_xpu", 0));
