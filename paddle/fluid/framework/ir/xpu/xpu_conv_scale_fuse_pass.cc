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
  auto xpu_conv = pattern->NewNode(conv_repr())->assert_is_op(conv_type_);
  auto conv_input = pattern->NewNode(conv_input_repr())
                        ->assert_is_op_input(conv_type_, "x")
                        ->AsInput()
                        ->assert_more([](Node* node) {
                          return node->Var()->GetShape().size() == 4;
                        });
  auto conv_filter = pattern->NewNode(conv_filter_repr())
                         ->assert_is_op_input(conv_type_, "filter")
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
    xpu_conv->LinksFrom({conv_input, conv_filter, conv_bias})
        .LinksTo({conv_out});
  } else {
    xpu_conv->LinksFrom({conv_input, conv_filter}).LinksTo({conv_out});
  }
  // scale op
  auto scale = pattern->NewNode(scale_repr())->assert_is_op("scale");
  auto scale_out = pattern->NewNode(scale_out_repr())
                       ->assert_is_op_output("scale", "Out")
                       ->AsOutput();
  scale->LinksFrom({conv_out}).LinksTo({scale_out});
}

}  // namespace patterns

/*
fuse conv2d block in resnet50-like model to xpu_conv2d op
For example:
sub block
                   conv_input
                      |
                   conv2d_xpu
                      |
                    scale
                      |
                   other_ops
------------------------------------------------------
After the pass is applied:
                    conv_input
                       |
                       |
  conv_filter ----- conv2d_xpu ----- conv_bias
                       |
                       |
                    other_ops
*/
class XPUConvScaleFusePass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  int ApplyImpl(ir::Graph* graph,
                const std::string& conv_type,
                bool conv_with_bias) const;

  const std::string name_scope_{"xpu_conv_scale_fuse_pass"};
};

void XPUConvScaleFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);

  int found_subgraph_count = 0;
  for (auto conv_type : {"conv2d_xpu"}) {
    for (auto conv_with_bias : {true, false}) {
      found_subgraph_count += ApplyImpl(graph, conv_type, conv_with_bias);
    }
  }
  AddStatis(found_subgraph_count);
}

int XPUConvScaleFusePass::ApplyImpl(ir::Graph* graph,
                                    const std::string& conv_type,
                                    bool conv_with_bias) const {
  GraphPatternDetector gpd;
  patterns::XPUConvScaleFusePattern pattern(
      gpd.mutable_pattern(), name_scope_, conv_type, conv_with_bias);
  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle XPUConv2dScaleFusePass";
    /* declare operator node's name */
    GET_IR_NODE(xpu_conv);
    GET_IR_NODE(scale);
    // declare variable node's name
    GET_IR_NODE(conv_input);
    GET_IR_NODE(conv_filter);
    GET_IR_NODE(conv_bias);
    GET_IR_NODE(conv_out);
    GET_IR_NODE(scale_out);

    auto* block = xpu_conv->Op()->Block();
    auto* scope = param_scope();
    PADDLE_ENFORCE_NOT_NULL(
        scope, platform::errors::InvalidArgument("Scope cannot be nullptr."));
    // get attrs of scale
    float scale_val = 1.f;
    float bias_val = 0.f;
    scale_val = PADDLE_GET_CONST(float, scale->Op()->GetAttr<float>("scale"));
    bias_val = PADDLE_GET_CONST(float, scale->Op()->GetAttr<float>("bias"));
    bool bias_after_scale_ =
        PADDLE_GET_CONST(bool, scale->Op()->GetAttr<bool>("bias_after_scale"));
    // recompute weight for conv2d_xpu op
    auto* filter_t =
        scope->FindVar(conv_filter->Name())->GetMutable<phi::DenseTensor>();
    // conv_filter fp16 --> fp32
    auto filter_dtype = filter_t->dtype();
    auto filter_dims = filter_t->dims();
    if (filter_dtype == phi::DataType::FLOAT16) {
      CastToFp32(filter_t, nullptr);
    }
    auto filter_len = filter_t->numel();
    for (int i = 0; i < filter_len; ++i) {
      float* filter_ptr =
          filter_t->mutable_data<float>(paddle::platform::CPUPlace());
      filter_ptr[i] *= scale_val;
    }
    // recompute bias for conv2d_xpu op
    Node* conv_bias_node = nullptr;
    if (!conv_with_bias) {
      // create conv2d_xpu bias node
      std::string conv_bias_name = xpu_conv->Name() + "_bias";
      VarDesc conv_bias_desc(conv_bias_name);
      conv_bias_desc.SetPersistable(true);
      conv_bias_desc.SetShape(std::vector<int64_t>({filter_dims[1]}));
      conv_bias_desc.SetDataType(phi::DataType::FLOAT32);
      conv_bias_node = graph->CreateVarNode(&conv_bias_desc);
      auto conv_bias_t =
          scope->Var(conv_bias_node->Name())->GetMutable<phi::DenseTensor>();
      auto bias_len = conv_bias_t->numel();
      float* conv_bias_ptr =
          conv_bias_t->mutable_data<float>(paddle::platform::CPUPlace());
      for (int i = 0; i < bias_len; ++i) {
        conv_bias_ptr[i] = 0.f;
      }
    } else {
      PrepareBias(graph, scope, block, conv_bias, &conv_bias_node);
    }
    auto conv_bias_t =
        scope->Var(conv_bias_node->Name())->GetMutable<phi::DenseTensor>();
    auto bias_len = conv_bias_t->numel();
    float* conv_bias_ptr =
        conv_bias_t->mutable_data<float>(paddle::platform::CPUPlace());
    for (int i = 0; i < bias_len; ++i) {
      if (bias_after_scale_) {
        conv_bias_ptr[i] = conv_bias_ptr[i] * scale_val + bias_val;
      } else {
        conv_bias_ptr[i] = (conv_bias_ptr[i] + bias_val) * scale_val;
      }
    }

    // Generate conv2d_xpu op
    auto* conv2d_xpu_op_desc = xpu_conv->Op();
    if (conv_with_bias) {
      IR_NODE_UNLINK(conv_bias, xpu_conv);
    }
    conv2d_xpu_op_desc->SetInput("bias", {conv_bias_node->Name()});
    IR_NODE_LINK_TO(conv_bias_node, xpu_conv);
    // link xpu_conv to ops which are behind scale
    auto scale_out_link_nodes = scale_out->outputs;
    for (auto out_link_node : scale_out_link_nodes) {
      auto op_desc = out_link_node->Op();
      op_desc->RenameInput(scale_out->Var()->Name(), conv_out->Var()->Name());
      op_desc->Flush();
      IR_NODE_LINK_TO(conv_out, out_link_node);
    }
    // delete useless node
    std::unordered_set<const Node*> delete_nodes = {scale, scale_out};
    if (conv_with_bias) {
      delete_nodes.insert(conv_bias);
    }
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  return found_subgraph_count;
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
