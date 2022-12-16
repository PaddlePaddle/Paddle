// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/conv_elementwise_add_act_fuse_pass.h"

#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace framework {
namespace ir {

#define GET_IR_NODE(node__) GET_IR_NODE_FROM_SUBGRAPH(node__, node__, pattern);
#define GET_NODES                    \
  GET_IR_NODE(conv_op);              \
  GET_IR_NODE(conv_out);             \
  GET_IR_NODE(conv_filter);          \
  GET_IR_NODE(elementwise_add_op);   \
  GET_IR_NODE(elementwise_add_in_y); \
  GET_IR_NODE(elementwise_add_out);  \
  GET_IR_NODE(act_op);               \
  GET_IR_NODE(act_out);

// Inherient the basic information from `base_desc`, and modify some fields.
framework::proto::OpDesc PrepareOpDesc(
    const framework::proto::OpDesc& base_desc,
    const std::string& bias,
    const std::string& activation,
    const std::string& output,
    float alpha) {
  auto proto = base_desc;
  framework::OpDesc desc(proto, nullptr);
  desc.SetType("conv2d_fusion");
  desc.SetInput("Bias", {bias});
  desc.SetInput("ResidualData", {});
  desc.SetAttr("activation", activation);
  desc.SetOutput("Output", {output});
  desc.SetAttr("is_test", true);
  desc.SetAttr("use_cudnn", false);
  // for leaky_relu use
  desc.SetAttr("fuse_alpha", alpha);
  desc.Flush();
  return *desc.Proto();
}

ConvElementwiseAddActFusePass::ConvElementwiseAddActFusePass() {
  AddOpCompat(OpCompat("conv2d"))
      .AddInput("Input")
      .IsTensor()
      .End()
      .AddInput("Filter")
      .IsTensor()
      .End()
      .AddInput("ResidualData")
      .IsOptional()
      .End()
      .AddOutput("Output")
      .IsTensor()
      .End()
      .AddAttr("strides")
      .End()
      .AddAttr("paddings")
      .End()
      .AddAttr("padding_algorithm")
      .IsOptional()
      .IsStringIn({"EXPLICIT", "SAME", "VALID"})
      .End()
      .AddAttr("groups")
      .IsNumGE(1)
      .End()
      .AddAttr("dilations")
      .End()
      .AddAttr("data_format")
      .IsStringIn({"NCHW", "NHWC", "AnyLayout"})
      .End();

  AddOpCompat(OpCompat("elementwise_add"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Y")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("axis")
      .IsNumEQ(1)
      .End();

  AddOpCompat(OpCompat("relu"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End();

  AddOpCompat(OpCompat("sigmoid"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End();

  AddOpCompat(OpCompat("tanh"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End();

  AddOpCompat(OpCompat("swish"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End();

  AddOpCompat(OpCompat("leaky_relu"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddAttr("alpha")
      .IsType<float>()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End();
}

void ConvElementwiseAddActFusePass::ApplyImpl(ir::Graph* graph) const {
  const std::string pattern_name = "conv_elementwise_add_act_fuse";
  FusePassBase::Init(pattern_name, graph);

  GraphPatternDetector gpd;
  auto* x = gpd.mutable_pattern()
                ->NewNode("x")
                ->assert_is_op_input("conv2d", "Input")
                ->AsInput();

#if CUDNN_VERSION >= 8000
  std::unordered_set<std::string> cudnn_act_set(
      {"identity", "relu", "sigmoid", "tanh"});
#else
  std::unordered_set<std::string> cudnn_act_set({"identity", "relu"});
#endif

  std::unordered_set<std::string> cutlass_act_set;
  std::unordered_set<std::string> all_act_set = cudnn_act_set;

  bool is_fp16_precision =
      static_cast<phi::DataType>(Get<int>("model_precision")) ==
          phi::DataType::FLOAT16 ||
      Get<bool>("enable_gpu_mixed");
  constexpr int CUTLASS_NHWC_ALIGNMENT = 8;
  if (is_fp16_precision) {
#ifdef PADDLE_WITH_CUTLASS
    // cutlass now support these activations
    // cutlass_act_set.insert("swish");
    // cutlass_act_set.insert("relu");
    // cutlass_act_set.insert("identity");
    // cutlass_act_set.insert("leaky_relu");

    all_act_set.insert(cutlass_act_set.begin(), cutlass_act_set.end());
#endif
  }

  patterns::ConvElementwiseaddAct pattern(gpd.mutable_pattern(), pattern_name);
  pattern(x, all_act_set);

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    if (!IsCompat(subgraph, g)) {
      LOG(WARNING) << "Pass in op compat failed.";
      return;
    }
    GET_NODES;

    auto base_op_desc = *conv_op->Op()->Proto();
    std::string bias_name = elementwise_add_in_y->Name();
    std::string act_op_type = act_op->Op()->Type();
    std::string act_op_out = act_out->Name();
    auto* scope = param_scope();
    auto* filter_var = scope->FindLocalVar(conv_filter->Name());
    auto* filter_tensor = filter_var->GetMutable<phi::DenseTensor>();
    CHECK_EQ(filter_tensor->dims().size() == 4UL, true);
    // when this conv2d_fusion problem size is not supported by cutlass and not
    // supported by cuDNN, we should not apply this pass
    int oc = filter_tensor->dims()[0];
    int ic = filter_tensor->dims()[1];
    bool cutlass_can_fuse = oc % CUTLASS_NHWC_ALIGNMENT == 0 &&
                            ic % CUTLASS_NHWC_ALIGNMENT == 0 &&
                            cutlass_act_set.count(act_op_type);
    bool cudnn_can_fuse = cudnn_act_set.count(act_op_type);
    if (!cutlass_can_fuse && !cudnn_can_fuse) {
      return;
    }

    float alpha = 0.f;
    alpha = act_op->Op()->GetAttrIfExists<float>("alpha");

    auto new_op_proto =
        PrepareOpDesc(base_op_desc, bias_name, act_op_type, act_op_out, alpha);
    framework::OpDesc new_op_desc(new_op_proto, nullptr);

    // Create a new node for the fused op.
    auto* new_conv_op = graph->CreateOpNode(&new_op_desc);

    // Link inputs and outputs.
    PADDLE_ENFORCE_NE(
        subgraph.count(x),
        0,
        platform::errors::NotFound("Detector did not find input x of conv2d."));
    auto* conv_in_node = subgraph.at(x);

    IR_NODE_LINK_TO(conv_in_node, new_conv_op);          // Input
    IR_NODE_LINK_TO(conv_filter, new_conv_op);           // Filter
    IR_NODE_LINK_TO(elementwise_add_in_y, new_conv_op);  // Bias
    IR_NODE_LINK_TO(new_conv_op, act_out);               // Output

    // Delete the unneeded nodes.
    GraphSafeRemoveNodes(
        graph,
        {conv_op, conv_out, elementwise_add_op, elementwise_add_out, act_op});
  };

  gpd(graph, handler);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(conv_elementwise_add_act_fuse_pass,
              paddle::framework::ir::ConvElementwiseAddActFusePass);
REGISTER_PASS_CAPABILITY(conv_elementwise_add_act_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("conv2d", 1)
            .LE("elementwise_add", 1)
            .EQ("relu", 0)
            .EQ("sigmoid", 0)
            .EQ("tanh", 0)
            .EQ("identity", 0)
            .LE("leaky_relu", 1)
            .EQ("swish", 0));
