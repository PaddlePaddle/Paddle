// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/merge_layernorm_fuse_pass.h"

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_version_registry.h"

#define GET_IR_NODE(node__) GET_IR_NODE_FROM_SUBGRAPH(node__, node__, pattern);
#define GET_NODES                     \
  GET_IR_NODE(reshape2_00_op);        \
  GET_IR_NODE(reshape2_00_out);       \
  GET_IR_NODE(strided_slice_10_op);   \
  GET_IR_NODE(strided_slice_10_out);  \
  GET_IR_NODE(strided_slice_11_op);   \
  GET_IR_NODE(strided_slice_11_out);  \
  GET_IR_NODE(strided_slice_12_op);   \
  GET_IR_NODE(strided_slice_12_out);  \
  GET_IR_NODE(strided_slice_13_op);   \
  GET_IR_NODE(strided_slice_13_out);  \
  GET_IR_NODE(concat_20_op);          \
  GET_IR_NODE(concat_20_out);         \
  GET_IR_NODE(reshape2_30_op);        \
  GET_IR_NODE(reshape2_30_out);       \
  GET_IR_NODE(layernorm_40_op);       \
  GET_IR_NODE(layernorm_40_in_bias);  \
  GET_IR_NODE(layernorm_40_in_scale); \
  GET_IR_NODE(layernorm_40_out);
namespace paddle::framework::ir {
MergeLayernormFusePass::MergeLayernormFusePass() {
  AddOpCompat(OpCompat("reshape2"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddOutput("XShape")
      .IsOptional()
      .IsTensor()
      .End()
      .AddAttr("shape")
      .IsType<std::vector<int>>()
      .End();
  AddOpCompat(OpCompat("strided_slice"))
      .AddInput("Input")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("axes")
      .IsType<std::vector<int>>()
      .End()
      .AddAttr("starts")
      .IsType<std::vector<int>>()
      .End()
      .AddAttr("strides")
      .IsType<std::vector<int>>()
      .End()
      .AddAttr("infer_flags")
      .IsType<std::vector<int>>()
      .End()
      .AddAttr("ends")
      .IsType<std::vector<int>>()
      .End();
  AddOpCompat(OpCompat("concat"))
      .AddInput("X")
      .End()
      .AddInput("AxisTensor")
      .IsTensor()
      .IsOptional()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("axis")
      .End();
  AddOpCompat(OpCompat("layer_norm"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Scale")
      .IsTensor()
      .End()
      .AddInput("Bias")
      .IsTensor()
      .End()
      .AddOutput("Y")
      .IsTensor()
      .End()
      .AddOutput("Mean")
      .IsTensor()
      .IsOptional()
      .End()
      .AddOutput("Variance")
      .IsTensor()
      .IsOptional()
      .End()
      .AddAttr("epsilon")
      .IsNumGE(0.0f)
      .IsNumLE(0.001f)
      .End()
      .AddAttr("begin_norm_axis")
      .IsNumEQ(2)
      .End();
}
void MergeLayernormFusePass::ApplyImpl(ir::Graph* graph) const {
  GraphPatternDetector gpd;
  const std::string pattern_name = "merge_layernorm";
  FusePassBase::Init(pattern_name, graph);
  // auto* scope = param_scope();

  PDNode* x = gpd.mutable_pattern()
                  ->NewNode("x")
                  ->assert_is_op_input("reshape2", "X")
                  ->AsInput();
  patterns::MergeLayernormPattern pattern(gpd.mutable_pattern(), pattern_name);
  pattern(x);
  int fusion_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    if (!IsCompat(subgraph, g)) {
      LOG(WARNING) << "Pass in op compat failed.";
      return;
    }
    GET_NODES;
    OpDesc merge_layer_op_desc(reshape2_00_op->Op()->Block());
    merge_layer_op_desc.SetType("merge_layernorm");
    merge_layer_op_desc.SetInput("X", {subgraph.at(x)->Name()});
    merge_layer_op_desc.SetInput("Bias", {layernorm_40_in_bias->Name()});
    merge_layer_op_desc.SetInput("Scale", {layernorm_40_in_scale->Name()});
    merge_layer_op_desc.SetOutput("Y", {layernorm_40_out->Name()});
    merge_layer_op_desc.SetAttr(
        "begin_norm_axis", layernorm_40_op->Op()->GetAttr("begin_norm_axis"));
    merge_layer_op_desc.SetAttr("epsilon",
                                layernorm_40_op->Op()->GetAttr("epsilon"));
    auto* merge_layer_op_node = graph->CreateOpNode(&merge_layer_op_desc);
    IR_NODE_LINK_TO(subgraph.at(x), merge_layer_op_node);
    IR_NODE_LINK_TO(layernorm_40_in_bias, merge_layer_op_node);
    IR_NODE_LINK_TO(layernorm_40_in_scale, merge_layer_op_node);
    IR_NODE_LINK_TO(merge_layer_op_node, layernorm_40_out);
    GraphSafeRemoveNodes(graph,
                         {reshape2_00_op,
                          reshape2_00_out,
                          strided_slice_10_op,
                          strided_slice_10_out,
                          strided_slice_11_op,
                          strided_slice_11_out,
                          strided_slice_12_op,
                          strided_slice_12_out,
                          strided_slice_13_op,
                          strided_slice_13_out,
                          concat_20_op,
                          concat_20_out,
                          reshape2_30_op,
                          reshape2_30_out,
                          layernorm_40_op});
    ++fusion_count;
  };
  gpd(graph, handler);
  AddStatis(fusion_count);
}
}  // namespace paddle::framework::ir
REGISTER_PASS(merge_layernorm_fuse_pass,
              paddle::framework::ir::MergeLayernormFusePass);
REGISTER_PASS_CAPABILITY(merge_layernorm_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .EQ("reshape2", 0)
            .EQ("concat", 0)
            .EQ("layer_norm", 0));
