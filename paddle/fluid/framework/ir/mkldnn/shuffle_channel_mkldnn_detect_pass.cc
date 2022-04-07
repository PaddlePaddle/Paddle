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

#include "paddle/fluid/framework/ir/mkldnn/shuffle_channel_mkldnn_detect_pass.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace framework {
namespace ir {

#define GET_IR_NODE(node__) GET_IR_NODE_FROM_SUBGRAPH(node__, node__, pattern);
#define GET_NODES             \
  GET_IR_NODE(reshape1_op);   \
  GET_IR_NODE(reshape1_out);  \
  GET_IR_NODE(transpose_op);  \
  GET_IR_NODE(transpose_out); \
  GET_IR_NODE(reshape2_op);   \
  GET_IR_NODE(reshape2_out);

ShuffleChannelMKLDNNDetectPass::ShuffleChannelMKLDNNDetectPass() {
  AddOpCompat(OpCompat("reshape2"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Shape")
      .IsOptional()
      .IsTensor()
      .End()
      .AddInput("ShapeTensor")
      .IsOptional()
      .IsTensor()
      .End()
      .AddOutput("XShape")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("shape")
      .IsType<std::vector<int>>()
      .End();

  AddOpCompat(OpCompat("transpose2"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("XShape")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("axis")
      .IsType<std::vector<int>>()
      .End();
}

void ShuffleChannelMKLDNNDetectPass::ApplyImpl(ir::Graph* graph) const {
  const std::string pattern_name = "shufflechannel_pattern";
  FusePassBase::Init(pattern_name, graph);

  GraphPatternDetector gpd;
  auto* x = gpd.mutable_pattern()
                ->NewNode("x")
                ->assert_is_op_input("reshape2", "X")
                ->AsInput();

  patterns::ShuffleChannelPattern pattern(gpd.mutable_pattern(), pattern_name);
  pattern(x);

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_NODES;
    if (!IsCompat(subgraph, g)) {
      LOG(WARNING) << "The Pass in op compat failed.";
      return;
    }
    PADDLE_ENFORCE_GT(
        subgraph.count(x), 0,
        platform::errors::NotFound("Detector did not find input X."));
    auto* input_node = subgraph.at(x);
    auto reshape1_desc = reshape1_op->Op();
    auto reshape2_desc = reshape2_op->Op();
    auto trans_desc = transpose_op->Op();
    std::string input_name = input_node->Name();
    std::string output_name = reshape2_out->Name();

    auto reshape1_shape =
        BOOST_GET_CONST(std::vector<int>, reshape1_desc->GetAttr("shape"));
    auto reshape2_shape =
        BOOST_GET_CONST(std::vector<int>, reshape2_desc->GetAttr("shape"));
    auto trans_axis =
        BOOST_GET_CONST(std::vector<int>, trans_desc->GetAttr("axis"));
    auto* block1 = reshape1_desc->Block();
    auto* block2 = reshape2_desc->Block();
    if (block1 && block2) {
      auto x_var_name = reshape1_desc->Input("X")[0];
      auto* x_var_desc = block1->FindVar(x_var_name);
      auto x_shape1 = x_var_desc->GetShape();
      x_var_name = reshape2_desc->Input("X")[0];
      x_var_desc = block2->FindVar(x_var_name);
      auto x_shape2 = x_var_desc->GetShape();
      // now shuffle_channel is 4D(NCHW) only.
      if (x_shape1.size() != 4 || reshape1_shape.size() != 5 ||
          reshape2_shape.size() != 4 || trans_axis.size() != 5) {
        return;
      }

      // process 0 and -1 in reshape.
      constexpr int64_t copy_dim_val = 0;
      for (size_t i = 0; i < reshape1_shape.size(); i++) {
        if (reshape1_shape[i] == copy_dim_val) {
          reshape1_shape[i] = x_shape1[i];
        }
      }
      for (size_t i = 0; i < reshape2_shape.size(); i++) {
        if (reshape2_shape[i] == copy_dim_val) {
          reshape2_shape[i] = x_shape2[i];
        }
      }
      constexpr int64_t unk_dim_idx = -1;
      bool all_positive = std::all_of(x_shape1.cbegin(), x_shape1.cend(),
                                      [](int64_t i) { return i > 0; });
      for (size_t i = 0; i < reshape1_shape.size(); ++i) {
        // if -1 is not in batch dim, try to calculate number
        if ((reshape1_shape[i] == unk_dim_idx) && (i != 0)) {
          // there is no sufficient info
          if (!all_positive) return;
          reshape1_shape[i] =
              std::accumulate(x_shape1.begin(), x_shape1.end(),
                              static_cast<int64_t>(1),
                              std::multiplies<int64_t>()) /
              std::accumulate(reshape1_shape.begin(), reshape1_shape.end(),
                              static_cast<int64_t>(-1),
                              std::multiplies<int64_t>());
          break;
        }
      }

      all_positive = std::all_of(x_shape2.cbegin(), x_shape2.cend(),
                                 [](int64_t i) { return i > 0; });
      for (size_t i = 0; i < reshape2_shape.size(); ++i) {
        // if -1 is not in batch dim, try to calculate number
        if ((reshape2_shape[i] == unk_dim_idx) && (i != 0)) {
          // there is no sufficient info
          if (!all_positive) return;
          reshape2_shape[i] =
              std::accumulate(x_shape2.begin(), x_shape2.end(),
                              static_cast<int64_t>(1),
                              std::multiplies<int64_t>()) /
              std::accumulate(reshape2_shape.begin(), reshape2_shape.end(),
                              static_cast<int64_t>(-1),
                              std::multiplies<int64_t>());
          break;
        }
      }

      // shuffle_channel dosen't change shape
      if ((reshape2_shape[0] != -1) && (x_shape1[0] != reshape2_shape[0])) {
        return;
      }
      for (size_t i = 1; i < x_shape1.size(); i++) {
        if (x_shape1[i] != reshape2_shape[i]) {
          return;
        }
      }
      if ((reshape2_shape[3] != reshape1_shape[4]) ||
          (reshape2_shape[2] != reshape1_shape[3])) {
        return;
      }
    } else {
      return;  // conservative judgement
    }

    int i_c = reshape1_shape[2];
    int o_c = reshape2_shape[1];
    int group = o_c / i_c;
    // should split on channel dim
    if (reshape2_shape[1] != reshape1_shape[2] * reshape1_shape[1]) return;
    // trans on channel dim
    if (trans_axis[0] != 0 || trans_axis[3] != 3 || trans_axis[4] != 4) return;
    if (group != 1 && i_c != 1) {
      if (trans_axis[1] != 2 && trans_axis[2] != 1) {
        return;
      }
    }

    framework::OpDesc new_op_desc;
    new_op_desc.SetType("shuffle_channel");
    new_op_desc.SetInput("X", {input_name});
    new_op_desc.SetOutput("Out", {output_name});

    new_op_desc.SetAttr("group", group);
    new_op_desc.SetAttr("use_mkldnn", true);
    new_op_desc.Flush();

    // Create a new node for the fused op.
    auto* new_op = graph->CreateOpNode(&new_op_desc);

    IR_NODE_LINK_TO(input_node, new_op);
    IR_NODE_LINK_TO(new_op, reshape2_out);

    // Delete the unneeded nodes.
    GraphSafeRemoveNodes(graph, {reshape1_op, reshape1_out, transpose_op,
                                 transpose_out, reshape2_op});
    LOG_FIRST_N(WARNING, 1)
        << "There is fluid.layers.shuffle_channel API already, maybe you can "
           "use it instead of (reshape + transpose + reshape)";
  };

  gpd(graph, handler);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(shuffle_channel_mkldnn_detect_pass,
              paddle::framework::ir::ShuffleChannelMKLDNNDetectPass);
REGISTER_PASS_CAPABILITY(shuffle_channel_mkldnn_detect_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .EQ("reshape2", 0)
            .EQ("transpose2", 0));
