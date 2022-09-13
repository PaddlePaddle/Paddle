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

#include "paddle/fluid/framework/ir/swin_attention1_fuse_pass.h"

#include <string>
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"

#define GET_IR_NODE(node__) GET_IR_NODE_FROM_SUBGRAPH(node__, node__, pattern);
#define GET_NODES                   \
  GET_IR_NODE(transpose_i00_op)     \
  GET_IR_NODE(transpose_i00_out)    \
  GET_IR_NODE(reshape_i10_op)       \
  GET_IR_NODE(reshape_i10_out)      \
  GET_IR_NODE(reshape_i20_op)       \
  GET_IR_NODE(reshape_i20_out)      \
  GET_IR_NODE(matmul_00_op);        \
  GET_IR_NODE(matmul_00_in_y);      \
  GET_IR_NODE(matmul_00_out);       \
  GET_IR_NODE(elementwise_10_op);   \
  GET_IR_NODE(elementwise_10_in_y); \
  GET_IR_NODE(elementwise_10_out);  \
  GET_IR_NODE(reshape_20_op);       \
  GET_IR_NODE(reshape_20_out);      \
  GET_IR_NODE(transpose_30_op);     \
  GET_IR_NODE(transpose_30_out);    \
  GET_IR_NODE(slice_40_op);         \
  GET_IR_NODE(slice_40_out);        \
  GET_IR_NODE(slice_41_op);         \
  GET_IR_NODE(slice_41_out);        \
  GET_IR_NODE(slice_42_op);         \
  GET_IR_NODE(slice_42_out);        \
  GET_IR_NODE(scale_50_op);         \
  GET_IR_NODE(scale_50_out);        \
  GET_IR_NODE(transpose_51_op);     \
  GET_IR_NODE(transpose_51_out);    \
  GET_IR_NODE(matmul_60_op);        \
  GET_IR_NODE(matmul_60_out);       \
  GET_IR_NODE(elementwise_70_op);   \
  GET_IR_NODE(elementwise_70_in_y); \
  GET_IR_NODE(elementwise_70_out);  \
  GET_IR_NODE(softmax_80_op);       \
  GET_IR_NODE(softmax_80_out);      \
  GET_IR_NODE(matmul_90_op);        \
  GET_IR_NODE(matmul_90_out);       \
  GET_IR_NODE(transpose_a0_op);     \
  GET_IR_NODE(transpose_a0_out);    \
  GET_IR_NODE(reshape_b0_op);       \
  GET_IR_NODE(reshape_b0_out);

namespace paddle {
namespace framework {
namespace ir {
// a example for swin attention fuse pass
//                  input(x)                                      input(x)
//                    | ?x8x7x8x96                                  | ?x8x7x8x96
//                transpose2                                      transpose2
//                    | ?x8x8x7x7x96                                | ?x8x8x7x7x96
//                 reshape2                                       reshape2
//                    | ?x7x7x96                                    | ?x7x7x96
//                 reshape2            W                          reshape2
//                    | ?x49x96        | 96x288                     | ?x49x96        W            Bias
//                 matmul_v2 ----------|        Bias      fuse      |                | 96x288      | 288
//                    | ?x49x288                 | 288     ->     multihead_matmul---|-------------|
//                 elementwise_add --------------|                  |                |1x3x49x49    | 64x49x49
//                    | ?x49x288                                  output            BiasQK        [BiasQK_mask](optional)
//                 reshape2
//                    | ?x49x3x3x32
//                 transpose2
//                    |
//    |---------------|---------------|
//    | 3x?x3x49x32   | 3x?x3x49x32   | 3x?x3x49x32
//   slice           slice           slice
//    | ?x3x49x32     | ?x3x49x32     | ?x3x49x32
//    |              scale           transpose2
//    |               | ?x3x49x32     | ?x3x32x49
//    |                \             /
//    |                 |-matmul_v2-|        BiasQK       [BiasQK_mask](optional) 
//    |                      | ?x3x49x49     | 1x3x49x49  | 64x49x49 
//    |                   elementwise_add----|------------| 
//    |                       | ?x3x49x49 
//    |                    softmax 
//    |                       | ?x3x49x49
//     \                     /
//      |----matmul_v2------|
//               | ?x3x49x32
//           transpose2
//               | ?x49x3x32
//           reshape2
//               | ?x49x96
//             output

SwinAttention1FusePass::SwinAttention1FusePass() {
  // (B,sqrt(W),sqrt(S),sqrt(W), sqrt(S), N*H) -> (B,sqrt(W),sqrt(W), sqrt(S),
  // sqrt(S), N*H) (B*W, S, 3, N*H) -> (3, B*W, H, S, N) (B*W, H, S, N) -> (B*W,
  // H, N, S) (B*W, H, S, N) -> (B*W, S, H, N)
  AddOpCompat(OpCompat("transpose2"))
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
      .AddAttr("axis")
      .IsType<std::vector<int>>()
      .End();

  AddOpCompat(OpCompat("reshape2"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Shape")
      .IsTensor()
      .IsOptional()
      .End()
      .AddInput("ShapeTensor")
      .IsTensor()
      .IsOptional()
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

  AddOpCompat(OpCompat("elementwise_add"))
      .AddInput("X")
      // in bias, shape is (B, S, N*H),
      // in biasqk, shape is (B, H, S, S)
      .IsTensor()
      .End()
      .AddInput("Y")
      // in bias, shape is (N*H)
      // in biasqk, shape is (B, H, S, S)
      .IsTensor()
      .End()
      .AddInput("BiasQK_mask")
      .IsTensor()
      .IsOptional()
      .End()
      // in bias, shape is (B, S, N*H)
      // in biasqk, shape is (B, H, S, S)
      .AddOutput("Out")
      .IsTensor()
      .End()
      // in bias, it equal to 2
      // in biasqk, it equal to -1 or 0
      .AddAttr("axis")
      .IsIntIn({2, -1, 0})
      .End();

  AddOpCompat(OpCompat("matmul_v2"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Y")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("trans_x")
      .IsBoolEQ(false)
      .End()
      .AddAttr("trans_y")
      .IsType<bool>()
      .End();

  AddOpCompat(OpCompat("softmax"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("axis")
      .IsIntIn({-1, 3})  // shape is (B, H, S, S), so axis is -1 or 3
      .End();

  AddOpCompat(OpCompat("scale"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("scale")
      .IsType<float>()
      .End()
      .AddAttr("bias")
      .IsNumEQ(0.f)
      .End()
      .AddAttr("bias_after_scale")
      .IsType<bool>()
      .End();
}

void SwinAttention1FusePass::ApplyImpl(ir::Graph* graph) const {
  GraphPatternDetector gpd;
  const std::string pattern_name = "swin_attention1_fuse";
  FusePassBase::Init(pattern_name, graph);
  auto* scope = param_scope();

  // std::unordered_set<std::string> matmul_ops{"matmul", "matmul_v2"};
  PDNode* x = gpd.mutable_pattern()
                  ->NewNode("x")
                  ->assert_is_op_input("transpose2", "X")
                  ->AsInput();
  patterns::SwinAttention1Fuse pattern(gpd.mutable_pattern(), pattern_name);
  pattern(x);

  int fusion_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    if (!IsCompat(subgraph, g)) {
      LOG(WARNING) << "Pass in op compat failed.";
      return;
    }
    GET_NODES;
    // configure new op node
    OpDesc desc(matmul_00_op->Op()->Block());
    desc.SetType("multihead_matmul");
    desc.SetInput("Input", {reshape_i20_out->Name()});
    // get window num here for swin's window attention
    std::vector<int64_t> window_num_tranpose_out_shape =
        transpose_i00_out->Var()->GetShape();
    int window_num =
        window_num_tranpose_out_shape[1] * window_num_tranpose_out_shape[2];
    VLOG(3) << "swin attention fused with window num: " << window_num;
    auto* weight_qkv_tensor =
        scope->FindVar(matmul_00_in_y->Name())->GetMutable<LoDTensor>();
    auto weight_qkv_dims = phi::make_ddim(
        {weight_qkv_tensor->dims()[0], 3, weight_qkv_tensor->dims()[1] / 3});
    weight_qkv_tensor->Resize(weight_qkv_dims);

    auto* bias_qkv_tensor =
        scope->FindVar(elementwise_10_in_y->Name())->GetMutable<LoDTensor>();
    auto bias_qkv_dims = phi::make_ddim({3, bias_qkv_tensor->dims()[0] / 3});
    bias_qkv_tensor->Resize(bias_qkv_dims);

    std::vector<int64_t> softmax_shape = softmax_80_out->Var()->GetShape();
    float alpha = PADDLE_GET_CONST(float, scale_50_op->Op()->GetAttr("scale"));
    auto qkbias_add_inputs = elementwise_70_op->Op()->Inputs(false);
    bool has_biasQK_mask = false;
    std::string biasQK_mask_name;
    for (auto input : qkbias_add_inputs) {
      if (input.first == "BiasQK_mask") {
        has_biasQK_mask = true;
        biasQK_mask_name = input.second[0].c_str();
        break;
      }
    }
    desc.SetInput("W", {matmul_00_in_y->Name()});
    desc.SetInput("Bias", {elementwise_10_in_y->Name()});
    desc.SetInput("BiasQK", {elementwise_70_in_y->Name()});
    desc.SetAttr("window_number", window_num);
    Node* biaskQK_mask_node = nullptr;
    if (has_biasQK_mask) {
      desc.SetInput("BiasQK_mask", {biasQK_mask_name});
      for (auto inputNode : elementwise_70_op->inputs) {
        if (inputNode->Name() == biasQK_mask_name) {
          biaskQK_mask_node = inputNode;
          break;
        }
      }
    }
    desc.SetOutput("Out", {reshape_b0_out->Name()});

    desc.SetAttr("head_number", static_cast<int>(softmax_shape[1]));
    desc.SetAttr("alpha", alpha);
    desc.SetAttr("BiasQK_directInput", true);

    // create a new node for the fused op
    auto swin_attention1_node = graph->CreateOpNode(&desc);

    PADDLE_ENFORCE_NE(
        subgraph.count(x),
        0,
        platform::errors::NotFound(
            "Detector did not find input x of tranpose2 for swin attention."));

    // link inputs and oupts to the new fused op node
    IR_NODE_LINK_TO(reshape_i20_out,
                    swin_attention1_node);  // input x of matmul/matmul_v2
    IR_NODE_LINK_TO(matmul_00_in_y, swin_attention1_node);       // weight
    IR_NODE_LINK_TO(elementwise_10_in_y, swin_attention1_node);  // Bias
    IR_NODE_LINK_TO(elementwise_70_in_y, swin_attention1_node);  // BiasQK
    if (has_biasQK_mask) {
      IR_NODE_LINK_TO(biaskQK_mask_node, swin_attention1_node);
    }
    IR_NODE_LINK_TO(swin_attention1_node, reshape_b0_out);

    // remove the origin nodes
    std::unordered_set<const Node*> marked_nodes(
        {matmul_00_op,       matmul_00_out,    elementwise_10_op,
         elementwise_10_out, reshape_20_op,    reshape_20_out,
         transpose_30_op,    transpose_30_out, slice_40_op,
         slice_40_out,       slice_41_op,      slice_41_out,
         slice_42_op,        slice_42_out,     scale_50_op,
         scale_50_out,       transpose_51_op,  transpose_51_out,
         matmul_60_op,       matmul_60_out,    elementwise_70_op,
         elementwise_70_out, softmax_80_op,    softmax_80_out,
         matmul_90_op,       matmul_90_out,    transpose_a0_op,
         transpose_a0_out,   reshape_b0_op});

    GraphSafeRemoveNodes(graph, marked_nodes);
    ++fusion_count;
  };
  gpd(graph, handler);
  AddStatis(fusion_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(swin_attention1_fuse_pass,
              paddle::framework::ir::SwinAttention1FusePass);

REGISTER_PASS_CAPABILITY(swin_attention1_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .EQ("reshape2", 0)
            .EQ("transpose2", 0)
            .EQ("slice", 0)
            .EQ("scale", 0)
            .EQ("softmax", 0)
            .EQ("matmul_v2", 0)
            .LE("elementwise_add", 1));
