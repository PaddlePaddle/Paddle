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

#include "paddle/fluid/framework/ir/bev_cross_attention_fuse_pass.h"

#include <string>
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_version_registry.h"
#ifdef PADDLE_WITH_TENSORRT
#include "paddle/fluid/inference/tensorrt/helper.h"
#endif
#include "paddle/phi/kernels/funcs/blas/blas.h"

#define GET_IR_NODE(node__) GET_IR_NODE_FROM_SUBGRAPH(node__, node__, pattern);
#define GET_NODES               \
  GET_IR_NODE(reshapeA1_op);    \
  GET_IR_NODE(reshapeA1_out);   \
  GET_IR_NODE(transposeA1_op);  \
  GET_IR_NODE(transposeA1_out); \
  GET_IR_NODE(scaleA1_op);      \
  GET_IR_NODE(scaleA1_out);     \
  GET_IR_NODE(matmulA1_op);     \
  GET_IR_NODE(matmulA1_out);    \
  GET_IR_NODE(softmaxA1_op);    \
  GET_IR_NODE(softmaxA1_out);   \
  GET_IR_NODE(matmulA2_op);     \
  GET_IR_NODE(matmulA2_out);    \
  GET_IR_NODE(transposeA2_op);  \
  GET_IR_NODE(transposeA2_out); \
  GET_IR_NODE(reshapeA2_op);    \
  GET_IR_NODE(reshapeA2_out);   \
  GET_IR_NODE(reshapeB1_op);    \
  GET_IR_NODE(reshapeB1_out);   \
  GET_IR_NODE(transposeB1_op);  \
  GET_IR_NODE(transposeB1_out); \
  GET_IR_NODE(reshapeC1_op);    \
  GET_IR_NODE(reshapeC1_out);   \
  GET_IR_NODE(transposeC1_op);  \
  GET_IR_NODE(transposeC1_out);

// fuse struct
//     in_q   in_k    in_v
//       |      |       |
//       |      |       |
//    reshape reshape reshape
//       |      |       |
//     trans   trans   trans
//       |      |       |
//        matmul        |
//          |           |
//        scale         |
//          |           |
//        softmax       |
//          |------matmul
//                    |
//                  trans
//                    |
//                  reshape
//                    |
//                   output
//
// -> fused to
//
//      in_q,in_k,in_v
//            |
//    flash_multihead_matmul
//            |
//          output

namespace paddle {
namespace framework {
namespace ir {

void BevCrossAttentionFusePass::ApplyImpl(ir::Graph* graph) const {
  GraphPatternDetector gpd;
  const std::string pattern_name = "bev_cross_attention_fuse";
  FusePassBase::Init(pattern_name, graph);
  auto* scope = param_scope();

#ifdef PADDLE_WITH_TENSORRT
  auto trt_version = paddle::inference::tensorrt::GetTrtRuntimeVersion();
  if (std::get<0>(trt_version) * 1000 + std::get<1>(trt_version) * 100 +
          std::get<2>(trt_version) * 10 <
      8520) {
    VLOG(3) << "Flash attention oss plugin only available for trt version >= "
               "8.5.2.2. Stop this pass";
    return;
  }
#else
  return;
#endif

  // pattern
  std::unordered_set<std::string> matmul_ops{"matmul", "matmul_v2"};

  PDNode* q = gpd.mutable_pattern()
                  ->NewNode("q")
                  ->assert_is_op_input("reshape2", "X")
                  ->AsInput();

  PDNode* k = gpd.mutable_pattern()
                  ->NewNode("k")
                  ->assert_is_op_input("reshape2", "X")
                  ->AsInput();

  PDNode* v = gpd.mutable_pattern()
                  ->NewNode("v")
                  ->assert_is_op_input("reshape2", "X")
                  ->AsInput();

  patterns::BevCrossAttention pattern(gpd.mutable_pattern(), pattern_name);
  pattern(q, k, v);

  int fusion_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_NODES;
    // new desc;
    OpDesc desc(reshapeA1_op->Op()->Block());
    desc.SetType("bev_cross_multihead_matmul");
    // input,output
    desc.SetInput("InputQ", {subgraph.at(q)->Name()});
    desc.SetInput("InputK", {subgraph.at(k)->Name()});
    desc.SetInput("InputV", {subgraph.at(v)->Name()});
    std::vector<int64_t> shape = softmaxA1_out->Var()->GetShape();
    desc.SetOutput("Out", {reshapeA2_out->Name()});
    // attr
    desc.SetAttr("head_number", static_cast<int>(shape[1]));
    desc.SetAttr("q_length", static_cast<int>(shape[2]));
    desc.SetAttr("kv_length", static_cast<int>(shape[3]));
    float alpha = PADDLE_GET_CONST(float, scaleA1_op->Op()->GetAttr("scale"));
    desc.SetAttr("alpha", alpha);

    // Create a new node for the fused op.
    auto bev_cross_attention_node = graph->CreateOpNode(&desc);

    // Link inputs and outputs.
    IR_NODE_LINK_TO(subgraph.at(q), bev_cross_attention_node);  // Input_q
    IR_NODE_LINK_TO(subgraph.at(k), bev_cross_attention_node);  // Input_k
    IR_NODE_LINK_TO(subgraph.at(v), bev_cross_attention_node);  // Input_v
    IR_NODE_LINK_TO(bev_cross_attention_node, reshapeA2_out);   // Output

    // Delete the unneeded nodes.
    std::unordered_set<const Node*> marked_nodes(
        {reshapeA1_op,   reshapeA1_out,   transposeA1_op,  transposeA1_out,
         scaleA1_op,     scaleA1_out,     matmulA1_op,     matmulA1_out,
         softmaxA1_op,   softmaxA1_out,   matmulA2_op,     matmulA2_out,
         transposeA2_op, transposeA2_out, reshapeA2_op,    reshapeB1_op,
         reshapeB1_out,  transposeB1_op,  transposeB1_out, reshapeC1_op,
         reshapeC1_out,  transposeC1_op,  transposeC1_out});

    GraphSafeRemoveNodes(graph, marked_nodes);
    ++fusion_count;
  };
  gpd(graph, handler);
  AddStatis(fusion_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(bev_cross_attention_fuse_pass,
              paddle::framework::ir::BevCrossAttentionFusePass);
REGISTER_PASS_CAPABILITY(bev_cross_attention_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .EQ("reshape2", 0)
            .EQ("transpose2", 0)
            .EQ("scale", 0)
            .EQ("softmax", 0)
            .EQ("matmul_v2", 0));
