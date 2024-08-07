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

#include "paddle/fluid/framework/ir/onednn/self_attention_fuse_pass.h"

#include <string>
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/phi/backends/cpu/cpu_info.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/utils/string/pretty_log.h"

#define GET_IR_NODE(node__) GET_IR_NODE_FROM_SUBGRAPH(node__, node__, pattern);
#define GET_NODES                \
  GET_IR_NODE(transpose2_0_op);  \
  GET_IR_NODE(transpose2_0_out); \
  GET_IR_NODE(slice_0_op);       \
  GET_IR_NODE(slice_0_out);      \
  GET_IR_NODE(slice_1_op);       \
  GET_IR_NODE(slice_1_out);      \
  GET_IR_NODE(slice_2_op);       \
  GET_IR_NODE(slice_2_out);      \
  GET_IR_NODE(matmul_0_op);      \
  GET_IR_NODE(matmul_0_out);     \
  GET_IR_NODE(matmul_1_op);      \
  GET_IR_NODE(matmul_1_out);     \
  GET_IR_NODE(transpose2_1_op);  \
  GET_IR_NODE(transpose2_1_out); \
  GET_IR_NODE(softmax_op);       \
  GET_IR_NODE(softmax_out);      \
  GET_IR_NODE(transpose2_2_op);  \
  GET_IR_NODE(transpose2_2_out);

namespace paddle::framework::ir {

using string::PrettyLogDetail;

void SelfAttentionFusePass::ApplyImpl(ir::Graph* graph) const {
#if !defined(__AVX512F__) || !defined(PADDLE_WITH_MKLML) || \
    !defined(PADDLE_WITH_DNNL)
  LOG(WARNING) << "No-avx512 or MKL supported!";
  return;
#endif

  if (!phi::backends::cpu::MayIUse(phi::backends::cpu::cpu_isa_t::avx512f)) {
    return;
  }

  // do something;
  GraphPatternDetector gpd;
  const std::string pattern_name = "self_attention_fuse";
  FusePassBase::Init(pattern_name, graph);

  // pattern
  PDNode* x = gpd.mutable_pattern()
                  ->NewNode("x")
                  ->assert_is_op_input("transpose2", "X")
                  ->AsInput();
  patterns::SelfAttention pattern(gpd.mutable_pattern(), pattern_name);
  pattern(x);

  int fusion_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_NODES;
    // do something;
    OpDesc desc(transpose2_0_op->Op()->Block());
    desc.SetType("self_dp_attention");
    desc.SetInput("X", {subgraph.at(x)->Name()});
    desc.SetOutput("Out", {transpose2_2_out->Name()});

    std::vector<int64_t> in_shape = subgraph.at(x)->Var()->GetShape();
    std::vector<int64_t> shape = transpose2_0_out->Var()->GetShape();
    // in shape should be [batch_size, seq_len, 3, num_heads, head_size]
    if (in_shape.size() != 5 || in_shape[2] != 3 || shape.size() != 5 ||
        shape[0] != 3 || shape[2] != in_shape[3]) {
      LOG(WARNING) << "Self-attention shape mismatch!";
      return;
    }
    desc.SetAttr("head_number", static_cast<int>(shape[2]));
    float alpha = 1.0;
    if (matmul_1_op->Op()->HasAttr("alpha"))
      alpha = PADDLE_GET_CONST(float, matmul_1_op->Op()->GetAttr("alpha"));
    desc.SetAttr("alpha", alpha);

    // Create a new node for the fused op.
    auto self_attention_node = graph->CreateOpNode(&desc);

    // Link inputs and outputs.
    PADDLE_ENFORCE_NE(subgraph.count(x),
                      0,
                      common::errors::NotFound(
                          "Detector did not find input x of self attention."));

    IR_NODE_LINK_TO(subgraph.at(x), self_attention_node);    // Input
    IR_NODE_LINK_TO(self_attention_node, transpose2_2_out);  // Output

    // Delete the unneeded nodes.
    std::unordered_set<const Node*> marked_nodes({transpose2_0_op,
                                                  transpose2_0_out,
                                                  slice_0_op,
                                                  slice_0_out,
                                                  slice_1_op,
                                                  slice_1_out,
                                                  slice_2_op,
                                                  slice_2_out,
                                                  matmul_0_op,
                                                  matmul_0_out,
                                                  matmul_1_op,
                                                  matmul_1_out,
                                                  transpose2_1_op,
                                                  transpose2_1_out,
                                                  softmax_op,
                                                  softmax_out,
                                                  transpose2_2_op});

    GraphSafeRemoveNodes(graph, marked_nodes);
    ++fusion_count;
  };
  gpd(graph, handler);
  AddStatis(fusion_count);
  if (!Has("disable_logs") || !Get<bool>("disable_logs")) {
    PrettyLogDetail(
        "---    fused %d self attention (of scaled_dp_attention) with %s",
        fusion_count,
        pattern_name);
  }
}

}  // namespace paddle::framework::ir

REGISTER_PASS(self_attention_fuse_pass,
              paddle::framework::ir::SelfAttentionFusePass);
REGISTER_PASS_CAPABILITY(self_attention_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .EQ("transpose2", 0)
            .EQ("slice", 0)
            .EQ("scale", 0)
            .EQ("softmax", 0)
            .EQ("matmul_v2", 0));
