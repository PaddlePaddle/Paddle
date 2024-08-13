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

#include "paddle/fluid/framework/ir/vit_attention_fuse_pass.h"

#include <string>
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"

#define GET_IR_NODE(node__) GET_IR_NODE_FROM_SUBGRAPH(node__, node__, pattern);
#define GET_NODES                 \
  GET_IR_NODE(matmul0_op);        \
  GET_IR_NODE(matmul0_in_y);      \
  GET_IR_NODE(matmul0_out);       \
  GET_IR_NODE(elementwise0_op);   \
  GET_IR_NODE(elementwise0_in_y); \
  GET_IR_NODE(elementwise0_out);  \
  GET_IR_NODE(reshape1_op);       \
  GET_IR_NODE(reshape1_out);      \
  GET_IR_NODE(transpose1_op);     \
  GET_IR_NODE(transpose1_out);    \
  GET_IR_NODE(slice1_op);         \
  GET_IR_NODE(slice1_out);        \
  GET_IR_NODE(slice2_op);         \
  GET_IR_NODE(slice2_out);        \
  GET_IR_NODE(slice3_op);         \
  GET_IR_NODE(slice3_out);        \
  GET_IR_NODE(matmul1_op);        \
  GET_IR_NODE(matmul1_out);       \
  GET_IR_NODE(scale1_op);         \
  GET_IR_NODE(scale1_out);        \
  GET_IR_NODE(transpose2_op);     \
  GET_IR_NODE(transpose2_out);    \
  GET_IR_NODE(softmax1_op);       \
  GET_IR_NODE(softmax1_out);      \
  GET_IR_NODE(matmul2_op);        \
  GET_IR_NODE(matmul2_out);       \
  GET_IR_NODE(transpose3_op);     \
  GET_IR_NODE(transpose3_out);    \
  GET_IR_NODE(reshape2_op);       \
  GET_IR_NODE(reshape2_out);

namespace paddle::framework::ir {

bool HasScale(OpDesc* const op_ptr,
              std::string* name,
              std::string regexp = "Input_scale_") {
  name->clear();
  std::unordered_map<std::string, Attribute> attr_map = op_ptr->GetAttrMap();
  std::unordered_map<std::string, Attribute>::iterator iter;
  int len = static_cast<int>(regexp.size());
  for (iter = attr_map.begin(); iter != attr_map.end(); iter++) {
    if (regexp == iter->first.substr(0, len)) {
      *name = iter->first;
      return true;
    }
  }
  return false;
}

void VitAttentionFusePass::ApplyImpl(ir::Graph* graph) const {
  GraphPatternDetector gpd;
  const std::string pattern_name = "vit_attention_fuse";
  FusePassBase::Init(pattern_name, graph);
  auto* scope = param_scope();

  // pattern
  std::unordered_set<std::string> matmul_ops{"matrix_multiply"};
  PDNode* x = gpd.mutable_pattern()
                  ->NewNode("x")
                  ->assert_is_ops_input(matmul_ops, "X")
                  ->AsInput();
  patterns::VitAttention pattern(gpd.mutable_pattern(), pattern_name);
  pattern(x);

  int fusion_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_NODES;
    // do something;
    OpDesc desc(matmul0_op->Op()->Block());
    desc.SetType("multihead_matmul");
    desc.SetInput("Input", {subgraph.at(x)->Name()});
    if (matmul0_out->Var()->GetShape().size() != 3) {
      VLOG(3) << "vit_attention_fuse_pass only support input.dim == 3";
      return;
    }
    // refactor W and Bias
    auto* w_tensor =
        scope->FindVar(matmul0_in_y->Name())->GetMutable<phi::DenseTensor>();
    auto w_dims =
        common::make_ddim({w_tensor->dims()[0], 3, w_tensor->dims()[1] / 3});
    w_tensor->Resize(w_dims);

    auto* b_tensor = scope->FindVar(elementwise0_in_y->Name())
                         ->GetMutable<phi::DenseTensor>();
    auto bias_dims = common::make_ddim({3, b_tensor->dims()[0] / 3});
    b_tensor->Resize(bias_dims);

    desc.SetInput("W", {matmul0_in_y->Name()});
    desc.SetInput("Bias", {elementwise0_in_y->Name()});
    std::vector<int64_t> shape = softmax1_out->Var()->GetShape();
    desc.SetOutput("Out", {reshape2_out->Name()});
    desc.SetAttr("head_number", static_cast<int>(shape[1]));
    float alpha = PADDLE_GET_CONST(float, scale1_op->Op()->GetAttr("scale"));
    desc.SetAttr("alpha", alpha);

    // int8 for fc
    std::string scale_name;
    if (HasScale(matmul0_op->Op(), &scale_name)) {
      desc.SetAttr("Input_scale", matmul0_op->Op()->GetAttr(scale_name));
    }
    if (HasScale(elementwise0_op->Op(), &scale_name, "Out")) {
      desc.SetAttr("fc_out_threshold",
                   elementwise0_op->Op()->GetAttr(scale_name));
    }

    // Create a new node for the fused op.
    auto vit_attention_node = graph->CreateOpNode(&desc);

    // Link inputs and outputs.
    PADDLE_ENFORCE_NE(
        subgraph.count(x),
        0,
        common::errors::NotFound("Detector did not find input x of conv2d."));

    IR_NODE_LINK_TO(subgraph.at(x), vit_attention_node);  // Input
    IR_NODE_LINK_TO(matmul0_in_y, vit_attention_node);
    IR_NODE_LINK_TO(elementwise0_in_y, vit_attention_node);
    IR_NODE_LINK_TO(vit_attention_node, reshape2_out);  // Output

    // Delete the unneeded nodes.
    std::unordered_set<const Node*> marked_nodes(
        {matmul0_op,    matmul0_out,    elementwise0_op, elementwise0_out,
         reshape1_op,   reshape1_out,   transpose1_op,   transpose1_out,
         slice1_op,     slice1_out,     slice2_op,       slice2_out,
         slice3_op,     slice3_out,     matmul1_op,      matmul1_out,
         scale1_op,     scale1_out,     transpose2_op,   transpose2_out,
         softmax1_op,   softmax1_out,   matmul2_op,      matmul2_out,
         transpose3_op, transpose3_out, reshape2_op});

    GraphSafeRemoveNodes(graph, marked_nodes);
    ++fusion_count;
  };
  gpd(graph, handler);
  AddStatis(fusion_count);
}

}  // namespace paddle::framework::ir

REGISTER_PASS(vit_attention_fuse_pass,
              paddle::framework::ir::VitAttentionFusePass);
REGISTER_PASS_CAPABILITY(vit_attention_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .EQ("reshape2", 0)
            .EQ("transpose2", 0)
            .EQ("slice", 0)
            .EQ("scale", 0)
            .EQ("softmax", 0));
