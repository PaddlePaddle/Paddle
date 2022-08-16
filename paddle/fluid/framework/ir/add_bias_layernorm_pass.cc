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

#include "paddle/fluid/framework/ir/add_bias_layernorm_pass.h"

#include <string>
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"

#define GET_IR_NODE(node__) GET_IR_NODE_FROM_SUBGRAPH(node__, node__, pattern);
#define GET_NODES                    \
  GET_IR_NODE(elementwise_add_op);   \
  GET_IR_NODE(elementwise_add_in_y); \
  GET_IR_NODE(elementwise_add_out);  \
  GET_IR_NODE(flatten_op);           \
  GET_IR_NODE(flatten_out);          \
  GET_IR_NODE(flatten_out_xshape);   \
  GET_IR_NODE(transpose_op);         \
  GET_IR_NODE(transpose_out);        \
  GET_IR_NODE(transpose_out_xshape); \
  GET_IR_NODE(layer_norm_op);        \
  GET_IR_NODE(layer_norm_in_bias);   \
  GET_IR_NODE(layer_norm_in_scale);  \
  GET_IR_NODE(layer_norm_out_y);     \
  GET_IR_NODE(layer_norm_out_mean);  \
  GET_IR_NODE(layer_norm_out_variance);

namespace paddle {
namespace framework {
namespace ir {

void UnlinkNodes(ir::Node* a, ir::Node* b) {
  a->outputs.erase(std::remove(a->outputs.begin(), a->outputs.end(), b),
                   a->outputs.end());
  b->inputs.erase(std::remove(b->inputs.begin(), b->inputs.end(), a),
                  b->inputs.end());
}

AddBiasLayernormPass::AddBiasLayernormPass() {
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
      .AddAttr("axis")  // {0, 2, 1, 3}
      .IsType<std::vector<int>>()
      .End();

  AddOpCompat(OpCompat("flatten_contiguous_range"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddOutput("XShape")
      .IsTensor()
      .End()
      .AddAttr("start_axis")
      .IsNumEQ(2)
      .End()
      .AddAttr("stop_axis")
      .IsNumEQ(3)
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
      .End()
      .AddOutput("Variance")
      .IsTensor()
      .End()
      .AddAttr("epsilon")
      .IsNumGE(0.0f)
      .IsNumLE(0.001f)
      .End()
      .AddAttr("begin_norm_axis")
      .IsNumGT(0)
      .End();
}

void AddBiasLayernormPass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));

  GraphPatternDetector gpd;
  const std::string pattern_name = "add_bias_layernorm";
  FusePassBase::Init(pattern_name, graph);

  // pattern
  PDNode* x = gpd.mutable_pattern()
                  ->NewNode("add_bias_layernorm/x")
                  ->AsInput()
                  ->assert_is_op_input("elementwise_add", "X")
                  ->assert_var_not_persistable();
  patterns::AddBiasLayernormPattern pattern(gpd.mutable_pattern(),
                                            pattern_name);
  pattern(x);

  int fusion_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    if (!IsCompat(subgraph, graph)) {
      LOG(WARNING) << "Pass in op compat failed.";
      return;
    }
    GET_NODES;

    PADDLE_ENFORCE_NE(subgraph.count(x),
                      0,
                      platform::errors::NotFound(
                          "Detector did not find input x of elementwise_add."));

    // update elem_add weight axis
    auto* op = elementwise_add_op->Op();
    if (op->HasAttr("axis")) {
      LOG(INFO) << "update elem_add weight";
      op->SetAttr("axis", 2);
    }

    // unlink nodes
    auto input_node = subgraph.at(x);
    UnlinkNodes(input_node, elementwise_add_op);
    UnlinkNodes(elementwise_add_out, flatten_op);
    UnlinkNodes(transpose_out, layer_norm_op);

    // relink nodes
    flatten_op->Op()->SetInput("X", {input_node->Name()});
    IR_NODE_LINK_TO(input_node, flatten_op);

    elementwise_add_op->Op()->SetInput("X", {transpose_out->Name()});
    IR_NODE_LINK_TO(transpose_out, elementwise_add_op);

    layer_norm_op->Op()->SetInput("X", {elementwise_add_out->Name()});
    IR_NODE_LINK_TO(elementwise_add_out, layer_norm_op);

    ++fusion_count;
  };
  gpd(graph, handler);
  AddStatis(fusion_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(add_bias_layernorm_pass,
              paddle::framework::ir::AddBiasLayernormPass);
REGISTER_PASS_CAPABILITY(add_bias_layernorm_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .GE("flatten_contiguous_range", 0)
            .GE("transpose2", 0)
            .LE("elementwise_add", 1)
            .EQ("layer_norm", 0));
