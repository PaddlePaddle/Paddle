// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/sparse_conv_optim_pass.h"

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace paddle {
namespace framework {
namespace ir {

#define GET_IR_NODE(node__) GET_IR_NODE_FROM_SUBGRAPH(node__, node__, pattern);
#define GET_NODES                \
  GET_IR_NODE(sp_conv3d_x);      \
  GET_IR_NODE(sp_conv3d_kernel); \
  GET_IR_NODE(sp_conv3d_op);     \
  GET_IR_NODE(sp_conv3d_out);

SparseConvOptimPass::SparseConvOptimPass() {
  AddOpCompat(OpCompat("sparse_conv3d"))
      .AddInput("x")
      .IsTensor()
      .End()
      .AddInput("kernel")
      .IsTensor()
      .End()
      .AddOutput("out")
      .IsTensor()
      .End()
      .AddAttr("dilations")
      .IsType<std::vector<int>>()
      .End()
      .AddAttr("paddings")
      .IsType<std::vector<int>>()
      .End()
      .AddAttr("strides")
      .IsType<std::vector<int>>()
      .End()
      .AddAttr("groups")
      .IsNumGE(1)
      .End()
      .AddAttr("subm")
      .IsType<bool>()
      .End();
}

void SparseConvOptimPass::ApplyImpl(ir::Graph* graph) const {
  const std::string pattern_name = "sparse_conv_optim_partern";
  FusePassBase::Init(pattern_name, graph);

  GraphPatternDetector gpd;
  auto* scope = param_scope();
  PADDLE_ENFORCE_NOT_NULL(
      scope,
      common::errors::InvalidArgument(
          "Scope in SparseConvOptimPass should not be null."));
  // Create pattern
  patterns::SparseConvOptimPartern pattern(gpd.mutable_pattern(), pattern_name);
  pattern();
  int found_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_NODES;

    /*
    if (!IsCompat(subgraph, g)) {
      LOG(WARNING) << "sparse_conv_optim_pass compat check failed.";
      return;
    }
    */

    std::vector<int> dilations = PADDLE_GET_CONST(
        std::vector<int>, sp_conv3d_op->Op()->GetAttr("dilations"));
    std::vector<int> paddings = PADDLE_GET_CONST(
        std::vector<int>, sp_conv3d_op->Op()->GetAttr("paddings"));
    std::vector<int> strides = PADDLE_GET_CONST(
        std::vector<int>, sp_conv3d_op->Op()->GetAttr("strides"));

    auto output_name = sp_conv3d_out->Name();

    auto base_op_desc = *sp_conv3d_op->Op()->Proto();

    PADDLE_ENFORCE_EQ((dilations.size() == paddings.size() &&
                       paddings.size() == strides.size()),
                      true,
                      common::errors::InvalidArgument(
                          "The dilations, paddings, strides must have the same "
                          "rank, but received %d, %d, %d.",
                          dilations.size(),
                          paddings.size(),
                          strides.size()));

    bool is2D = dilations.size() == 2 ? true : false;

    auto sp_conv3d_to_2d = [&]() {
      if (is2D || paddings[0] != 0) return false;

      Node* sp_reshape_unsqueeze = nullptr;
      for (auto* node : sp_conv3d_x->inputs) {
        if (!node->IsOp() || node->Op()->Type() != "sparse_reshape")
          return false;
        auto shape = PADDLE_GET_CONST(std::vector<int64_t>,
                                      node->Op()->GetAttr("shape"));
        if (shape.size() != 5 || shape[0] != 1) return false;
        sp_reshape_unsqueeze = node;
      }
      if (!sp_reshape_unsqueeze || sp_reshape_unsqueeze->inputs.size() != 1)
        return false;
      auto sp_reshape_unsqueeze_x = sp_reshape_unsqueeze->inputs[0];

      Node* reshape = nullptr;
      for (auto* node : sp_conv3d_kernel->inputs) {
        if (!node->IsOp() || node->Op()->Type() != "reshape2") return false;
        auto shape =
            PADDLE_GET_CONST(std::vector<int>, node->Op()->GetAttr("shape"));
        if (shape.size() != 5 || shape[0] != 1) return false;
        reshape = node;
      }
      if (!reshape || reshape->inputs.size() != 1) return false;
      auto reshape_x = reshape->inputs[0];

      Node* sp_reshape_squeeze = nullptr;
      for (auto* node : sp_conv3d_out->outputs) {
        if (!node->IsOp() || node->Op()->Type() != "sparse_reshape")
          return false;
        auto shape = PADDLE_GET_CONST(std::vector<int64_t>,
                                      node->Op()->GetAttr("shape"));
        if (shape.size() != 4) return false;
        sp_reshape_squeeze = node;
      }
      if (!sp_reshape_squeeze || sp_reshape_squeeze->outputs.size() != 1)
        return false;
      auto sp_reshape_squeeze_out = sp_reshape_squeeze->outputs[0];

      dilations = {dilations[1], dilations[2]};
      paddings = {paddings[1], paddings[2]};
      strides = {strides[1], strides[2]};

      sp_conv3d_op->Op()->SetAttr("dilations", dilations);
      sp_conv3d_op->Op()->SetAttr("paddings", paddings);
      sp_conv3d_op->Op()->SetAttr("strides", strides);

      sp_conv3d_op->Op()->SetInput("x", {sp_reshape_unsqueeze_x->Name()});
      sp_conv3d_op->Op()->SetInput("kernel", {reshape_x->Name()});
      sp_conv3d_op->Op()->SetOutput("out", {sp_reshape_squeeze_out->Name()});

      IR_NODE_LINK_TO(sp_reshape_unsqueeze_x, sp_conv3d_op);  // Input
      IR_NODE_LINK_TO(reshape_x, sp_conv3d_op)                // Filter
      IR_NODE_LINK_TO(sp_conv3d_op, sp_reshape_squeeze_out);  // Output

      std::unordered_set<const Node*> nodes2rm = {};

      nodes2rm.insert(sp_reshape_unsqueeze);
      nodes2rm.insert(sp_conv3d_x);
      nodes2rm.insert(reshape);
      nodes2rm.insert(sp_conv3d_kernel);
      nodes2rm.insert(sp_conv3d_out);
      nodes2rm.insert(sp_reshape_squeeze);

      GraphSafeRemoveNodes(graph, nodes2rm);
      is2D = true;
      return true;
    };

    if (sp_conv3d_to_2d()) {
      VLOG(4) << "SparseConv3D(output:" << output_name
              << ") has been converted to 2D implementation!";
    }

    bool is_subm = PADDLE_GET_CONST(bool, sp_conv3d_op->Op()->GetAttr("subm"));

    if (is2D) {
      if (is_subm && strides[0] == 1 && strides[1] == 1 && dilations[0] == 1 &&
          dilations[1] == 1) {
        sp_conv3d_op->Op()->SetType("sparse_conv3d_implicit_gemm");
      }
    } else {
      if (is_subm && strides[0] == 1 && strides[1] == 1 && strides[2] == 1 &&
          dilations[0] == 1 && dilations[1] == 1 && dilations[2] == 1) {
        sp_conv3d_op->Op()->SetType("sparse_conv3d_implicit_gemm");
      }
    }

    found_count++;
  };
  gpd(graph, handler);
  AddStatis(found_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(sparse_conv_optim_pass,
              paddle::framework::ir::SparseConvOptimPass);
