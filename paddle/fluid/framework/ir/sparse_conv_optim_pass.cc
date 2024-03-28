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
      .End();
}

void SparseConvOptimPass::ApplyImpl(ir::Graph* graph) const {
  const std::string pattern_name = "sparse_conv_optim_partern";
  FusePassBase::Init(pattern_name, graph);

  GraphPatternDetector gpd;
  auto* scope = param_scope();
  PADDLE_ENFORCE_NOT_NULL(
      scope,
      platform::errors::InvalidArgument(
          "Scope in SparseConvOptimPass should not be null."));
  // Create pattern
  patterns::SparseConvOptimPartern pattern(gpd.mutable_pattern(), pattern_name);
  pattern();
  int found_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_NODES;

    // if (!IsCompat(subgraph, g)) {
    //   LOG(WARNING) << "sparse_conv_optim_pass compat check failed.";
    //   return;
    // }

    bool is_subm = PADDLE_GET_CONST(bool, sp_conv3d_op->Op()->GetAttr("subm"));
    int groups = PADDLE_GET_CONST(int, sp_conv3d_op->Op()->GetAttr("groups"));
    std::vector<int> dilations = PADDLE_GET_CONST(
        std::vector<int>, sp_conv3d_op->Op()->GetAttr("dilations"));
    std::vector<int> paddings = PADDLE_GET_CONST(
        std::vector<int>, sp_conv3d_op->Op()->GetAttr("paddings"));
    std::vector<int> strides = PADDLE_GET_CONST(
        std::vector<int>, sp_conv3d_op->Op()->GetAttr("strides"));

    auto base_op_desc = *sp_conv3d_op->Op()->Proto();

    std::string kernel_name = sp_conv3d_kernel->Name();

    if (scope->FindVar(kernel_name) == nullptr) return;

    auto* kernel = scope->GetVar(kernel_name)->GetMutable<phi::DenseTensor>();
    bool is2D = kernel->dims().size() == 4 ? true : false;

    if (is2D) {
      if (is_subm && groups == 1 && strides[0] == 1 && strides[1] == 1 &&
          dilations[0] == 1 && dilations[1] == 1) {
        sp_conv3d_op->Op()->SetType("sparse_conv3d_implicit_gemm");
      }
    } else {
      if (is_subm && groups == 1 && strides[0] == 1 && strides[1] == 1 &&
          strides[2] == 1 && dilations[0] == 1 && dilations[1] == 1 &&
          dilations[2] == 1) {
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
