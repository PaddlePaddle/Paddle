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

#include "paddle/fluid/framework/ir/relu6_fuse_pass.h"

#include <cmath>
#include <string>

#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle::framework::ir {

void Relu6FusePass::ApplyImpl(ir::Graph* graph) const {
  // This pass is now used for xpu, because xpu can fuse conv + bias + relu6
  const std::string pattern_name = "relu6_fuse";
  FusePassBase::Init(pattern_name, graph);

  GraphPatternDetector gpd;

  auto* clip_x = gpd.mutable_pattern()
                     ->NewNode("clip_x")
                     ->assert_is_op_input("clip", "X")
                     ->assert_var_not_persistable()
                     ->AsInput();
  auto clip_op =
      gpd.mutable_pattern()->NewNode("clip_op")->assert_is_op("clip");
  auto clip_min = gpd.mutable_pattern()
                      ->NewNode("clip_min")
                      ->assert_is_op_input("clip", "Min")
                      ->assert_is_persistable_var()
                      ->assert_more([](Node* node) {
                        return node->Var()->GetShape().size() == 1;
                      })
                      ->AsInput();
  auto clip_max = gpd.mutable_pattern()
                      ->NewNode("clip_max")
                      ->assert_is_op_input("clip", "Max")
                      ->assert_is_persistable_var()
                      ->assert_more([](Node* node) {
                        return node->Var()->GetShape().size() == 1;
                      })
                      ->AsInput();
  auto clip_out = gpd.mutable_pattern()
                      ->NewNode("clip_out")
                      ->assert_is_op_output("clip")
                      ->AsOutput();

  clip_op->LinksFrom({clip_x, clip_min, clip_max}).LinksTo({clip_out});

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    Node* clip_x_node = subgraph.at(clip_x);
    Node* clip_op_node = subgraph.at(clip_op);
    Node* clip_max_node = subgraph.at(clip_max);
    Node* clip_min_node = subgraph.at(clip_min);
    Node* clip_out_node = subgraph.at(clip_out);

    auto* scope = param_scope();
    PADDLE_ENFORCE_NOT_NULL(
        scope, common::errors::InvalidArgument("Scope cannot be nullptr."));

    const auto& clip_max_t =
        scope->GetVar(clip_max_node->Name())->Get<phi::DenseTensor>();
    auto clip_max_t_dims = clip_max_t.dims();
    PADDLE_ENFORCE_EQ(
        clip_max_t_dims.size(),
        1,
        common::errors::InvalidArgument("the size(%d) of clip max tensor "
                                        "must equal 1",
                                        clip_max_t_dims.size()));
    const auto& clip_min_t =
        scope->GetVar(clip_min_node->Name())->Get<phi::DenseTensor>();
    auto clip_min_t_dims = clip_min_t.dims();
    PADDLE_ENFORCE_EQ(
        clip_min_t_dims.size(),
        1,
        common::errors::InvalidArgument("the size(%d) of clip max tensor "
                                        "must equal 1",
                                        clip_min_t_dims.size()));
    auto tensor_type = clip_max_t.dtype();
    float max_val_ = 0.f;
    float min_val_ = 1.f;
    if (tensor_type == phi::DataType::FLOAT16) {
      auto* clip_max_t_fp16_ptr = clip_max_t.data<phi::dtype::float16>();
      auto* clip_min_t_fp16_ptr = clip_min_t.data<phi::dtype::float16>();
      max_val_ = static_cast<float>(clip_max_t_fp16_ptr[0]);
      min_val_ = static_cast<float>(clip_min_t_fp16_ptr[0]);
    } else if (tensor_type == phi::DataType::FLOAT32) {
      auto* clip_max_t_fp32_ptr = clip_max_t.data<float>();
      auto* clip_min_t_fp32_ptr = clip_min_t.data<float>();
      max_val_ = clip_max_t_fp32_ptr[0];
      min_val_ = clip_min_t_fp32_ptr[0];
    } else {
      PADDLE_THROW(common::errors::Unavailable(
          "relu6_fuse_pass do not supported weight dtype. "
          "we now only support fp32/fp16."));
    }
    if (std::abs(max_val_ - 6.0) < 1e-3 && std::abs(min_val_ - 0.0) < 1e-3) {
      OpDesc new_desc;
      new_desc.SetType("relu6");
      new_desc.SetAttr("threshold", 6.f);
      new_desc.SetInput("X", {clip_x_node->Name()});
      new_desc.SetOutput("Out", {clip_out_node->Name()});
      new_desc.Flush();

      std::unordered_set<const Node*> del_node_set;
      del_node_set.insert(clip_op_node);
      del_node_set.insert(clip_max_node);
      del_node_set.insert(clip_min_node);
      GraphSafeRemoveNodes(graph, del_node_set);

      auto fused_node = graph->CreateOpNode(&new_desc);
      IR_NODE_LINK_TO(clip_x_node, fused_node);
      IR_NODE_LINK_TO(fused_node, clip_out_node);
    }
  };
  gpd(graph, handler);
}

}  // namespace paddle::framework::ir

REGISTER_PASS(relu6_fuse_pass, paddle::framework::ir::Relu6FusePass);
