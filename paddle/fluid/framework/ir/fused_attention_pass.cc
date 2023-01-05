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

#include "paddle/fluid/framework/ir/fused_attention_pass.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {

PDNode* FusedAttentionPattern::operator()(PDNode* x,
                                          bool pre_layer_norm,
                                          bool post_layer_norm,
                                          bool has_attn_mask,
                                          bool do_dropout,
                                          bool add_residual) {
  return nullptr;
}

PDNode* FusedAttentionGradPattern::operator()(PDNode* x,
                                              bool pre_layer_norm,
                                              bool post_layer_norm,
                                              bool has_attn_mask,
                                              bool do_dropout,
                                              bool add_residual) {
  return nullptr;
}

}  // namespace patterns

void FusedAttentionsPass::ApplyImpl(Graph* graph) const {
  FusePassBase::Init(name_scope_, graph);
  auto* scope = param_scope();

  graph = PreMaskDropResFwd(graph, name_scope_, scope);
  graph = PreMaskDropResBwd(graph, name_scope_, scope);
}

ir::Graph* FusedAttentionsPass::PreMaskDropResFwd(Graph* graph,
                                                  const std::string& name_scope,
                                                  Scope* scope) const {
  return graph;
}

ir::Graph* FusedAttentionsPass::PreMaskDropResBwd(Graph* graph,
                                                  const std::string& name_scope,
                                                  Scope* scope) const {
  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fused_attention_pass, paddle::framework::ir::FusedAttentionsPass);
