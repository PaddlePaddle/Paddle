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

#pragma once

#include <memory>
#include <string>

#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph.h"

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {

struct MultiTransformerLayerPattern : public PatternBase {
  MultiTransformerLayerPattern(PDPattern* pattern,
                               const std::string& name_scope)
      : PatternBase(pattern, name_scope, "fuse_multi_transformer_layer") {}

  std::unordered_map<std::string, std::string> operator()(
      bool enable_int8, int num_fused_op = 1, bool is_decoder = false);

  PATTERN_DECL_NODE(src_mask);
  PATTERN_DECL_NODE(x0);
};

}  // namespace patterns

class FuseMultiTransformerLayerPass : public FusePassBase {
 public:
  FuseMultiTransformerLayerPass() {}
  virtual ~FuseMultiTransformerLayerPass() {}

 protected:
  void ApplyImpl(Graph* graph) const;

  const std::string name_scope_{"fuse_multi_transformer_layer"};

 private:
  int BuildFusion(Graph* graph,
                  const std::string& name_scope,
                  Scope* scope) const;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
