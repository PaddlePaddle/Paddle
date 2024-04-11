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
#include <utility>

#include "paddle/fluid/framework/ir/fuse_pass_base.h"

namespace paddle {
namespace framework {
namespace ir {
class Graph;
}  // namespace ir
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {
struct EmbEltwiseLayernorm : public PatternBase {
  EmbEltwiseLayernorm(PDPattern *pattern, const std::string &name_scope)
      : PatternBase(pattern, name_scope, "emb_elt_layernorm") {}

  void operator()();
  PATTERN_DECL_NODE(emb_elt_layernorm_op);
  PATTERN_DECL_NODE(emb_elt_layernorm_out);
};

struct PrelnEmbEltwiseLayernorm : public PatternBase {
  PrelnEmbEltwiseLayernorm(PDPattern *pattern, const std::string &name_scope)
      : PatternBase(pattern, name_scope, "preln_emb_elt_layernorm") {}

  void operator()();
  PATTERN_DECL_NODE(preln_emb_elt_layernorm_op);
  PATTERN_DECL_NODE(preln_emb_elt_layernorm_out_0);
  PATTERN_DECL_NODE(preln_emb_elt_layernorm_out_1);
};

struct SkipLayernorm : public PatternBase {
  SkipLayernorm(PDPattern *pattern, const std::string &name_scope)
      : PatternBase(pattern, name_scope, "skip_layernorm") {}

  void operator()();

  PATTERN_DECL_NODE(skip_layernorm_x);
  PATTERN_DECL_NODE(skip_layernorm_y);
  PATTERN_DECL_NODE(skip_layernorm_op);
  PATTERN_DECL_NODE(skip_layernorm_out);
};

struct PrelnSkipLayernorm : public PatternBase {
  PrelnSkipLayernorm(PDPattern *pattern, const std::string &name_scope)
      : PatternBase(pattern, name_scope, "preln_skip_layernorm") {}

  void operator()();

  PATTERN_DECL_NODE(preln_skip_layernorm_x);
  PATTERN_DECL_NODE(preln_skip_layernorm_y);
  PATTERN_DECL_NODE(preln_skip_layernorm_op);
  PATTERN_DECL_NODE(preln_skip_layernorm_out_0);
  PATTERN_DECL_NODE(preln_skip_layernorm_out_1);
};

struct MultiheadMatmul : public PatternBase {
  MultiheadMatmul(PDPattern *pattern, const std::string &name_scope)
      : PatternBase(pattern, name_scope, "multihead_matmul") {}

  void operator()();

  PATTERN_DECL_NODE(multihead_matmul_input);
  PATTERN_DECL_NODE(multihead_matmul_op);
  PATTERN_DECL_NODE(multihead_matmul_out);
};

struct MatrixMultiply : public PatternBase {
  MatrixMultiply(PDPattern *pattern, const std::string &name_scope)
      : PatternBase(pattern, name_scope, "matrix_multiply") {}

  void operator()();

  PATTERN_DECL_NODE(matrix_multiply_input);
  PATTERN_DECL_NODE(matrix_multiply_op);
};

struct Activation : public PatternBase {
  Activation(PDPattern *pattern, const std::string &name_scope)
      : PatternBase(pattern, name_scope, "activation") {}

  void operator()();

  PATTERN_DECL_NODE(activation_input);
  PATTERN_DECL_NODE(activation_op);
  PATTERN_DECL_NODE(activation_out);
};

struct FusedTokenPrune : public PatternBase {
  FusedTokenPrune(PDPattern *pattern, const std::string &name_scope)
      : PatternBase(pattern, name_scope, "fused_token_prune") {}

  void operator()();

  PATTERN_DECL_NODE(fused_token_prune_input);
  PATTERN_DECL_NODE(fused_token_prune_op);
  PATTERN_DECL_NODE(fused_token_prune_output);
};

struct ElementWise : public PatternBase {
  ElementWise(PDPattern *pattern, const std::string &name_scope)
      : PatternBase(pattern, name_scope, "elementwise") {}

  void operator()();

  PATTERN_DECL_NODE(elementwise_input);
  PATTERN_DECL_NODE(elementwise_weight);
  PATTERN_DECL_NODE(elementwise_op);
  PATTERN_DECL_NODE(elementwise_out);
};
}  // namespace patterns

class RemovePaddingRecoverPaddingPass : public FusePassBase {
 public:
  RemovePaddingRecoverPaddingPass() {}
  virtual ~RemovePaddingRecoverPaddingPass() {}

 protected:
  void ApplyImpl(Graph *graph) const;
  const std::string name_scope_{"remove_padding_recover_padding_pass"};
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
