// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {

struct KVCacheDecoderPattern : public PatternBase {
  KVCacheDecoderPattern(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "kvcache_decoder") {}

  PDNode* operator()();

  // declare operator node's name
  PATTERN_DECL_NODE(input0);

  PATTERN_DECL_NODE(mul0);
  PATTERN_DECL_NODE(mul1);
  PATTERN_DECL_NODE(mul2);
  PATTERN_DECL_NODE(mul0_w);
  PATTERN_DECL_NODE(mul1_w);
  PATTERN_DECL_NODE(mul2_w);
  PATTERN_DECL_NODE(mul0_out);
  PATTERN_DECL_NODE(mul1_out);
  PATTERN_DECL_NODE(mul2_out);
  PATTERN_DECL_NODE(eltadd0);    // ELEMENTWISE_ADD
  PATTERN_DECL_NODE(eltadd1);    // ELEMENTWISE_ADD
  PATTERN_DECL_NODE(eltadd2);    // ELEMENTWISE_ADD
  PATTERN_DECL_NODE(eltadd0_b);  // ELEMENTWISE_ADD
  PATTERN_DECL_NODE(eltadd1_b);  // ELEMENTWISE_ADD
  PATTERN_DECL_NODE(eltadd2_b);  // ELEMENTWISE_ADD
  PATTERN_DECL_NODE(eltadd0_out);
  PATTERN_DECL_NODE(eltadd1_out);
  PATTERN_DECL_NODE(eltadd2_out);
  PATTERN_DECL_NODE(reshape2_0);
  PATTERN_DECL_NODE(reshape2_1);
  PATTERN_DECL_NODE(reshape2_2);
  PATTERN_DECL_NODE(reshape2_qkv);
  PATTERN_DECL_NODE(reshape2_0_out);
  PATTERN_DECL_NODE(reshape2_1_out);
  PATTERN_DECL_NODE(reshape2_2_out);
  PATTERN_DECL_NODE(reshape2_qkv_out);
  PATTERN_DECL_NODE(transpose2_0);
  PATTERN_DECL_NODE(transpose2_1);
  PATTERN_DECL_NODE(transpose2_2);
  PATTERN_DECL_NODE(transpose2_qkv);
  PATTERN_DECL_NODE(transpose2_0_out);
  PATTERN_DECL_NODE(transpose2_1_out);
  PATTERN_DECL_NODE(transpose2_2_out);
  PATTERN_DECL_NODE(transpose2_qkv_out);
  PATTERN_DECL_NODE(scale);
  PATTERN_DECL_NODE(scale_out);
  PATTERN_DECL_NODE(matmul_qk);
  PATTERN_DECL_NODE(matmul_qk_out);
  PATTERN_DECL_NODE(eltadd_qk);
  PATTERN_DECL_NODE(eltadd_qk_b);
  PATTERN_DECL_NODE(eltadd_qk_out);
  PATTERN_DECL_NODE(softmax_qk);
  PATTERN_DECL_NODE(softmax_qk_out);

  PATTERN_DECL_NODE(matmul_qkv);
  PATTERN_DECL_NODE(matmul_qkv_out);


  // KVCache
  PATTERN_DECL_NODE(concat1);
  PATTERN_DECL_NODE(concat2);
  PATTERN_DECL_NODE(concat1_out);
  PATTERN_DECL_NODE(concat2_out);
  PATTERN_DECL_NODE(assign1);
  PATTERN_DECL_NODE(assign2);
  PATTERN_DECL_NODE(k_cache_w);
  PATTERN_DECL_NODE(v_cache_w);
  PATTERN_DECL_NODE(k_cache_r);
  PATTERN_DECL_NODE(v_cache_r);
  
};

}  // namespace patterns


class KVCacheDecoderFusePass : public FusePassBase {
 public:
  KVCacheDecoderFusePass();

 protected:
  void ApplyImpl(Graph* graph) const;

  const std::string name_scope_{"kvcache_decoder_fuse"};

 private:
  int BuildFusion(Graph* graph, const std::string& name_scope,
                    Scope* scope) const;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
