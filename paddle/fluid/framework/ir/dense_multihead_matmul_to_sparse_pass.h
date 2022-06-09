/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. */

#pragma once

#include <string>

#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/inference/api/paddle_analysis_config.h"

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {

struct DenseMultiheadMatmul : public PatternBase {
  DenseMultiheadMatmul(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "dense_multihead_matmul") {}

  PDNode* operator()();

  // declare operator node's name
  PATTERN_DECL_NODE(multihead_matmul);
  PATTERN_DECL_NODE(multihead_matmul_out);
  PATTERN_DECL_NODE(multihead_matmul_input);
  PATTERN_DECL_NODE(multihead_matmul_weights);
  PATTERN_DECL_NODE(multihead_matmul_bias);
  PATTERN_DECL_NODE(multihead_matmul_biasqk);
};
}  // namespace patterns
/**
 * Replace dense multihead_matmul op with sparse multihead_matmul op
 */
class Graph;

class DenseMultiheadMatmulToSparsePass : public FusePassBase {
 public:
  DenseMultiheadMatmulToSparsePass();

 protected:
  void ApplyImpl(ir::Graph* graph) const override;

  const std::string name_scope_{"dense_multihead_matmul_to_sparse_pass"};
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
