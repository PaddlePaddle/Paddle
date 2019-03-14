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

#include <string>
#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {

struct ReshapeTransposeScaleMatmul : public PatternBase {
  ReshapeTransposeScaleMatmul(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "reshape_transpose_matmul_fuse") {}
  PDNode* operator()(PDNode* matmul_output, bool scale = false);
  // declare operator node's name
  PATTERN_DECL_NODE(reshape);
  PATTERN_DECL_NODE(transpose);
  PATTERN_DECL_NODE(scale);
  PATTERN_DECL_NODE(matmul);
  PATTERN_DECL_NODE(eltwise_add);
  // declare variable node's name
  PATTERN_DECL_NODE(reshape_input);
  PATTERN_DECL_NODE(transpose_input);
  PATTERN_DECL_NODE(scale_input);
  PATTERN_DECL_NODE(matmul_input);
};

};  // namespace patterns

/*
 * Fuse Matmul+Reshape+Transpose+Scale operators to a Matmul.
 */
class ReshapeTransposeScaleMatmulFusePass : public FusePassBase {
 public:
  virtual ~ReshapeTransposeScaleMatmulFusePass() {}

 protected:
  std::unique_ptr<ir::Graph> ApplyImpl(std::unique_ptr<ir::Graph> graph) const;
  const std::string name_scope_{"reshape_transpose_scale_matmul_fuse"};
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
