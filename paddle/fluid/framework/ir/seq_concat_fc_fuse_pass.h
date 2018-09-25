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
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace ir {

namespace patterns {

// this pattern used to fuse two sequence_expand + concat
// + fc (with bias) + activation
struct SeqConcatFCPattern : public PatternBase {
  SeqConcatFCPattern(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "seq_concat_fc") {}

  PDNode* operator()(PDNode* x0, PDNode* x1, PDNode* y);

  // build sequence_expand and concat fusion pattern.
  // x0 input of sequence_expand0
  // x1 input of sequence_expand1
  // concat y, outputs of sequence_expand0 and sequence_expand1
  PDNode* BuildSeqExpandConcatPattern(PDPattern* pattern, PDNode* x0,
                                      PDNode* x1, PDNode* y);

  // Operators
  PATTERN_DECL_NODE(seq_expand0);
  PATTERN_DECL_NODE(seq_expand1);
  PATTERN_DECL_NODE(concat);
  PATTERN_DECL_NODE(mul);
  PATTERN_DECL_NODE(elementwise_add);
  PATTERN_DECL_NODE(activation)

  // Variables
  PATTERN_DECL_NODE(weight);
  PATTERN_DECL_NODE(bias);

  // Outputs
  PATTERN_DECL_NODE(seq_expand0_out);
  PATTERN_DECL_NODE(seq_expand1_out);
  PATTERN_DECL_NODE(concat_out);
  PATTERN_DECL_NODE(mul_out);
  PATTERN_DECL_NODE(elementwise_add_out);
  PATTERN_DECL_NODE(activation_out);
};

}  // namespace patterns

class SeqConcatFcFusePass : public FusePassBase {
 public:
  virtual ~SeqConcatFcFusePass() {}

 protected:
  std::unique_ptr<ir::Graph> ApplyImpl(std::unique_ptr<ir::Graph> graph) const;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
