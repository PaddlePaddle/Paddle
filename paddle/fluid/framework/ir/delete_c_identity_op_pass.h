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
#include <vector>

#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"

namespace paddle {
namespace framework {
namespace ir {

namespace patterns {
struct DeleteCIdentityOpPattern : public PatternBase {
  DeleteCIdentityOpPattern(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "delete_c_identity_op_pattern") {}

  void operator()();

  PATTERN_DECL_NODE(any_op_out);
  PATTERN_DECL_NODE(c_identity_op);
  PATTERN_DECL_NODE(c_identity_op_out);
  PATTERN_DECL_NODE(any_op2);
};
}  // namespace patterns

class Graph;

class DeleteCIdentityOpPass : public FusePassBase {
 public:
  DeleteCIdentityOpPass();
  virtual ~DeleteCIdentityOpPass() {}

 protected:
  void ApplyImpl(ir::Graph* graph) const override;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
