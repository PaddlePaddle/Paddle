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
struct RecoverPadding : public PatternBase {
  RecoverPadding(PDPattern *pattern, const std::string &name_scope)
      : PatternBase(pattern, name_scope, "recover_padding") {}

  void operator()();

  PATTERN_DECL_NODE(recover_padding_input);
  PATTERN_DECL_NODE(recover_padding_op);
  PATTERN_DECL_NODE(recover_padding_out);
};
}  // namespace patterns

class DeleteRemovePaddingRecoverPaddingPass : public FusePassBase {
 public:
  DeleteRemovePaddingRecoverPaddingPass() {}
  virtual ~DeleteRemovePaddingRecoverPaddingPass() {}

 protected:
  void ApplyImpl(Graph *graph) const;
  const std::string name_scope_{"delete_remove_padding_recover_padding_pass"};
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
