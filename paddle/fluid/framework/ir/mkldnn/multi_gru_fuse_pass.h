// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

namespace paddle {
namespace framework {
namespace ir {

// This pass fuses two concatenated fusion_gru ops into a single multi_gru op.
// It turns
// a -> fusion_gru -> c -> concat -> e
//   \> fusion_gru -> d /
// into
// a -> multi_gru -> e
class MultiGRUFusePass : public FusePassBase {
 public:
  virtual ~MultiGRUFusePass() {}

 protected:
  void ApplyImpl(ir::Graph* graph) const override;
  const std::string name_scope_{"multi_gru"};
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
