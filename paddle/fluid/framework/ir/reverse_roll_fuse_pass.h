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

#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace ir {

//    |
// reshape2
//    |
// reshape2
//    |
// transpose2     ->           reverse_roll (shift_size=0)
//    |          fuse
// reshape2
//    |
// reshape2
//    |
//
//   or
//
//    |
// reshape2
//    |
// reshape2
//    |           ->         reverse_roll (shift_size>0)
// transpose2    fuse
//    |
// reshape2
//    |
//   roll
//    |
// reshape2
//    |

class ReverseRollFusePass : public FusePassBase {
 public:
  ReverseRollFusePass();
  virtual ~ReverseRollFusePass() {}

 protected:
  void ApplyImpl(ir::Graph *graph) const override;
  int ApplyPattern(ir::Graph *graph, bool with_roll) const;

 private:
  const std::string scope_name_{"reverse_roll_fuse"};
};
}  // namespace ir
}  // namespace framework
}  // namespace paddle
