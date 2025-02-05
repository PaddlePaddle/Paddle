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

#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace ir {

// BatchMergePass is used to copy forward and backward ops for several
// times to run several batches to simulate large batch size training
// as if we have more than 1 GPUs.
// User can define how many batches to run, gradients will be merged
// through those repeats, and then do optimization using merged gradients.
// This pass is extremely useful when doing large batch-size distributed
// sync training, we can simulate even large batch size as if we have more
// GPUs.

class Graph;

class BatchMergePass : public Pass {
 public:
  virtual ~BatchMergePass() {}

 protected:
  void ApplyImpl(Graph* graph) const override;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
