/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include "paddle/fluid/framework/ir/fuse_pass_base.h"

namespace paddle {
namespace framework {
namespace ir {

class Graph;

/*
1. fuse sigmoid + elementwise_mul into swish

Origin subgraph:
              input
            /      \
            |      |
            |     sigmoid
            |      |
            |      |
          elementwise_mul
                |
                |
               out

Fused subgraph:
              input
                |
                |
              swish
                |
                |
               out
*/
class SigmoidElementmulFusePass : public FusePassBase {
 public:
  SigmoidElementmulFusePass();
  virtual ~SigmoidElementmulFusePass() {}

 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  const std::string name_scope_{"sigmoid_elementmul_fuse_pass"};
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
