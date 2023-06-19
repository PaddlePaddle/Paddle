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
1. fuse clip (Min: 6.0, Max: 6.0) to relu6 op

Origin subgraph:
              Input
                |
                |
              clip (Min: 6.0, Max: 6.0)
                |
                |
              Output

Fused subgraph:
              Input
                |
                |
              relu6
                |
                |
              Output
*/
class Relu6FusePass : public FusePassBase {
 public:
  Relu6FusePass();
  virtual ~Relu6FusePass() {}

 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  const std::string name_scope_{"relu6_fuse_pass"};
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
