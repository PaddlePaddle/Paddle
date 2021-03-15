/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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

//     |(rank 4)   |(rank 2)                    |(rank 4)    |(rank 2)
//     |       unsqueeze2(axes=[2,3])           |            |
//     |           |                    fuse     \          /
//     |------elementwise_mul(axis=-1)   ->   elementwise_mul(axis=0)
//                 |                                   |
//                 |                                   |
//
// Notice:
// the rank of input is obtained from var_desc,
// it maybe change in runtime.
class UnsqueezeEltwiseFusePass : public FusePassBase {
 public:
  virtual ~UnsqueezeEltwiseFusePass() {}

 protected:
  void ApplyImpl(ir::Graph* graph) const override;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
