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

namespace paddle {
namespace framework {
namespace ir {

class ReshapeTransposeMatmulMkldnnFusePass : public FusePassBase {
 public:
  ReshapeTransposeMatmulMkldnnFusePass();
  virtual ~ReshapeTransposeMatmulMkldnnFusePass() {}

 protected:
<<<<<<< HEAD
  void ApplyImpl(ir::Graph* graph) const override;
  const std::string name_scope_{"reshape_transpose_matmul_fuse"};

  void Fuse(Graph* graph,
=======
  void ApplyImpl(Graph* graph) const override;
  void Fuse(Graph* graph,
            const std::string& matmul_type,
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
            bool with_reshape_xshape,
            bool with_transpose_xshape) const;
};
}  // namespace ir
}  // namespace framework
}  // namespace paddle
