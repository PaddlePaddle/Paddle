// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/mkldnn/reshape_transpose_matmul_mkldnn_fuse_pass.h"

namespace paddle {
namespace framework {
namespace ir {
/*
 * Fuse Reshape->Transpose->MatMulV2 when MatMulV2 uses mkldnn.
 */

class ReshapeTransposeMatmulV2MkldnnFusePass
    : public ReshapeTransposeMatmulMkldnnFusePass {
 public:
  ReshapeTransposeMatmulV2MkldnnFusePass();
  virtual ~ReshapeTransposeMatmulV2MkldnnFusePass() {}

 protected:
  const std::string name_scope_{"reshape_transpose_matmul_v2_fuse"};
};
}  // namespace ir
}  // namespace framework
}  // namespace paddle
