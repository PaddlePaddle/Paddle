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
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace ir {

class Graph;

/*
 * Update pool2d's attr to speed up trt engine.
 *
 * when adaptive=true, ksize=[1,1], we turn to adaptive=false,
 * global_pooling=true.
 */
class AdaptivePool2dConvertGlobalPass : public FusePassBase {
 public:
  AdaptivePool2dConvertGlobalPass();
  virtual ~AdaptivePool2dConvertGlobalPass() = default;

 protected:
  void ApplyImpl(ir::Graph* graph) const override;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
