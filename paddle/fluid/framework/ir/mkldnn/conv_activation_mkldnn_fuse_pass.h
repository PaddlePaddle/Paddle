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

#include <string>
#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace ir {
/*
 * Fuse Conv and Activation base class.
 */
class ConvActivationFusePass : public FusePassBase {
 public:
  virtual ~ConvActivationFusePass() {}
  virtual std::string conv_type() const { return "conv2d"; }
  virtual std::string activation_type() const { return "relu"; }

 protected:
  void ApplyImpl(ir::Graph* graph) const override;
  const std::string name_scope_{"conv_activation_mkldnn_fuse"};
};
/*
 * Fuse Conv and LeakyReLU class
 */
class Conv2DLeakyReLUFusePass : public ConvActivationFusePass {
 public:
  std::string activation_type() const { return "leaky_relu"; }
};
/*
 * Fuse Conv and BoundedReLU class
 */
class Conv2DReLU6FusePass : public ConvActivationFusePass {
 public:
  std::string activation_type() const { return "relu6"; }
};
/*
 * Fuse Conv and Swish class
 */
class Conv2DSwishFusePass : public ConvActivationFusePass {
 public:
  std::string activation_type() const { return "swish"; }
};
}  // namespace ir
}  // namespace framework
}  // namespace paddle
