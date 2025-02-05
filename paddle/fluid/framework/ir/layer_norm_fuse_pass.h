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

#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph.h"

namespace paddle {
namespace framework {
namespace ir {

/*
 * \brief   Fuse the subgraph representing layer normalization into
 *          layer_norm op.
 *
 * \note    The following graph represents this equation:
 *
 *                       x - u(x)
 *          y(c) * -------------------  + b(c)
 *                 sqrt(sigma^2 + eps)
 *
 *          x        - input data
 *          u(x)     - mean
 *          sigma^2  - standard deviation
 *          eps      - epsilon
 *          y(c)     - gamma (scale) channelwise
 *          b(c)     - beta (shift) channelwise
 *
 *
 *            X
 *           / \
 *          /   reduce_mean "u(x)"
 *          \   /
 *      elementwise_sub     "x - u(x)"
 *      /           \    2
 *      |            \  /
 *      |      elementwise_pow  "(x - u(x))^2"
 *      |             |
 *      |       reduce_mean     "sigma^2 = 1/C*Sum{(x - u(x))^2}"
 *      |             |     eps
 *      |             |     /
 *      |       elementwise_add "sigma^2 + epsilon"
 *      \             |
 *       \           sqrt       "sqrt(sigma^2 + epsilon)"
 *        \          /
 *         \        /
 *       elementwise_div        "lnorm = {x-u(x)}/{sqrt(sigma^2 + epsilon)}"
 *              |
 *       gamma  |
 *          \   |
 *       elementwise_mul        "scale: gamma(C) * lnorm"
 *              |
 *        beta  |
 *          \   |
 *       elementwise_add        "shift: gamma(C) * lnorm + beta(C)"
 */
class LayerNormFusePass : public FusePassBase {
 public:
  LayerNormFusePass();
  virtual ~LayerNormFusePass() {}

 protected:
  void ApplyImpl(ir::Graph *graph) const override;

 private:
  const std::string scope_name_{"layer_norm_fuse"};
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
