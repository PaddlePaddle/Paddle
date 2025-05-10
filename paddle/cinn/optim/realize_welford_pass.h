// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/cinn/pass/pass.h"

namespace cinn {
namespace optim {

/**
 * Realize the variance computation with the Welford algorithm.
 *
 * This pass realize the `cinn_reduce_variance` op using the highly-accurate
 * Welford algorithm. The input graph is like a normal reduce with a virtual
 * reduce op `cinn_reduce_variance`. After this pass, the operators and data
 * types will be properly set to adapt to the Welford algorithm.
 *
 * Here, we use a reduce graph to explain what this pass does:
 *
 * Input IR:
 * function fn_variance(const float* var_0, float* var_2) {
 *   float var_1_rf [ 1 ]
 *   float var_1 [ 1 ]
 *   for (thread.x, 0, 256) {
 *     var_1_rf[0] = 0.0f
 *     for (k, 0, 32) {
 *       var_1_rf[0] = cinn_reduce_variance(var_1_rf[0],
 *                                          var_0[k * 256 + thread.x])
 *     }
 *   }
 *   for (thread.x, 0, 256) {
 *     var_1[0] = cinn_reduce_variance(var_1[0], var_1_rf[0])
 *   }
 *   var_2[0] = var_1[0]
 * }
 *
 * Output IR:
 * function fn_variance(const float* var_0, float* var_2) {
 *   welford_fp32 var_1_rf [ 1 ]
 *   welford_fp32 var_1 [ 1 ]
 *   for (thread.x, 0, 256) {
 *     var_1_rf[0] = welford_fp32(0.0f, 0.0f, 0.0f)
 *     for (k, 0, 32) {
 *       var_1_rf[0] = var_1_rf[0] + (welford_fp32)var_0[k * 256 + thread.x]
 *     }
 *   }
 *   for (thread.x, 0, 256) {
 *     var_1[0] = var_1[0] + var_1_rf[0]
 *   }
 *   var_2[0] = (float)var_1[0]
 * }
 *
 * This pass applies the following changes to the graph:
 * 1) Change the intermediate values of Welford computation (`var_1` and
 *    `var_1_rf`) to their corresponding Welford type (`welford_fp32` here).
 *    Note that the types of the function arguments (`var_0` and `var_2`) are
 *    not changed at all.
 * 2) Replace the `cinn_reduce_variance` call with a simple `operator+`, which
 *    is implemented by C++ operator overloading. This makes reduce templates
 *    replacement (the next pass) easier.
 * 3) Add casts at the beginning of Welford computation (casting `var_0` to
 *    `welford_fp32`) and at the end (casting `var_1` back to `float`).
 */
class RealizeWelfordPass : public FuncPass {
 public:
  RealizeWelfordPass() : FuncPass("realize_welford") {}
  LogicalResult Run(ir::LoweredFunc func) override;
};

std::unique_ptr<FuncPass> CreateRealizeWelfordPass();

}  // namespace optim
}  // namespace cinn
