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
 * The former version of this file is realize_welford_pass.h
 * the updated version intends to generalize this pass, to support more
 * customized reduced pass
 *
 * Here, we use a reduce graph to explain what this pass does:
 *
 * Input IR:
 * function fn_customized_reduce(const float* var_0, target_type* var_2) {
 *   float var_1_rf [ 1 ]
 *   float var_1 [ 1 ]
 *   for (thread.x, 0, 256) {
 *     var_1_rf[0] = 0.0f
 *     for (k, 0, 32) {
 *       var_1_rf[0] = reduce_op(var_1_rf[0],
 *                                          var_0[k * 256 + thread.x])
 *     }
 *   }
 *   for (thread.x, 0, 256) {
 *     var_1[0] = reduce_op(var_1[0], var_1_rf[0])
 *   }
 *   var_2[0] = var_1[0]
 * }
 *
 * Output IR:
 * function fn_customized_reduce(const float* var_0, target_type* var_2) {
 *   welford_fp32 var_1_rf [ 1 ]
 *   welford_fp32 var_1 [ 1 ]
 *   for (thread.x, 0, 256) {
 *     var_1_rf[0] = welford_fp32(0.0f, 0.0f, 0.0f)
 *     for (k, 0, 32) {
 *       var_1_rf[0] = transformed_reduced_op(
 *                var_1_rf[0],
 *                (customized_type)var_0[k * 256 + thread.x]
 *       )
 *     }
 *   }
 *   for (thread.x, 0, 256) {
 *     var_1[0] = var_1[0] + var_1_rf[0]
 *   }
 *   var_2[0] = (target_type)var_1[0]
 * }
 *
 * This pass applies the following changes to the graph:
 * 1) Change the intermediate values of reduce computation (`var_1` and
 *    `var_1_rf`) to their corresponding customized type, for example:.
 *    `welford_fp32` for Welford variance, and `argidx_fp32_i32` for argmin/max
 *    Note that the types of the function arguments (`var_0` and `var_2`) are
 *    not changed at all.
 * 2) Replace the `reduce_op` call with a simple `transformed_reduced_op`. for
 * example:
 *  - welford: operator+ is implemented by C++ operator overloading.
 *  - arg reduce: max, min op does not need to be replaced due to C++ operator
 * overloading 3) Add casts at the beginning of reduce computation (casting
 * `var_0` to `welford_fp32`) and at the end (casting `var_1` to `target type`).
 */
class RealizeCompositeReducePass : public FuncPass {
 public:
  explicit RealizeCompositeReducePass(Target target)
      : FuncPass("realize_composite_reduce"), target_(target) {}
  LogicalResult Run(ir::LoweredFunc func) override;

 private:
  const Target target_;
};

std::unique_ptr<FuncPass> CreateRealizeCompositeReducePass(Target target);

}  // namespace optim
}  // namespace cinn
