// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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
#include <algorithm>
#include <unordered_set>
#include <utility>

#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/lowered_func.h"
#include "paddle/cinn/poly/isl_utils.h"
#include "paddle/cinn/poly/stage.h"

namespace cinn {
namespace optim {

void OptimizeExprGPU(Expr* expr);
/*
  // replace 'for' loop to gpu 'block/thread'
  // update buffer index to save memory size.
  // re-compute buffer size.
*/

/**
 * Remove the forloops of block and thread axis, add the kernel launch thread
 * dimension information to the outermost LoweredFunc.
 *
 * For example, input the code:
 * \code
 * // Note here, the outermost expression should be a LoweredFunc
 * _LoweredFunc_:
 *   for (blockIdx.x, 0, 10)
 *     for (threadIdx.x, 0, 20)
 *       A(blockIdx.x, threadIdx.x)
 * \endcode
 *
 * will be modified to
 * \code
 * _LoweredFunc_<blockDim:10, threadDim:20>:
 *   A(blockIdx.x, threadIdx.x)
 * \endcode
 *
 * \note For that the dimensions of each threadIdx or blockIdx should be
 * constant, so this only takes For nodes, not \note PolyFor nodes is allowed to
 * be GPU related.
 */
void RemoveGpuForloopsAxis(Expr* expr);

/**
 * Add __syncthreads() to shared memory producer.
 */
void CudaSyncThreadsDropIfThenElse(Expr* expr);

}  // namespace optim
}  // namespace cinn
