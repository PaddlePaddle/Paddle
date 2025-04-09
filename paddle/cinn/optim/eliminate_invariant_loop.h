// Copyright (c) 2024 CINN Authors. All Rights Reserved.
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
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/lowered_func.h"

namespace cinn {
namespace optim {

/**
 * Eliminate serial or bound loops that have a loop-invariant body.
 *
 * We can eliminate a serial loop if it satisfies rule (1) and (2):
 * (1) The loop variable is not used in any load/store indices or computation
 *     within child schedule blocks. This ensures that the loop writes the same
 *     value to the same index in each iteration.
 * (2) It is not a Reduce (e.g. a[0] = a[0] + b[k]), because for a Reduce, even
 *     though its indices don't change in each iteration, its value changes.
 *
 * We can eliminate a bound loop if it also satisfies rule (3) and (4):
 * (3) It doesn't write to the local buffer (for thread-bound loop) or shared
 *     memory (for block-bound loop), because the index for these storages
 *     implicitly includes the thread/block index, thereby violating (1).
 * (4) Its child schedule blocks don't have consumers, otherwise consumers in
 *     other blocks/threads may read an undefined value.
 *
 * When a loop can be eliminated, we:
 * (a) For serial loop, we just replace the For node with its body, and replace
 *     all use of the loop variable to 0.
 * (b) For bound loop, we cannot remove the For node because it contains the
 *     binding info. Instead, we warp the loop body in `if (loop_var == 0)`.
 *
 *
 * Example 1:
 *   serial for (k, 0, 32):
 *     ScheduleBlock(A):
 *       A[0] = B[0]
 * =>
 *   ScheduleBlock(A):
 *     A[0] = B[0]
 *
 *
 * Example 2:
 *   bind for (threadIdx.x, 0, 32):
 *     ScheduleBlock(A):
 *       A[0] = B[0]  # assume that A is global
 * =>
 *   bind for (threadIdx.x, 0, 32):
 *     if threadIdx.x == 0:
 *       ScheduleBlock(A):
 *         A[0] = B[0]
 * Note:
 *   We don't remove a bound loop, but wrap its body in `threadIdx.x == 0`.
 *
 *
 * Example 3:
 *   serial for (k, 0, 32):
 *     ScheduleBlock(A):
 *       A[0] = A[0] + B[0]
 * =>
 *   DO NOTHING!
 * Note:
 *   Even though k is not used in the body, as ScheduleBlock(A) contains an
 *   inplace operation, each iteration updates A, so we cannot eliminate it.
 *
 *
 * Example 4:
 *   bind for (blockIdx.x, 0, 32):
 *     ScheduleBlock(A):
 *       A[0] = 3.14
 *   bind for (blockIdx.x, 0, 32):
 *     ScheduleBlock(B):
 *       B[blockIdx.x] = A[0]
 * =>
 *   DO NOTHING!
 * Note:
 *   If we eliminate the For node of A, only the block at `blockIdx.x == 0`
 *   can get the correct value of A[0] when computing B.
 *
 *
 * Example 5:
 *   serial for (k, 0, 32):
 *     if (k * 256) + threadIdx.x < 8000:
 *       ScheduleBlock(A):
 *         A[threadIdx.x] = B[threadIdx.x]
 * =>
 *   if threadIdx.x < 8000:
 *     ScheduleBlock(A):
 *       A[threadIdx.x] = B[threadIdx.x]
 * Note:
 *   Although k is used in the if condition, it doesn't affect the index to
 *   update, but just limits how many times to update. So rewriting k to 0
 *   doesn't change the semantics.
 */
void EliminateInvariantLoop(ir::Expr *expr);

}  // namespace optim
}  // namespace cinn
