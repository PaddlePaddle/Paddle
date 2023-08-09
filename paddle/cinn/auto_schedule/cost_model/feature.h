// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include <cmath>
#include <vector>

#include "paddle/cinn/common/target.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"

namespace cinn {
namespace auto_schedule {

/* Loop feature enums */
enum class ForOptimizeFeatureEnum : int {
  kNone,
  kGpuBind,
  kParallel,
  kUnroll,
  kVectorize
};

/* function to scale feature numbers */
inline float slog(float x) {
  return x < 0 ? std::log2(-x + 1) : std::log2(x + 1);
}

class LoopBlockFeature {
 public:
  // TODO(zhhsplendid): distinguish more types such as float16, float32,
  // float64, etc. However speed the gap between float and int are larger than
  // different bits, so we just distinguished int and float here
  /* Arithmetic features */
  int float_add_or_sub = 0;
  int float_mul = 0;
  int float_div_or_mod = 0;
  int float_cmp = 0;
  int float_math_func = 0;
  int float_other_call = 0;  // like simple assign, cast, etc.

  int int_add_or_sub = 0;
  int int_mul = 0;
  int int_div_or_mod = 0;
  int int_cmp = 0;
  int int_math_func = 0;
  int int_other_call = 0;  // like simple assign, cast, etc.

  int bool_op = 0;
  int select_op = 0;

  static constexpr int kArithSize = 6 * 2 + 2;

  /**
   * Buffer memory features, which is the number of memory operations.
   * Note that different size of memory operation can have various speed,
   * however the speed difference would be small in OS. A meticulous TODO
   * may be collect operand sizes (like alloc size, write size, or so)
   */
  int mem_alloc = 0;
  int mem_free = 0;
  int mem_read = 0;
  int mem_write = 0;

  static constexpr int kMemSize = 4;

  /**
   * Reduce and Broadcast features
   */
  int float_reduce_sum_or_sub = 0;
  int float_reduce_mul = 0;
  int float_reduce_div = 0;
  int float_reduce_max_or_min = 0;
  int float_broadcast = 0;

  int int_reduce_sum_or_sub = 0;
  int int_reduce_mul = 0;
  int int_reduce_div = 0;
  int int_reduce_max_or_min = 0;
  int int_broadcast = 0;

  static constexpr int kReduceBroadcastSize = 10;

  /* Loop type features */

  // A TODO maybe add loop position (Inner, Outer, Middle) feature

  ForOptimizeFeatureEnum loop_opt_type = ForOptimizeFeatureEnum::kNone;

  static constexpr int kOptApplySize = 5;

  /* Thread features if loop is optimized by GPU or CPU parallelism.
   * Useless in other cases.
   */
  int len_blockIdx_x = 0;
  int len_blockIdx_y = 0;
  int len_blockIdx_z = 0;
  int len_threadIdx_x = 0;
  int len_threadIdx_y = 0;
  int len_threadIdx_z = 0;
  int len_vthread = 0;  // length of virtual thread
  int vectorize_factor = 0;

  static constexpr int kThreadFeatureSize = 8;

  static constexpr int kTotalSize = kArithSize + kMemSize +
                                    kReduceBroadcastSize + kOptApplySize +
                                    kThreadFeatureSize;

  /* Non-feature attributes, used to maintain during feature_extractor */

  // Number to indicate the loop block inside current one
  int num_sub_loops = 0;

  // Number of repeats of this loop, -1 represents unknown
  int loop_length = 1;
};

/**
 * Feature of Expr. It is used in CostModel
 */
class Feature {
 public:
  Feature();

  explicit Feature(const common::Target& target);

  // Convert the various-length loop block features to fixed-size vector
  std::vector<float> ToFixedSizeVector();

  // Call when visit into a loop block to collect LoopBlockFeature
  void IntoLoopBlock();
  // Call when exit a loop block to collect LoopBlockFeature
  void ExitLoopBlock();
  // The current loop block which we should collect feature on
  LoopBlockFeature& CurrentLoopBlock();
  // The current loop block which we should collect feature on
  const LoopBlockFeature& CurrentLoopBlock() const;

 private:
  // We treat a computation feature to be encoded as variable-length vector.
  // The root compute block is not a loop, but we treat it as a size-1 loop.
  // Blocks are encoded like a stack. Each LoopBlockFeature contains a
  // num_sub_loops to indicate the next level sub-loop-block it contains.
  //
  // For example, code like:
  //
  // some_compute_0
  // loop1 {
  //   some_compute_1
  //   loop2 {
  //     some_compute_2
  //   }
  // }
  //
  // loop3 {
  //   some_compute_3
  // }
  //
  // We go through the code and push loops into stack, then the features are
  // encoded as [loop_block_feature_0, loop_block_feature_1,
  // loop_block_feature_2, loop_block_feature_3] where loop_block_feature_i
  // stores the features of some_compute_i (such as number of arithmetic
  // operations)
  //
  // loop_block_feature_0.num_sub_loops = 2
  // loop_block_feature_1.num_sub_loops = 1
  // loop_block_feature_2.num_sub_loops = 0
  // loop_block_feature_3.num_sub_loops = 0
  std::vector<LoopBlockFeature> stack_encoded_feature_;
  int current_loop_block_index_;
  std::vector<int> parent_indices_;

  common::Target target_;
};

}  // namespace auto_schedule
}  // namespace cinn
