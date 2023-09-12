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

#include "paddle/cinn/auto_schedule/cost_model/feature.h"

#include <glog/logging.h>

#include <vector>

#include "paddle/cinn/common/target.h"

namespace cinn {
namespace auto_schedule {

Feature::Feature()
    : target_(common::UnkTarget()),
      stack_encoded_feature_(1),  // initialize a LoopBlockFeature as root block
      current_loop_block_index_(0),
      parent_indices_(1, -1) {}

Feature::Feature(const common::Target& target)
    : target_(target),
      stack_encoded_feature_(1),  // initialize a LoopBlockFeature as root block
      current_loop_block_index_(0),
      parent_indices_(1, -1) {}

std::vector<float> Feature::ToFixedSizeVector() {
  std::vector<float> ret(LoopBlockFeature::kTotalSize + 1,
                         0);  // LoopBlockFeature::kTotalSize plus 1 for target

  if (target_ == common::DefaultNVGPUTarget()) {
    ret[0] = 1;
  }  // else 0 for other cases

  // loop[i] feature count should multiply iter_multi_num[i]
  std::vector<int> iter_multi_num;
  for (size_t i = 0; i < stack_encoded_feature_.size(); ++i) {
    int j = 1;
    const LoopBlockFeature& loop_feature = stack_encoded_feature_[i];
    int loop_prod = 1;
    int parent_prod = 1;
    if (i != 0) {
      parent_prod = iter_multi_num[parent_indices_[i]];
      loop_prod = parent_prod * loop_feature.loop_length;
    }
    iter_multi_num.push_back(loop_prod);

    ret[j] += (loop_feature.float_add_or_sub * loop_prod);
    ++j;
    ret[j] += (loop_feature.float_mul * loop_prod);
    ++j;
    ret[j] += (loop_feature.float_div_or_mod * loop_prod);
    ++j;
    ret[j] += (loop_feature.float_cmp * loop_prod);
    ++j;
    ret[j] += (loop_feature.float_math_func * loop_prod);
    ++j;
    ret[j] += (loop_feature.float_other_call * loop_prod);
    ++j;

    ret[j] += (loop_feature.int_add_or_sub * loop_prod);
    ++j;
    ret[j] += (loop_feature.int_mul * loop_prod);
    ++j;
    ret[j] += (loop_feature.int_div_or_mod * loop_prod);
    ++j;
    ret[j] += (loop_feature.int_cmp * loop_prod);
    ++j;
    ret[j] += (loop_feature.int_math_func * loop_prod);
    ++j;
    ret[j] += (loop_feature.int_other_call * loop_prod);
    ++j;

    ret[j] += (loop_feature.bool_op * loop_prod);
    ++j;
    ret[j] += (loop_feature.select_op * loop_prod);
    ++j;

    ret[j] += (loop_feature.mem_alloc * loop_prod);
    ++j;
    ret[j] += (loop_feature.mem_free * loop_prod);
    ++j;
    ret[j] += (loop_feature.mem_read * loop_prod);
    ++j;
    ret[j] += (loop_feature.mem_write * loop_prod);
    ++j;

    ret[j] += (loop_feature.float_reduce_sum_or_sub * loop_prod);
    ++j;
    ret[j] += (loop_feature.float_reduce_mul * loop_prod);
    ++j;
    ret[j] += (loop_feature.float_reduce_div * loop_prod);
    ++j;
    ret[j] += (loop_feature.float_reduce_max_or_min * loop_prod);
    ++j;
    ret[j] += (loop_feature.float_broadcast * loop_prod);
    ++j;

    ret[j] += (loop_feature.int_reduce_sum_or_sub * loop_prod);
    ++j;
    ret[j] += (loop_feature.int_reduce_mul * loop_prod);
    ++j;
    ret[j] += (loop_feature.int_reduce_div * loop_prod);
    ++j;
    ret[j] += (loop_feature.int_reduce_max_or_min * loop_prod);
    ++j;
    ret[j] += (loop_feature.int_broadcast * loop_prod);
    ++j;

    ret[j + static_cast<int>(loop_feature.loop_opt_type)] += 1;
    j += LoopBlockFeature::kOptApplySize;

    ret[j] += (loop_feature.len_blockIdx_x * parent_prod);
    ++j;
    ret[j] += (loop_feature.len_blockIdx_y * parent_prod);
    ++j;
    ret[j] += (loop_feature.len_blockIdx_z * parent_prod);
    ++j;
    ret[j] += (loop_feature.len_threadIdx_x * parent_prod);
    ++j;
    ret[j] += (loop_feature.len_threadIdx_y * parent_prod);
    ++j;
    ret[j] += (loop_feature.len_threadIdx_z * parent_prod);
    ++j;
    ret[j] += (loop_feature.len_vthread * parent_prod);
    ++j;
    ret[j] += (loop_feature.vectorize_factor * parent_prod);
    ++j;
  }

  for (size_t i = 0; i < ret.size(); ++i) {
    ret[i] = slog(ret[i]);
  }

  return ret;
}

void Feature::IntoLoopBlock() {
  stack_encoded_feature_.emplace_back(LoopBlockFeature());
  stack_encoded_feature_[current_loop_block_index_].num_sub_loops += 1;
  parent_indices_.push_back(current_loop_block_index_);
  current_loop_block_index_ = stack_encoded_feature_.size() - 1;
}

void Feature::ExitLoopBlock() {
  current_loop_block_index_ = parent_indices_[current_loop_block_index_];
}

LoopBlockFeature& Feature::CurrentLoopBlock() {
  return stack_encoded_feature_[current_loop_block_index_];
}

const LoopBlockFeature& Feature::CurrentLoopBlock() const {
  return stack_encoded_feature_[current_loop_block_index_];
}

}  // namespace auto_schedule
}  // namespace cinn
