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

#include "paddle/cinn/auto_schedule/search_space/block_sampler.h"

#include <algorithm>

#include "paddle/cinn/ir/ir.h"

namespace cinn {
namespace auto_schedule {

std::unique_ptr<BlockSampler> BlockSampler::Make(
    const std::vector<ir::Expr>& all_blocks,
    bool default_remove_policy,
    const std::string& strategy,
    utils::LinearRandomEngine::StateType rand_seed,
    const std::vector<int>& weights) {
  CHECK_GT(all_blocks.size(), 0) << "Empty block list";
  if (strategy == "traversal") {
    VLOG(6) << "Init TraversalBlockSampler with block num = "
            << all_blocks.size();
    return std::make_unique<TraversalBlockSampler>(all_blocks,
                                                   default_remove_policy);
  } else if (strategy == "probabilistic") {
    VLOG(6) << "Init ProbabilisticBlockSampler with block num = "
            << all_blocks.size();
    return std::make_unique<ProbabilisticBlockSampler>(
        all_blocks, default_remove_policy, rand_seed, weights);
  }

  LOG(FATAL) << "Unimplemented strategy:" << strategy;
  return nullptr;
}

BlockSampler::BlockSampler(const std::vector<ir::Expr>& all_blocks,
                           bool default_remove_policy) {
  default_remove_policy_ = default_remove_policy;
  std::transform(all_blocks.begin(),
                 all_blocks.end(),
                 std::back_inserter(all_blocks_),
                 [](const ir::Expr& block_expr) {
                   const ir::ScheduleBlockRealize* block_realize =
                       block_expr.As<ir::ScheduleBlockRealize>();
                   const ir::ScheduleBlock* block =
                       block_realize->schedule_block.As<ir::ScheduleBlock>();
                   return block->name;
                 });
}

std::string TraversalBlockSampler::NextBlock(bool remove) {
  if (cur_idx_ < all_blocks_.size()) {
    VLOG(6) << "[TraversalBlockSampler] next block: "
            << all_blocks_.at(cur_idx_);
    std::string block_name = all_blocks_.at(cur_idx_);
    if (remove) {
      ++cur_idx_;
    }
    return block_name;
  }

  VLOG(6) << "[TraversalBlockSampler] next block: empty";
  return "";
}

ProbabilisticBlockSampler::ProbabilisticBlockSampler(
    const std::vector<ir::Expr>& all_blocks,
    bool default_remove_policy,
    utils::LinearRandomEngine::StateType rand_seed,
    const std::vector<int>& weights)
    : BlockSampler(all_blocks, default_remove_policy),
      weights_(weights),
      rand_seed_(rand_seed) {
  if (weights.empty()) {
    weights_.resize(all_blocks.size(), 1);
  } else {
    CHECK_EQ(all_blocks.size(), weights_.size());
  }
  remains_ = all_blocks.size();
}

std::string ProbabilisticBlockSampler::NextBlock(bool remove) {
  if (remains_ == 0) {
    return "";
  }
  int block_idx =
      utils::SampleDiscreteFromDistribution<int>(weights_, &rand_seed_);
  if (remove) {
    weights_[block_idx] = 0;
    --remains_;
  }
  VLOG(6) << "[ProbabilisticBlockSampler] next block: "
          << all_blocks_.at(block_idx);
  return all_blocks_.at(block_idx);
}

}  // namespace auto_schedule
}  // namespace cinn
