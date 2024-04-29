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

#include <memory>
#include <random>
#include <vector>

#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/utils/random_engine.h"

namespace cinn {
namespace auto_schedule {

class SearchState;

// Select the next block to be operated for SearchState during the search
// process
class BlockSampler {
 public:
  /**
   * @brief Create a BlockSampler with the specific strategy name and necessary
   * construct parameters.
   * @param all_blocks All possible blocks to be sampled.
   * @param default_remove_policy The default option to determine whether to
   * delete the next block after selecting it.
   * @param strategy The block sampling strategy.
   *                 Currently, the available strategies are "traversal" and
   * "probabilistic", where "traversal" means to select blocks one by one until
   * all blocks are traversed, and "probabilistic" means randomly picking blocks
   * according to the given distribution.
   * @param weights Used for the probabilistic policy, giving each candidate a
   * weight.
   */
  static std::unique_ptr<BlockSampler> Make(
      const std::vector<ir::Expr>& all_blocks,
      bool default_remove_policy = true,
      const std::string& strategy = "traversal",
      utils::LinearRandomEngine::StateType rand_seed = 0,
      const std::vector<int>& weights = {});

  // Return the name of sample strategy
  virtual const char* Name() const = 0;

  // Reset associated states to sample at the beginning
  virtual void Reset() = 0;

  // Select a block with default remove policy.
  std::string NextBlock() { return NextBlock(default_remove_policy_); }

 protected:
  // A BlockSampler object should be created with the static function Make()
  BlockSampler(const std::vector<ir::Expr>& all_blocks,
               bool default_remove_policy);

  // Select a block to apply rule
  // The param remove is used to determine whether to delete the next block
  // after selecting it, If remove == true, it will not be sampled in the
  // future.
  virtual std::string NextBlock(bool remove) = 0;

  // The names of all blocks
  // Because the Block Expr will be changed in the search process, the name is
  // saved for indexing
  std::vector<std::string> all_blocks_;

  // The default policy to determine whether to delete the next block after
  // selecting it.
  bool default_remove_policy_;
};

// Sample blocks with traversal strategy,
// witch means to select blocks one by one until all blocks are traversed.
class TraversalBlockSampler : public BlockSampler {
 public:
  TraversalBlockSampler(const std::vector<ir::Expr>& all_blocks,
                        bool default_remove_policy)
      : BlockSampler(all_blocks, default_remove_policy), cur_idx_(0) {}

  const char* Name() const override { return "traversal"; }

  void Reset() override { cur_idx_ = 0; }

 private:
  std::string NextBlock(bool remove) override;

 private:
  int cur_idx_;
};

// Sample blocks with probabilistic strategy,
// witch means randomly picking blocks according to the given distribution.
class ProbabilisticBlockSampler : public BlockSampler {
 public:
  ProbabilisticBlockSampler(const std::vector<ir::Expr>& all_blocks,
                            bool default_remove_policy,
                            utils::LinearRandomEngine::StateType rand_seed = 0,
                            const std::vector<int>& weights = {});

  const char* Name() const override { return "probabilistic"; }

  void Reset() override {}

 private:
  std::string NextBlock(bool remove) override;

 private:
  std::vector<int> weights_;
  utils::LinearRandomEngine::StateType rand_seed_;
  int remains_;
};

}  // namespace auto_schedule
}  // namespace cinn
