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

#include <gtest/gtest.h>

#include "paddle/cinn/ir/ir.h"

namespace cinn {
namespace auto_schedule {

std::vector<ir::Expr> CreateTestBlocks() {
  std::vector<ir::Expr> blocks;
  for (int i = 0; i < 3; ++i) {
    ir::Expr block = ir::ScheduleBlock::Make(
        {}, {}, {}, "block_" + std::to_string(i), ir::Expr());
    blocks.push_back(ir::ScheduleBlockRealize::Make({}, block));
  }
  return blocks;
}

TEST(BlockSampler, Make) {
  std::vector<ir::Expr> mock_blocks = CreateTestBlocks();
  auto traversal_block_sampler =
      BlockSampler::Make(mock_blocks, true, "traversal");
  ASSERT_STREQ(traversal_block_sampler->Name(), "traversal");
  auto probabilistic_block_sampler =
      BlockSampler::Make(mock_blocks, true, "probabilistic");
  ASSERT_STREQ(probabilistic_block_sampler->Name(), "probabilistic");
}

TEST(TraversalBlockSampler, NextBlock) {
  std::vector<ir::Expr> blocks = CreateTestBlocks();
  auto traversal_block_sampler = BlockSampler::Make(blocks, true, "traversal");
  ASSERT_EQ("block_0", traversal_block_sampler->NextBlock());
  ASSERT_EQ("block_1", traversal_block_sampler->NextBlock());
  ASSERT_EQ("block_2", traversal_block_sampler->NextBlock());
  ASSERT_EQ("", traversal_block_sampler->NextBlock());
  traversal_block_sampler->Reset();
  ASSERT_EQ("block_0", traversal_block_sampler->NextBlock());

  traversal_block_sampler = BlockSampler::Make(blocks, false, "traversal");
  ASSERT_EQ("block_0", traversal_block_sampler->NextBlock());
  ASSERT_EQ("block_0", traversal_block_sampler->NextBlock());
}

TEST(ProbabilisticBlockSampler, NextBlock) {
  std::vector<ir::Expr> blocks = CreateTestBlocks();
  auto probabilistic_block_sampler =
      BlockSampler::Make(blocks, false, "probabilistic", 0, {4, 2, 1});
  std::string block_name;
  for (int i = 0; i < 20; ++i) {
    block_name = probabilistic_block_sampler->NextBlock();
    VLOG(6) << "next block name: " << block_name;
  }

  probabilistic_block_sampler =
      BlockSampler::Make(blocks, true, "probabilistic", 0, {4, 2, 1});
  probabilistic_block_sampler->NextBlock();
  probabilistic_block_sampler->NextBlock();
  probabilistic_block_sampler->NextBlock();
  ASSERT_EQ("", probabilistic_block_sampler->NextBlock());
}

}  // namespace auto_schedule
}  // namespace cinn
