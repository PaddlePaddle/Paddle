// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <sstream>

#include "paddle/cinn/optim/merge_block_utils.h"

namespace cinn {
namespace optim {

namespace {

std::vector<ir::Expr> MakeFor(const std::vector<int> extents) {
  ir::Expr sb = ir::ScheduleBlock::Make(std::vector<Var>(),
                                        std::vector<Expr>(),
                                        std::vector<Expr>(),
                                        "block",
                                        ir::Expr(0));
  std::vector<ir::Expr> for_loops;
  for (size_t i = 0; i < extents.size(); ++i) {
    ir::Expr extent = ir::Expr(extents.at(i));
    ir::Expr for_expr = ir::For::Make(ir::Var("i"),
                                      ir::Expr(0),
                                      extent,
                                      ir::ForType::Serial,
                                      ir::DeviceAPI::CUDA,
                                      sb,
                                      ir::VectorizeInfo(),
                                      ir::BindInfo());
    for_loops.push_back(for_expr);
  }

  return for_loops;
}

std::vector<ir::For*> ConvertForLoops(std::vector<ir::Expr> loops) {
  std::vector<ir::For*> p_for_loops;
  for (auto& loop : loops) {
    p_for_loops.push_back(loop.As<ir::For>());
  }
  return p_for_loops;
}

TEST(ForInfo, ForInfoCheckerEqual) {
  std::vector<ir::Expr> for_loop1;
  std::vector<ir::Expr> for_loop2;
  std::vector<ir::For*> f1;
  std::vector<ir::For*> f2;

  for_loop1 = MakeFor({10});
  for_loop2 = MakeFor({10});
  f1 = ConvertForLoops(for_loop1);
  f2 = ConvertForLoops(for_loop2);
  EXPECT_TRUE(CanMergeBlocks(f1, f2));

  for_loop1 = MakeFor({10, 5});
  for_loop2 = MakeFor({10, 5});
  f1 = ConvertForLoops(for_loop1);
  f2 = ConvertForLoops(for_loop2);
  EXPECT_TRUE(CanMergeBlocks(f1, f2));

  for_loop1 = MakeFor({10, 5, 3});
  for_loop2 = MakeFor({10, 5, 3});
  f1 = ConvertForLoops(for_loop1);
  f2 = ConvertForLoops(for_loop2);
  EXPECT_TRUE(CanMergeBlocks(f1, f2));
}

TEST(ForInfo, ForInfoCheckerNotEqual) {
  std::vector<ir::Expr> for_loop1;
  std::vector<ir::Expr> for_loop2;
  std::vector<ir::For*> f1;
  std::vector<ir::For*> f2;

  for_loop1 = MakeFor({10});
  for_loop2 = MakeFor({9});
  f1 = ConvertForLoops(for_loop1);
  f2 = ConvertForLoops(for_loop2);
  EXPECT_FALSE(CanMergeBlocks(f1, f2));

  for_loop1 = MakeFor({10, 5});
  for_loop2 = MakeFor({10, 2});
  f1 = ConvertForLoops(for_loop1);
  f2 = ConvertForLoops(for_loop2);
  EXPECT_FALSE(CanMergeBlocks(f1, f2));

  for_loop1 = MakeFor({10, 5, 3});
  for_loop2 = MakeFor({10, 5, 1});
  f1 = ConvertForLoops(for_loop1);
  f2 = ConvertForLoops(for_loop2);
  EXPECT_FALSE(CanMergeBlocks(f1, f2));
}

}  // namespace

}  // namespace optim
}  // namespace cinn
