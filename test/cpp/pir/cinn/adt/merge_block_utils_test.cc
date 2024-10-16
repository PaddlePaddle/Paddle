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

#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/optim/merge_block_utils.h"

namespace cinn {
namespace optim {

namespace {

bool IsBlockForAllEqual(const ForTreeNode& first, const ForTreeNode& second) {
  auto ForVarExtentEqual = [&](const ForTreeNode& first,
                               const ForTreeNode& second) -> bool {
    const ir::Expr lhs = first.val->extent;
    const ir::Expr rhs = second.val->extent;
    if (cinn::common::AutoSimplify(ir::Sub::Make(lhs, rhs)) != ir::Expr(0)) {
      return false;
    }
    return true;
  };

  if (!ForVarExtentEqual(first, second)) return false;
  if (first.children.size() != second.children.size()) return false;
  for (size_t i = 0; i < first.children.size(); ++i) {
    if (!IsBlockForAllEqual(first.children[i], second.children[i])) {
      return false;
    }
  }

  return true;
}

ir::Expr MakeForLoops(const std::vector<int> extents, int index) {
  if (index >= extents.size()) {
    ir::Expr sb = ir::ScheduleBlock::Make(std::vector<Var>(),
                                          std::vector<Expr>(),
                                          std::vector<Expr>(),
                                          "block",
                                          ir::Expr(0));
    return sb;
  }

  ir::Expr extent = ir::Expr(extents.at(index));
  ir::Expr for_expr = ir::For::Make(ir::Var("i"),
                                    ir::Expr(0),
                                    extent,
                                    ir::ForType::Serial,
                                    ir::DeviceAPI::CUDA,
                                    MakeForLoops(extents, index + 1),
                                    ir::VectorizeInfo(),
                                    ir::BindInfo());

  return for_expr;
}

void TestHelper(const std::vector<int>& extents1,
                const std::vector<int>& extents2,
                bool is_same) {
  auto for_loop1 = MakeForLoops(extents1, 0);
  auto for_loop2 = MakeForLoops(extents2, 0);
  auto f1 = for_loop1.As<ir::For>();
  auto f2 = for_loop2.As<ir::For>();

  if (is_same) {
    EXPECT_TRUE(CanMergeBlocks(f1, f2, IsBlockForAllEqual));
  } else {
    EXPECT_FALSE(CanMergeBlocks(f1, f2, IsBlockForAllEqual));
  }
}

void TestHelper2(const std::vector<std::vector<int>>& extents1,
                 const std::vector<std::vector<int>>& extents2,
                 bool is_same) {
  auto MakeNestLoops =
      [&](const std::vector<std::vector<int>>& extents) -> ir::Expr {
    std::vector<ir::Expr> for_loops;
    for (size_t i = 0; i < extents.size(); ++i) {
      for_loops.push_back(MakeForLoops(extents[i], 0));
    }
    ir::Expr block = ir::Block::Make(for_loops);
    ir::Expr for_expr = ir::For::Make(ir::Var("i"),
                                      ir::Expr(0),
                                      ir::Expr(1),
                                      ir::ForType::Serial,
                                      ir::DeviceAPI::CUDA,
                                      block,
                                      ir::VectorizeInfo(),
                                      ir::BindInfo());
    return for_expr;
  };

  auto for_expr1 = MakeNestLoops(extents1);
  auto for_expr2 = MakeNestLoops(extents2);
  auto f1 = for_expr1.As<ir::For>();
  auto f2 = for_expr2.As<ir::For>();

  if (is_same) {
    EXPECT_TRUE(CanMergeBlocks(f1, f2, IsBlockForAllEqual));
  } else {
    EXPECT_FALSE(CanMergeBlocks(f1, f2, IsBlockForAllEqual));
  }
}

TEST(ForInfo, ForInfoEqual) {
  TestHelper({10}, {10}, true);
  TestHelper({10, 5}, {10, 5}, true);
  TestHelper({10, 5, 3}, {10, 5, 3}, true);

  TestHelper2({{10}, {10}}, {{10}, {10}}, true);
  TestHelper2({{10, 5}, {4, 7}}, {{10, 5}, {4, 7}}, true);
  TestHelper2(
      {{10, 5, 3}, {4, 7, 9}, {2, 8}}, {{10, 5, 3}, {4, 7, 9}, {2, 8}}, true);
}

TEST(ForInfo, ForInfoNotEqual) {
  TestHelper({10}, {9}, false);
  TestHelper({10, 5}, {10, 4}, false);
  TestHelper({10, 5, 3}, {10, 5, 2}, false);

  TestHelper2({{10}, {10}}, {{10}, {9}}, false);
  TestHelper2({{10, 5}, {4, 7}}, {{10, 5}, {4, 3}}, false);
  TestHelper2(
      {{10, 5, 3}, {4, 7, 9}, {2, 8}}, {{10, 5, 3}, {4, 7, 9}, {2, 7}}, false);
}

}  // namespace

}  // namespace optim
}  // namespace cinn
