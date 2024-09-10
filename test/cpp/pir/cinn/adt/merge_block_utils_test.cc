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
#include "test/cpp/pir/tools/test_pir_utils.h"

namespace cinn {
namespace optim {

namespace {

ir::Expr MakeFor(const std::vector<int> extents,
                 const std::string& name,
                 const int index) {
  if (index == extents.size()) {
    ir::Expr expr = ir::Expr(0);
    ir::Expr sb = ir::ScheduleBlock::Make(std::vector<Var>(),
                                          std::vector<Expr>(),
                                          std::vector<Expr>(),
                                          name,
                                          expr);
    return sb;
  }
  ir::Expr min = ir::Expr(0);
  ir::Expr extent = ir::Expr(extents.at(index));
  ir::Expr f1 = ir::For::Make(ir::Var("i"),
                              min,
                              extent,
                              ir::ForType::Serial,
                              ir::DeviceAPI::CUDA,
                              MakeFor(extents, name, index + 1),
                              ir::VectorizeInfo(),
                              ir::BindInfo());
  return f1;
}

TEST(ForInfo, ForInfoCheckerEqual) {
  ir::Expr* source;
  ir::Expr expr;
  ir::Expr f1;
  ir::Expr f2;

  f1 = MakeFor({10}, "f1", 0);
  f2 = MakeFor({10}, "f2", 0);
  expr = ir::Block::Make({f1, f2});
  source = &expr;
  EXPECT_TRUE(CanMergeBlocks(source, "f1", "f2"));

  f1 = MakeFor({10, 5}, "f1", 0);
  f2 = MakeFor({10, 5}, "f2", 0);
  expr = ir::Block::Make({f1, f2});
  source = &expr;
  EXPECT_TRUE(CanMergeBlocks(source, "f1", "f2"));

  f1 = MakeFor({10, 5, 3}, "f1", 0);
  f2 = MakeFor({10, 5, 3}, "f2", 0);
  expr = ir::Block::Make({f1, f2});
  source = &expr;
  EXPECT_TRUE(CanMergeBlocks(source, "f1", "f2"));
}

TEST(ForInfo, ForInfoCheckerNotEqual) {
  ir::Expr* source;
  ir::Expr expr;
  ir::Expr f1;
  ir::Expr f2;

  f1 = MakeFor({10}, "f1", 0);
  f2 = MakeFor({9}, "f2", 0);
  expr = ir::Block::Make({f1, f2});
  source = &expr;
  EXPECT_FALSE(CanMergeBlocks(source, "f1", "f2"));

  f1 = MakeFor({10, 5}, "f1", 0);
  f2 = MakeFor({10, 4}, "f2", 0);
  expr = ir::Block::Make({f1, f2});
  source = &expr;
  EXPECT_FALSE(CanMergeBlocks(source, "f1", "f2"));

  f1 = MakeFor({10, 5, 3}, "f1", 0);
  f2 = MakeFor({10, 5, 2}, "f2", 0);
  expr = ir::Block::Make({f1, f2});
  source = &expr;
  EXPECT_FALSE(CanMergeBlocks(source, "f1", "f2"));
}

}  // namespace

}  // namespace optim
}  // namespace cinn
