// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/memory_optimize_pass/reference_count_pass_helper.h"

#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/framework/var_desc.h"
#include "paddle/phi/common/place.h"

namespace paddle::framework::ir {

TEST(ReferenceCountPassHelperTest, TryGetLatestVarDescSkipsTrailingNullDesc) {
  VarDesc old_var_desc("x_old");
  VarDesc latest_var_desc("x_latest");

  std::unique_ptr<Node> old_node(CreateNodeForTest(&old_var_desc));
  std::unique_ptr<Node> latest_node(CreateNodeForTest(&latest_var_desc));
  std::unique_ptr<Node> trailing_empty_node(
      CreateNodeForTest("x_empty", Node::Type::kVariable));

  // VarHandleBase registers itself as the Node wrapper; Node owns the wrapper.
  auto* old_var_handle =
      new details::VarHandle(old_node.get(), 0, 0, "x", phi::CPUPlace());
  auto* latest_var_handle =
      new details::VarHandle(latest_node.get(), 1, 0, "x", phi::CPUPlace());
  auto* trailing_empty_var_handle = new details::VarHandle(
      trailing_empty_node.get(), 2, 0, "x", phi::CPUPlace());

  std::vector<details::VarHandle*> vars{
      old_var_handle, latest_var_handle, trailing_empty_var_handle};

  EXPECT_EQ(TryGetLatestVarDesc(vars)->Name(), "x_latest");
}

TEST(ReferenceCountPassHelperTest,
     TryGetLatestVarDescReturnsNullptrWhenAbsent) {
  std::unique_ptr<Node> empty_node(
      CreateNodeForTest("x_empty", Node::Type::kVariable));
  // VarHandleBase registers itself as the Node wrapper; Node owns the wrapper.
  auto* empty_var_handle =
      new details::VarHandle(empty_node.get(), 0, 0, "x", phi::CPUPlace());

  std::vector<details::VarHandle*> vars{empty_var_handle};

  EXPECT_EQ(TryGetLatestVarDesc(vars), nullptr);
}

}  // namespace paddle::framework::ir
