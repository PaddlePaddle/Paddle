/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/framework/block.h"

#include <gtest/gtest.h>

namespace paddle {
namespace framework {

// there may be some complex init latter.
class SymbolTableTester : public ::testing::Test {
 protected:
  SymbolTable table_;
};

TEST_F(SymbolTableTester, NewOp) { ASSERT_NE(table_.NewOp(), nullptr); }

TEST_F(SymbolTableTester, NewVar) { ASSERT_NE(table_.NewVar("var1"), nullptr); }

TEST_F(SymbolTableTester, FindOp) {
  table_.NewOp();
  ASSERT_NE(table_.FindOp(0), nullptr);
}

TEST_F(SymbolTableTester, Compile) {
  table_.NewOp();
  table_.NewOp();
  table_.NewVar("var1");
  table_.NewVar("var2");

  auto block_desc = table_.Compile();
  ASSERT_EQ(block_desc.ops_size(), 2);
  ASSERT_EQ(block_desc.vars_size(), 2);
  ASSERT_TRUE(block_desc.vars(0).name() == "var1" ||
              block_desc.vars(1).name() == "var2");
}

}  // namespace framework
}  // namespace paddle
