// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/infrt/host_context/core_runtime.h"

#include <gtest/gtest.h>

#include "paddle/infrt/host_context/kernel_registry.h"
#include "paddle/infrt/host_context/kernel_utils.h"
#include "paddle/infrt/host_context/op_executable.h"
#include "paddle/infrt/host_context/symbol_table.h"

namespace infrt {
namespace host_context {

int add(int a, int b) { return a + b; }
int sub(int a, int b) { return a - b; }

TEST(CoreRuntime, basic) {
  KernelRegistry registry;
  registry.AddKernel("infrt.test.addi32", INFRT_KERNEL(add));
  registry.AddKernel("infrt.test.subi32", INFRT_KERNEL(sub));

  CoreRuntimeBuilder builder(&registry);
  auto* table = builder.symbol_table();
  table->Register("a", 1);
  table->Register("b", 2);
  table->Register("d", 4);

  // c = a + b
  auto* op0 = builder.NewOpExecutable("infrt.test.addi32");
  op0->AppendArgument("a");
  op0->AppendArgument("b");
  op0->SetResults({"c"});

  // e = c - d
  auto* op1 = builder.NewOpExecutable("infrt.test.subi32");
  op1->AppendArgument("c");
  op1->AppendArgument("d");
  op1->SetResults({"e"});

  builder.Execute();

  ASSERT_EQ(table->GetValue("d")->get<int>(), 4);
  ASSERT_EQ(table->GetValue("c")->get<int>(), 3);
  ASSERT_EQ(table->GetValue("e")->get<int>(), -1);
}

TEST(CoreRuntime, function) {
  // The function:
  // func(int a, int b) {
  //   int c = a + b
  //   return c
  // }
  KernelRegistry registry;
  registry.AddKernel("infrt.test.addi32", INFRT_KERNEL(add));
  registry.AddKernel("infrt.test.subi32", INFRT_KERNEL(sub));

  CoreRuntimeBuilder builder(&registry);
  auto* table = builder.symbol_table();

  std::vector<std::pair<std::string, ValueRef>> feeds{
      {std::make_pair("a", ValueRef(new Value(1))),  //
       std::make_pair("b", ValueRef(new Value(2)))}};
  builder.FeedInArgs(llvm::ArrayRef<std::pair<std::string, ValueRef>>(
      feeds.data(), feeds.size()));

  ASSERT_EQ(table->Get<int>("a"), 1);
  ASSERT_EQ(table->Get<int>("b"), 2);
  ASSERT_EQ(table->size(), 2UL);

  auto* op = builder.NewOpExecutable("infrt.test.addi32");
  op->AppendArgument("a");
  op->AppendArgument("b");
  op->SetResults({"c"});

  builder.Execute();

  auto res = builder.GetResults({"c"});
  ASSERT_EQ(res.size(), 1UL);
  ASSERT_EQ(res[0].get<int>(), 3);
}

}  // namespace host_context
}  // namespace infrt
