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

#include "paddle/infrt/host_context/op_executable.h"

#include <gtest/gtest.h>

#include "paddle/infrt/host_context/kernel_registry.h"
#include "paddle/infrt/host_context/kernel_utils.h"
#include "paddle/infrt/host_context/symbol_table.h"

namespace infrt {
namespace host_context {

int add(int a, int b) { return a + b; }

TEST(OpExecutable, basic) {
  // register kernel
  KernelRegistry registry;
  registry.AddKernel("infrt.test.add.i32", INFRT_KERNEL(add));

  SymbolTable table;
  table.Register("a", 1);
  table.Register("b", 2);

  OpExecutableBuilder executable("infrt.test.add.i32", &table, &registry);
  executable.AppendArgument("a");
  executable.AppendArgument("b");
  executable.SetResults({"c"});

  executable.Execute();

  // check the kernel frame has the result.
  auto results = executable.frame().GetResults();
  ASSERT_EQ(results.size(), 1UL);
  ASSERT_EQ(results.front()->get<int32_t>(), 3);

  // check symbol table contains the same result instance.
  LOG(INFO) << "type: " << table.GetValue("c")->type_info();
  int c = table.GetValue("c")->get<int32_t>();
  ASSERT_EQ(c, 3);
}

}  // namespace host_context
}  // namespace infrt
